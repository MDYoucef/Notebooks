from metaflow import FlowSpec, step, IncludeFile, Parameter, JSONType
import pandas as pd
import numpy as np
from scipy.stats import lognorm, norm
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.metrics import mean_absolute_error
from kernels import ArcCosine, Matern, ConstantKernel, ExpSineSquared, WhiteKernel, RBF, Polynomial
from joblib import load
import os
from io import StringIO


def date_encoding(train, test, time_col):
    start_date = train[time_col].iloc[0]
    train['delta_t'] = (train[time_col] - start_date) / np.timedelta64(1, 'D')
    test['delta_t'] = (test[time_col] - start_date) / np.timedelta64(1, 'D')
    train['norm_delta_t'] = train['delta_t']
    test['norm_delta_t'] = test['delta_t']
    return train, test, start_date


def categorical_encoding(train, test, categorical):
    new_cat, OHEncoders = [], {}
    for cat in categorical:
        OE = OneHotEncoder(sparse=False, drop='if_binary')
        train_ohe = OE.fit_transform(train[[cat]])
        test_ohe = OE.transform(test[[cat]].astype(str))
        for i in range(train_ohe.shape[1]):
            c = OE.categories_[0][i]
            train[cat + '_' + str(c)] = train_ohe[:, i]
            test[cat + '_' + str(c)] = test_ohe[:, i]
            new_cat.append(cat + '_' + str(c))
        OHEncoders[cat] = OE
    return train, test, OHEncoders, new_cat


def numerical_scaling(train, test, numerical):
    MS = MinMaxScaler(feature_range=(0, 1))
    scaled_train = MS.fit_transform(train[numerical])
    scaled_test = MS.transform(test[numerical])
    train[numerical] = scaled_train
    test[numerical] = scaled_test
    return train, test, MS


def output_scaling(train, test, output_col):
    YScaler = MinMaxScaler(feature_range=(0, 1))
    Y_train = YScaler.fit_transform(train[[output_col]]).ravel() + 1e-15
    Y_test = test[output_col]
    return Y_train, Y_test, YScaler


def final_features(X_train, new_cat, numerical, binary):
    features = new_cat + numerical + binary + ['norm_delta_t', 'delta_t']
    unique_col = [col for col in features if len(pd.unique(X_train[col])) == 1]
    return [col for col in features if col not in unique_col]

def shift_df(df, shift, dropna=True):
    origin = df.copy()
    for i in range(1, shift+1):
        shifted_df = origin.shift(i)
        shifted_df = shifted_df.rename(columns=dict(zip(shifted_df.columns, [str(c)+'_'+str(i) for c in shifted_df.columns])))
        df = pd.concat([shifted_df, df], axis=1)
    return df.dropna() if dropna else df


def get_explaination(expl_kernels, X_train, Y_train, X_test):
    mus = []
    for i in expl_kernels:
        k = expl_kernels[i]
        gpr = GPR(kernel=k, optimizer=None, alpha=1e-5).fit(X_train, Y_train)
        mus.append(gpr.predict(X_test, return_std=False))
    return mus


def get_output(kernel, X_train, Y_train, X_test, mus, scaler):
    gpr = GPR(kernel=kernel, optimizer=None).fit(X_train, Y_train)
    mu_test, std_test = gpr.predict(X_test, return_std=True)
    lb, ub = norm.ppf(0.025, mu_test, std_test), norm.ppf(0.975, mu_test, std_test)
    mus = np.stack(mus)
    pred = scaler.inverse_transform(np.concatenate((mu_test[None, :], lb[None, :], ub[None, :], mus)))
    invt_train = scaler.inverse_transform(Y_train[:, np.newaxis]).ravel()
    return gpr, mu_test, std_test, pred, invt_train


def mape(true, pred):
    return np.round(np.mean(np.abs(100 * (true - pred) / (true + 1e-9))), 0)


class TrainTestFlow(FlowSpec):
    history = IncludeFile("history", help="The path to train file", default='history.csv')
    models_folder = Parameter("models_folder", type=str, default='models/')
    output_col = Parameter("output_col", type=str, default="Sales")
    time_col = Parameter("time_col", type=str, default="Date")
    sku_col = Parameter("sku_col", type=str, default="Store")
    categorical = Parameter("categorical", type=JSONType, default=[])
    binary = Parameter("binary", type=JSONType, default=[])
    old_numerical = Parameter("old_numerical", type=JSONType, default=['TV', 'Radio', 'Banners'])
    numerical = Parameter("numerical", type=JSONType, default=['TV_3','Radio_3','Banners_3','TV_2','Radio_2','Banners_2',
                                                               'TV_1','Radio_1','Banners_1','TV','Radio','Banners'])
    to_remove = Parameter("to_remove", type=JSONType, default=[])
    split_point = Parameter("split_point", type=str, default='2021-04-30')
    shift = Parameter("shift", type=int, default=3)

    @step
    def start(self):
        self.df = pd.read_csv(StringIO(self.history))
        self.df[self.time_col] = pd.to_datetime(self.df[self.time_col], format="%Y-%m-%d")
        self.skus = self.df[self.sku_col].unique().tolist()
        self.train = self.df[self.df[self.time_col] < self.split_point]
        self.test = self.df[self.df[self.time_col] >= self.split_point]
        self.next(self.build_shifted_df, foreach="skus")
        
    @step
    def build_shifted_df(self):
        self.sku = self.input
        self.df = self.df[self.df[self.sku_col] == self.sku]
        shifted_df = shift_df(self.df[self.old_numerical], self.shift, False)
        self.df = pd.concat((self.df[[self.time_col, self.sku_col] + self.to_remove + self.categorical + self.binary + [self.output_col]], shifted_df),1).dropna()
        self.next(self.join_build_shifted_df)
        
    @step
    def join_build_shifted_df(self, inputs):
        self.df = pd.concat([inp.df for inp in inputs])
        self.merge_artifacts(inputs)  # MERGING TO PROPAGATE SKUS VARIABLE
        self.next(self.feature_encoding_scaling__)
        
    @step
    def feature_encoding_scaling__(self):
        self.next(self.feature_encoding_scaling, foreach="skus")
        

    @step
    def feature_encoding_scaling(self):
        self.sku = self.input
        self.df = self.df[self.df[self.sku_col] == self.sku]
        train = self.df[self.df[self.time_col] < self.split_point]
        test = self.df[self.df[self.time_col] >= self.split_point]
        self.unused_col = [col for col in train.columns if len(train[col].unique()) == 1] + self.to_remove
        self.raw_train, self.raw_test = train.copy(), test.copy()
        train, test, self.start_date = date_encoding(train, test, self.time_col)  # DATE ENCODING
        train, test, self.OHEncoders, new_cat = categorical_encoding(train, test, self.categorical)  # CATEGORICAL ENCODING
        if len(self.numerical) > 0:
            train, test, self.MS = numerical_scaling(train, test, self.numerical + ['norm_delta_t'])  # NUMERCIAL SCALING
        self.features = final_features(train, new_cat, self.numerical, self.binary)  # REMOVE COLUMNS WITH UNIQUE VALUE
        self.X_train, self.T_train = train[self.features], train[self.time_col]
        self.X_test, self.T_test = test[self.features], test[self.time_col]
        self.Y_train, self.Y_test, self.YScaler = output_scaling(train, test, self.output_col)
        self.next(self.join_feature_encoding_scaling)

    @step
    def join_feature_encoding_scaling(self, inputs):
        sku_idx = [inp.sku for inp in inputs]
        features = dict(zip(sku_idx, [inp.features for inp in inputs]))
        unused_col = dict(zip(sku_idx, [inp.unused_col for inp in inputs]))
        start_date = dict(zip(sku_idx, [inp.start_date for inp in inputs]))
        raw = dict(zip(sku_idx, [(inp.raw_train, inp.raw_test) for inp in inputs]))
        X_train = dict(zip(sku_idx, [inp.X_train for inp in inputs]))
        T_train = dict(zip(sku_idx, [inp.T_train for inp in inputs]))
        X_test = dict(zip(sku_idx, [inp.X_test for inp in inputs]))
        T_test = dict(zip(sku_idx, [inp.T_test for inp in inputs]))
        OHEncoders = dict(zip(sku_idx, [inp.OHEncoders for inp in inputs]))
        MMScalers = dict(zip(sku_idx, [inp.MS for inp in inputs])) if len(self.numerical) > 0 else None
        Y_train = dict(zip(sku_idx, [inp.Y_train for inp in inputs]))
        Y_test = dict(zip(sku_idx, [inp.Y_test for inp in inputs]))
        YScaler = dict(zip(sku_idx, [inp.YScaler for inp in inputs]))
        self.flow_res = dict(
            zip(['Raw', 'Features', 'Unused', 'start_date', 'X_train', 'T_train', 'X_test', 'T_test', 'Y_train', 'Y_test',
                 'OHEncoders', 'MMScalers', 'YS'],
                [raw, features, unused_col, start_date, X_train, T_train, X_test, T_test, Y_train, Y_test, OHEncoders, MMScalers,
                 YScaler]))
        self.merge_artifacts(inputs, include=['flow_res', 'skus'])  # MERGING TO PROPAGATE SKUS VARIABLE
        self.next(self.models_training)

    @step
    def models_training(self):
        self.next(self.models_training__, foreach="skus")

    @step
    def models_training__(self):
        self.sku = self.input
        kernel_path = os.path.join(os.path.dirname(__file__), self.models_folder + str(self.sku) + '.joblib')
        self.kernels_d, mus = load(kernel_path), []
        kernel, self.expl_kernels = self.kernels_d['Kernel'], {k: v for k, v in self.kernels_d.items() if k != 'Kernel'}
        X_train, Y_train = self.flow_res['X_train'][self.sku], self.flow_res['Y_train'][self.sku]
        X_test, Y_test = self.flow_res['X_test'][self.sku], self.flow_res['Y_test'][self.sku]
        y_scaler = self.flow_res['YS'][self.sku]
        mus = get_explaination(self.expl_kernels, X_train, Y_train, X_test)  # EXPLAINATION
        self.gpr, self.mu_test, self.std_test, self.pred, self.Y_train = get_output(kernel, X_train, Y_train, X_test,
                                                                                    mus, y_scaler)  # INFERENCE
        self.mae, self.mape_ = mean_absolute_error(Y_test, self.pred[0]), mape(Y_test, self.pred[0])  # EVALUATION
        self.mle = self.gpr.log_marginal_likelihood_value_
        self.next(self.training_join)

    @step
    def training_join(self, inputs):
        self.merge_artifacts(inputs, include=['flow_res', 'skus'])  # MERGING TO PROPAGATE SKUS VARIABLE
        sku_idx = [inp.sku for inp in inputs]
        errors = dict(zip(sku_idx, [(inp.mle, inp.mae, inp.mape_) for inp in inputs]))
        gp_output = dict(zip(sku_idx, [(inp.mu_test, inp.std_test) for inp in inputs]))
        pred = dict(zip(sku_idx, [inp.pred for inp in inputs]))
        expl_kernels = dict(zip(sku_idx, [inp.expl_kernels for inp in inputs]))
        gprs = dict(zip(sku_idx, [inp.gpr for inp in inputs]))
        Y_train = dict(zip(sku_idx, [inp.Y_train for inp in inputs]))
        tmp = {'GPRS': gprs, 'Expl': expl_kernels, 'Errors': errors, 'GPOut': gp_output, 'Pred': pred,
               'Y_train': Y_train}
        self.flow_res = {**self.flow_res, **tmp}
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == '__main__':
    TrainTestFlow()