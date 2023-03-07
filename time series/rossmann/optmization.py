import warnings
import traceback
import dask
import numpy as np
import pandas as pd
import time
from numpy.linalg import LinAlgError
from numpy.linalg import cholesky, det, lstsq, inv
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.gaussian_process import GaussianProcessRegressor as GPR, kernels as gpk
from sklearn.metrics.pairwise import check_pairwise_arrays
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import optimization.numpy_version.single_objective.continuous as co
from dateutil.relativedelta import relativedelta
import datetime

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class SpectralMixture(gpk.Kernel):
    def __init__(self, q, w, m, v, d, active_dim=None):
        self.q, self.w, self.m, self.v, self.d = q, w, m, v, d
        self.active_dim = active_dim

    @property
    def anisotropic(self):
        return False

    @property
    def hyperparameter_variance(self):
        return gpk.Hyperparameter("v", "numeric", self.v.ravel(), len(self.v.ravel()))

    @property
    def hyperparameter_mean(self):
        return gpk.Hyperparameter("m", "numeric", self.m.ravel(), len(self.m.ravel()))

    @property
    def hyperparameter_weight(self):
        return gpk.Hyperparameter("w", "numeric", self.w.ravel(), len(self.w.ravel()))

    def __call__(self, X, Y=None, eval_gradient=False):
        w, m, v = self.w[:, np.newaxis], np.reshape(self.m, (self.d, self.q)), np.reshape(self.v, (self.d, self.q))
        assert w.shape == (q, 1), 'Weights must be [q x 1]'
        assert m.shape[1] == q
        assert v.shape[1] == q
        X = np.atleast_2d(X)
        X = X[:, self.active_dim] if self.active_dim is not None else X
        if Y is None:
            Y = X
        else:
            Y = np.atleast_2d(Y)
            Y = Y[:, self.active_dim] if self.active_dim is not None else Y
        tau = X[:, np.newaxis, :] - Y

        # tau(m,n,p) tensordot means(p,q) -> dot_prod(m,n,q)
        # where dot_prod[i,j,k] = tau[i,j]'*means[:,k]
        K = np.cos(2 * np.pi * np.tensordot(tau, m, axes=1)) * \
            np.exp(-2 * np.pi ** 2 * np.tensordot(tau ** 2, v, axes=1))

        # return the weighted sum of the individual
        # Gaussian kernels, dropping the third index
        return np.tensordot(K, w, axes=1).squeeze(axis=(2,))

    def diag(self, X):
        return np.diag(self(X))

    def is_stationary(self):
        """Returns whether the kernel is stationary. """
        return True

    def __repr__(self):
        return "{0}(weight=[{1}], mean=[{2}], variance=[{3}])".format(
            self.__class__.__name__, ", ".join(map("{0:.3g}".format, self.w)),
            ", ".join(map("{0:.3g}".format, self.m)), ", ".join(map("{0:.3g}".format, self.v)))


class Polynomial(gpk.Kernel):

    def __init__(self, variance=1.0, offset=0.0, degree=1.0, active_dim=None):
        self.degree = degree
        self.variance = variance
        self.offset = offset
        self.active_dim = active_dim
        if active_dim is not None and self.anisotropic:
            assert len(self.active_dim) == len(self.variance), 'variance and active_dim must have the same length'

    @property
    def anisotropic(self):
        return np.iterable(self.variance) and len(self.variance) > 1

    @property
    def hyperparameter_periodicity(self):
        return gpk.Hyperparameter("degree", "numeric", self.degree)

    @property
    def hyperparameter_periodicity(self):
        return gpk.Hyperparameter("offset", "numeric", self.offset)

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return gpk.Hyperparameter("variance", "numeric", self.variance, len(self.variance))
        return gpk.Hyperparameter("variance", "numeric", self.variance)

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        X = X[:, self.active_dim] if self.active_dim is not None else X
        if Y is None:
            return (np.matmul(X * self.variance, X.T) + self.offset) ** self.degree
        else:
            Y = np.atleast_2d(Y)
            Y = Y[:, self.active_dim] if self.active_dim is not None else Y
            return (np.tensordot(X * self.variance, Y, [[-1], [-1]]) + self.offset) ** self.degree

    def diag(self, X):
        return np.diag(self(X))

    def is_stationary(self):
        """Returns whether the kernel is stationary. """
        return False

    def __repr__(self):
        if self.anisotropic:
            return "{0}(variance=[{1}], offset={2:.3g}, degree={3:.3g})".format(
                self.__class__.__name__, ", ".join(map("{0:.3g}".format, self.variance)), self.offset, self.degree)
        else:  # isotropic
            return "{0}(variance={1:.3g}, offset={2:.3g}, degree={3:.3g})".format(
                self.__class__.__name__, self.variance, self.offset, self.degree)


class Brownian(gpk.Kernel):

    def __init__(self, variance=1.0, active_dim=None):
        if len(active_dim) != 1:
            raise ValueError("Input dimensional for Brownian kernel must be 1.")
        self.variance = variance
        self.active_dim = active_dim

    @property
    def hyperparameter_variance(self):
        return gpk.Hyperparameter("variance", "numeric", self.variance)

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        X = X[:, self.active_dim] if self.active_dim is not None else X
        if Y is None:
            Y = X
        else:
            Y = np.atleast_2d(Y)
            Y = Y[:, self.active_dim] if self.active_dim is not None else Y

        return np.where(np.sign(X) == np.sign(Y.T), self.variance * np.fmin(np.abs(X), np.abs(Y.T)), 0.)

    def diag(self, X):
        return np.diag(self(X))

    def is_stationary(self):
        """Returns whether the kernel is stationary. """
        return False

    def __repr__(self):
        return "{0}(variance={1:.3g})".format(self.__class__.__name__, self.variance)


class ArcCosine(gpk.Kernel):
    implemented_orders = {0, 1, 2}

    def __init__(self, order=0, variance=1.0, weight_variances=1.0, bias_variance=1.0, active_dim=None):
        if order not in self.implemented_orders:
            raise ValueError("Requested kernel order is not implemented.")
        self.order = order
        self.variance = variance
        self.bias_variance = bias_variance
        self.weight_variances = weight_variances
        self.active_dim = active_dim
        if active_dim is not None and self.anisotropic:
            assert len(self.active_dim) == len(
                self.weight_variances), 'weight_variances and active_dim must have the same length'

    @property
    def anisotropic(self):
        return np.iterable(self.weight_variances) and len(self.weight_variances) > 1

    @property
    def hyperparameter_variance(self):
        return gpk.Hyperparameter("variance", "numeric", self.variance)

    @property
    def hyperparameter_weight_variances(self):
        if self.anisotropic:
            return gpk.Hyperparameter("weight_variances", "numeric", self.weight_variances, len(self.weight_variances))
        return gpk.Hyperparameter("weight_variances", "numeric", self.weight_variances)

    @property
    def hyperparameter_bias_variance(self):
        return gpk.Hyperparameter("bias_variance", "numeric", self.bias_variance)

    def _weighted_product(self, X, X2=None):
        if X2 is None:
            return np.sum(self.weight_variances * X ** 2, axis=1) + self.bias_variance
        return np.matmul((self.weight_variances * X), X2.T) + self.bias_variance

    def _J(self, theta):
        """
        Implements the order dependent family of functions defined in equations
        4 to 7 in the reference paper.
        """
        if self.order == 0:
            return np.pi - theta
        elif self.order == 1:
            return np.sin(theta) + (np.pi - theta) * np.cos(theta)
        elif self.order == 2:
            return 3.0 * np.sin(theta) * np.cos(theta) + (np.pi - theta) * (
                    1.0 + 2.0 * np.cos(theta) ** 2)

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        X = X[:, self.active_dim] if self.active_dim is not None else X
        X_denominator = np.sqrt(self._weighted_product(X))
        if Y is None:
            Y = X
            Y_denominator = X_denominator
        else:
            Y = np.atleast_2d(Y)
            Y = Y[:, self.active_dim] if self.active_dim is not None else Y
            Y_denominator = np.sqrt(self._weighted_product(Y))

        numerator = self._weighted_product(X, Y)
        cos_theta = numerator / X_denominator[:, None] / Y_denominator[None, :]
        jitter = 1e-15
        theta = np.arccos(jitter + (1 - 2 * jitter) * cos_theta)

        return self.variance * (1.0 / np.pi) * self._J(theta) * X_denominator[:, None] ** self.order * Y_denominator[
                                                                                                       None,
                                                                                                       :] ** self.order

    def diag(self, X):
        return np.diag(self(X))

    def is_stationary(self):
        """Returns whether the kernel is stationary. """
        return False

    def __repr__(self):
        if self.anisotropic:
            return "{0}(variance={1:.3g}, weight_variances=[{2}], bias_variance={3:.3g})".format(
                self.__class__.__name__, self.variance, ", ".join(map("{0:.3g}".format, self.weight_variances)),
                self.bias_variance)
        else:  # isotropic
            return "{0}(variance={1:.3g}, weight_variances={2:.3g}, bias_variance={2:.3g})".format(
                self.__class__.__name__, self.variance, self.weight_variances, self.bias_variance)


class Gibbs(gpk.Kernel):

    def __init__(self, lfunc, args, active_dim=None):
        self.lfunc = lfunc
        self.args = args
        self.active_dim = active_dim

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        X = X[:, self.active_dim] if self.active_dim is not None else X
        rx = self.lfunc(X, **self.args)
        if Y is None:
            rz = self.lfunc(X, **self.args)
            dists = squareform(pdist(X, metric='sqeuclidean'))
            np.fill_diagonal(dists, 1)
        else:
            Y = np.atleast_2d(Y)
            Y = Y[:, self.active_dim] if self.active_dim is not None else Y
            rz = self.lfunc(Y, **self.args)
            dists = cdist(X, Y, metric='sqeuclidean')

        rx2, rz2 = np.reshape(rx ** 2, (-1, 1)), np.reshape(rz ** 2, (1, -1))
        return np.sqrt((2.0 * np.outer(rx, rz)) / (rx2 + rz2)) * np.exp(-1.0 * dists / (rx2 + rz2))

    def diag(self, X):
        return np.alloc(1.0, X.shape[0])

    def is_stationary(self):
        """Returns whether the kernel is stationary. """
        return False

    def __repr__(self):
        if self.anisotropic:
            return "{0}".format(self.__class__.__name__)


class WarpedInput(gpk.Kernel):

    def __init__(self, stationary, func, args, active_dim=None):
        self.stationary = stationary
        self.func = func
        self.args = args
        self.active_dim = active_dim

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        X = X[:, self.active_dim] if self.active_dim is not None else X
        X = self.func(X, **self.args)
        if Y is not None:
            Y = np.atleast_2d(Y)
            Y = Y[:, self.active_dim] if self.active_dim is not None else Y
            Y = self.func(Y, **self.args)

        return self.stationary(X, Y, eval_gradient)

    def diag(self, X):
        return np.diag(self(X))

    def is_stationary(self):
        """Returns whether the kernel is stationary. """
        return False

    def __repr__(self):
        return ''


class Gabor(gpk.Kernel):

    def __init__(self, stationary, length_scale=1.0, periodicity=1.0, active_dim=None):
        self.stationary = stationary
        self.length_scale = length_scale
        self.periodicity = periodicity
        self.active_dim = active_dim
        if active_dim is not None and self.anisotropic:
            assert len(self.active_dim) == len(
                self.length_scale), 'length_scale and active_dim must have the same length'

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_periodicity(self):
        return gpk.Hyperparameter("periodicity", "numeric", self.periodicity)

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return gpk.Hyperparameter("length_scale", "numeric", self.length_scale, len(self.length_scale))
        return gpk.Hyperparameter("length_scale", "numeric", self.length_scale)

    def __call__(self, X, Y=None, eval_gradient=False):
        stationary = self.stationary(length_scale=self.length_scale)
        X = np.atleast_2d(X)
        X = X[:, self.active_dim] if self.active_dim is not None else X
        if Y is None:
            dists = squareform(pdist(X / self.length_scale, metric='sqeuclidean'))
            np.fill_diagonal(dists, 1)
            tmp1 = stationary(X, Y, eval_gradient)
        else:
            Y = np.atleast_2d(Y)
            Y = Y[:, self.active_dim] if self.active_dim is not None else Y
            dists = cdist(X / self.length_scale, Y / self.length_scale, metric='sqeuclidean')
            tmp1 = stationary(X, Y, eval_gradient)

        tmp2 = 2 * np.pi * np.sqrt(dists) * self.length_scale / self.periodicity
        return tmp1 * np.cos(tmp2)

    def diag(self, X):
        return np.diag(self(X))

    def is_stationary(self):
        """Returns whether the kernel is stationary. """
        return True

    def __repr__(self):
        if self.anisotropic:
            return "{0}(length_scale=[{1}], periodicity={2:.3g})".format(
                self.__class__.__name__, ", ".join(map("{0:.3g}".format, self.length_scale)), self.periodicity)
        else:  # isotropic
            return "{0}(length_scale={1:.3g}, periodicity={2:.3g})".format(
                self.__class__.__name__, self.length_scale, self.periodicity)


class ConstantKernel(gpk.ConstantKernel):
    def __init__(self, constant_value=1.0, constant_value_bounds=(1e-5, 1e5), active_dim=None):
        super().__init__(constant_value=constant_value, constant_value_bounds=constant_value_bounds)
        self.active_dim = active_dim

    def __call__(self, X, Y=None, eval_gradient=False):
        if self.active_dim == None:
            return super().__call__(X, Y, eval_gradient)
        else:
            X = np.atleast_2d(X)
            X = X[:, self.active_dim]
            if Y is not None:
                Y = np.atleast_2d(Y)
                Y = Y[:, self.active_dim]
            return super().__call__(X, Y, eval_gradient)


class Matern(gpk.Matern):
    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5), nu=1.5, active_dim=None):
        super().__init__(length_scale=length_scale, length_scale_bounds=length_scale_bounds, nu=nu)
        self.active_dim = active_dim
        if active_dim is not None and self.anisotropic:
            assert len(self.active_dim) == len(
                self.length_scale), 'weight_variances and active_dim must have the same length'

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    def __call__(self, X, Y=None, eval_gradient=False):
        if self.active_dim == None:
            return super().__call__(X, Y, eval_gradient)
        else:
            X = np.atleast_2d(X)
            X = X[:, self.active_dim]
            if Y is not None:
                Y = np.atleast_2d(Y)
                Y = Y[:, self.active_dim]
            return super().__call__(X, Y, eval_gradient)


class RationalQuadratic(gpk.RationalQuadratic):
    def __init__(self, length_scale=1.0, alpha=1.0, length_scale_bounds=(1e-05, 100000.0),
                 alpha_bounds=(1e-05, 100000.0),
                 active_dim=None):
        super().__init__(length_scale=length_scale, length_scale_bounds=length_scale_bounds, alpha=alpha,
                         alpha_bounds=alpha_bounds)
        self.active_dim = active_dim
        if active_dim is not None and self.anisotropic:
            assert len(self.active_dim) == len(
                self.length_scale), 'weight_variances and active_dim must have the same length'

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    def __call__(self, X, Y=None, eval_gradient=False):
        if self.active_dim == None:
            return super().__call__(X, Y, eval_gradient)
        else:
            X = np.atleast_2d(X)
            X = X[:, self.active_dim]
            if Y is not None:
                Y = np.atleast_2d(Y)
                Y = Y[:, self.active_dim]
            return super().__call__(X, Y, eval_gradient)


class RBF(gpk.RBF):
    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5), active_dim=None):
        super().__init__(length_scale=length_scale, length_scale_bounds=length_scale_bounds)
        self.active_dim = active_dim
        if active_dim is not None and self.anisotropic:
            assert len(self.active_dim) == len(
                self.length_scale), 'weight_variances and active_dim must have the same length'

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    def __call__(self, X, Y=None, eval_gradient=False):
        if self.active_dim == None:
            return super().__call__(X, Y, eval_gradient)
        else:
            X = np.atleast_2d(X)
            X = X[:, self.active_dim]
            if Y is not None:
                Y = np.atleast_2d(Y)
                Y = Y[:, self.active_dim]
            return super().__call__(X, Y, eval_gradient)


class ExpSineSquared(gpk.ExpSineSquared):
    def __init__(self, length_scale=1.0, periodicity=1.0, length_scale_bounds=(1e-5, 1e5),
                 periodicity_bounds=(1e-5, 1e5), active_dim=None):
        super().__init__(length_scale=length_scale, periodicity=periodicity, length_scale_bounds=length_scale_bounds,
                         periodicity_bounds=periodicity_bounds)
        self.active_dim = active_dim
        if active_dim is not None and self.anisotropic:
            assert len(self.active_dim) == len(
                self.length_scale), 'weight_variances and active_dim must have the same length'

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    def __call__(self, X, Y=None, eval_gradient=False):
        if self.active_dim == None:
            return super().__call__(X, Y, eval_gradient)
        else:
            X = np.atleast_2d(X)
            X = X[:, self.active_dim]
            if Y is not None:
                Y = np.atleast_2d(Y)
                Y = Y[:, self.active_dim]
            return super().__call__(X, Y, eval_gradient)


class WhiteKernel(gpk.WhiteKernel):
    def __init__(self, noise_level=1.0, noise_level_bounds=(1e-05, 100000.0), active_dim=None):
        super(WhiteKernel, self).__init__(noise_level=noise_level, noise_level_bounds=noise_level_bounds)
        self.active_dim = active_dim

    def __call__(self, X, Y=None, eval_gradient=False):
        if self.active_dim == None:
            return super().__call__(X, Y, eval_gradient)
        else:
            X = np.atleast_2d(X)
            X = X[:, self.active_dim]
            if Y is not None:
                Y = np.atleast_2d(Y)
                Y = Y[:, self.active_dim]
            return super().__call__(X, Y, eval_gradient)

def mape(test_y, pred):
    return np.round(np.mean(np.abs(100*(test_y-pred)/test_y)), 0)

def create_kernel(hp):
    poly_comp = RBF(hp[0:len(f)], active_dim=f) * ConstantKernel(hp[len(f)], active_dim=f)
    open_comp = RBF(hp[len(f) + 1], active_dim=[o]) * ConstantKernel(hp[len(f) + 2], active_dim=[o])
    week_comp = ExpSineSquared(hp[len(f) + 3], periodicity=7, active_dim=[t]) * ConstantKernel(hp[len(f) + 4], active_dim=[t])
    year_comp = ExpSineSquared(hp[len(f) + 5], periodicity=364, active_dim=[t]) * ConstantKernel(hp[len(f) + 6], active_dim=[t])
    trend_comp = Polynomial(hp[len(f) + 7], hp[len(f) + 8], hp[len(f) + 10], active_dim=[t])
    return (poly_comp + year_comp + week_comp + trend_comp + gpk.WhiteKernel(hp[len(f) + 9])) * open_comp


def _create_kernel(hp):
    poly_comp = RBF(hp[0:len(f)], active_dim=f) * ConstantKernel(hp[len(f)], active_dim=f)
    open_comp = RBF(hp[len(f) + 1], active_dim=[o]) * ConstantKernel(hp[len(f) + 2], active_dim=[o])
    week_comp = ExpSineSquared(hp[len(f) + 3], periodicity=7, active_dim=[t]) * ConstantKernel(hp[len(f) + 4], active_dim=[t])
    year_comp = ExpSineSquared(hp[len(f) + 5], periodicity=364, active_dim=[t]) * ConstantKernel(hp[len(f) + 6], active_dim=[t])
    trend_comp = ArcCosine(1, hp[len(f) + 7], hp[len(f) + 8], hp[len(f) + 9], active_dim=[t])
    return (poly_comp + year_comp + week_comp + trend_comp + gpk.WhiteKernel(hp[len(f) + 10])) * open_comp

df = pd.read_csv('/home/skyolia/JupyterProjects/data/time_series/rossmann/rossmann.csv')
df = df.loc[df['Store'] == 1045]
df = df.drop(columns=[col for col in df.columns if len(df[col].unique()) == 1])
df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")

output_col = 'Sales'
df['delta_t'] = (df['Date'] - df['Date'].iloc[0])/np.timedelta64(1,'D')

train = df[df['Date'] < '2015-06-14']
test = df[df['Date']>='2015-06-14']

OE = OneHotEncoder(sparse=False)
train_ohe = OE.fit_transform(train[['StateHoliday']])
test_ohe = OE.transform(test[['StateHoliday']])
for i, c in enumerate(OE.categories_[0]):
    train['StateHoliday_'+str(c)] = train_ohe[:, i]
    test['StateHoliday_'+str(c)] = test_ohe[:, i]
train.drop(columns=['StateHoliday'], inplace=True)
test.drop(columns=['StateHoliday'], inplace=True)

y_scaler = MinMaxScaler(feature_range=(0, 1))
train_y, test_y = y_scaler.fit_transform(train[[output_col]]).ravel(), test[output_col].values
train_idx, test_idx = train['Date'].tolist(), test['Date'].tolist()
train_x, test_x = train.drop(columns=['Date',output_col]), test.drop(columns=['Date',output_col])
print(train_x.shape)
print(train_y.shape)
c = np.asarray([0.5, 1.5, 2.5])
n = train_x.shape[1]
f = list(range(n))
t, o = train_x.columns.get_loc("delta_t"), train_x.columns.get_loc("Open")
f.remove(t), f.remove(o)

#################### MLE ##########################
def sk_nll_stable(hp):
    done = False
    while not done:
        try:
            #hp[n+7], hp[n+8] = np.round(hp[n+7]).astype(int), c[np.argmin(np.abs(c - hp[n+8]))]#,
            hp[len(f)+10] = np.round(hp[len(f)+10]).astype(int)
            kernel = create_kernel(hp)
            gpr = GPR(kernel=kernel, optimizer=None).fit(train_x, train_y)
            done = True
            return hp, -1 * gpr.log_marginal_likelihood_value_
        except (LinAlgError, ValueError):
            #traceback.print_exc()
            hp = np.random.uniform(lb, ub, dim)


def mle_objf(pop, dim=None):
    population, fitness = [], []
    s = time.time()
    for hp in pop:
        _hp, mse = dask.delayed(sk_nll_stable, nout=2)(hp)
        population.append(_hp), fitness.append(mse)
    population, fitness = dask.compute(population, fitness)
    population, fitness = np.asarray(population), np.asarray(fitness)
    e = time.time()
    print('DURATION = ', e-s)
    return population, fitness  # duration =  27.99079418182373

####################### TCSV ###########################
tscv = TimeSeriesSplit(n_splits=5)

def nv_lossf(true, pred):
    loss = 500 * np.clip(true - pred, a_min=0, a_max=None) + 160 * np.clip(pred - true, a_min=0, a_max=None)
    return np.mean(loss)

nv_loss = make_scorer(nv_lossf, greater_is_better=False)

def inference_pop(hp):
    lost, done = [], False
    while not done:
        try:
            hp[n + 7] = c[np.argmin(np.abs(c - hp[-1]))]  # , np.round(hp[-2]).astype(int)#
            kernel = create_kernel(hp)
            gpr = GPR(kernel=kernel, optimizer=None)
            score = cross_val_score(gpr, train_x, train_y, cv=tscv, scoring=nv_loss,).mean()
            done = True
            return hp, -score
        except (LinAlgError, ValueError):
            hp = np.random.uniform(lb, ub, dim)

def cv_objf(pop, dim=None):
    population, fitness = [], []
    for hp in pop:
        _hp, mse = dask.delayed(inference_pop, nout=2)(hp)
        population.append(_hp), fitness.append(mse)
    population, fitness = dask.compute(population, fitness)
    population, fitness = np.asarray(population), np.asarray(fitness)
    return population, fitness  # duration =  27.99079418182373

def ccf(population):
    return population

dim = len(f) + 11#n + 8#len(not_t) * 3 + 4 #len(not_t) * 4 + 4 #
lb, ub = 1e-5, 1e3
_lb, _ub = np.full(dim - 1, lb), np.full(dim - 1, ub)
lb, ub = np.hstack((_lb, [1])), np.hstack((_ub, [5]))

'''_lb, _ub = np.full((len(not_t), 4), (lb, 0, lb, lb)).ravel(), np.full((len(not_t), 4), (ub, 3, ub, ub)).ravel()
lb, ub = np.hstack((_lb, [lb, lb, lb, lb])), np.hstack((_ub, [ub, ub, ub, ub]))'''

s = time.time()
optimizer = co.CPSO(mle_objf, ccf, dim, lb, ub, 50, 5000)
e = time.time()
print('duration = ', e - s)
print(optimizer.gbest)
