import numpy as np
from sklearn.gaussian_process import kernels as gpk

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