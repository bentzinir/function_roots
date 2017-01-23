import numpy as np
from numpy.linalg import norm
from scipy.optimize import fsolve


class FunctionFamily(object):
    def __init__(self, function_type, data, tol=1e-4):
        self.f = self._create_function_object()
        self.type = function_type

    def _create_function_object(self):
        if self.type == 'POLYNOMIAL':
            return np.asarray([np.dot(params, np.asarray([x[0] ** 2, x[0], 1])), 0])
        elif self.type == 'TRIGONOMETRIC':
            # return -params[0] + params[1]*np.sin(x[0]) + params[2]*np.sin(x[1]), 0
            return np.asarray([-params[0] + params[1] * np.square(x[0]) + params[2] * np.abs(x[1]), 0])

    def valid_root(self, x, f, params, tol=1e-4):
        y = f(x, params)  # y is ndarray type
        if norm(y) < tol:
            return True
        else:
            return False

    def create_data(self, n, f, tol):
        data = []
        i = 0
        while i < n:
            roots = fsolve(f, x0=np.random.normal(size=x_dim), args=params)
