import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from blox2 import Predictor

class SVRPredictor(Predictor):
    def __init__(self, C: float=1.0, epsilon: float=0.1, kernel: str="rbf", gamma: str | float="scale", degree: int=3, coef0: float=0.0,shrinking: bool=True, tol: float=1e-3, max_iter: int=-1, n_jobs: int=None):
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.shrinking = shrinking
        self.tol = tol
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self._model = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        base = SVR(C=self.C, epsilon=self.epsilon, kernel=self.kernel, gamma=self.gamma, degree=self.degree, coef0=self.coef0, shrinking=self.shrinking, tol=self.tol, max_iter=self.max_iter)
        self._model = MultiOutputRegressor(base, n_jobs=self.n_jobs)
        self._model.fit(X, Y)

    def pred(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before pred().")
        y = self._model.predict(X)
        if y.ndim == 1:
            y = y[:, None]
        return y