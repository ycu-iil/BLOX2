import numpy as np
from sklearn.linear_model import Lasso
from sklearn.multioutput import MultiOutputRegressor
from blox2 import Predictor

class LassoPredictor(Predictor):
    def __init__(self, alpha: float=1.0, fit_intercept: bool=True, max_iter: int=1000, tol: float=1e-4, selection: str="cyclic", random_state: int=0, n_jobs: int=None, positive: bool=False):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.selection = selection
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.positive = positive
        self._model = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        base = Lasso(alpha=self.alpha, fit_intercept=self.fit_intercept, max_iter=self.max_iter, tol=self.tol, selection=self.selection, random_state=self.random_state, positive=self.positive)
        self._model = MultiOutputRegressor(base, n_jobs=self.n_jobs)
        self._model.fit(X, Y)

    def pred(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before pred().")
        y = self._model.predict(X)
        if y.ndim == 1:
            y = y[:, None]
        return y