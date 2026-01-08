import numpy as np
from sklearn.linear_model import Ridge
from blox2 import Predictor

class RidgePredictor(Predictor):
    def __init__(self, alpha: float=1.0, fit_intercept: bool=True, random_state: int=0):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.random_state = random_state

    def fit(self, X: np.ndarray, Y: np.ndarray):        
        # multi-target regression
        self._model = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept, random_state=self.random_state)
        self._model.fit(X, Y)

    def pred(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before pred().")
        y = self._model.predict(X)
        if y.ndim == 1:
            y = y[:, None]
        return y