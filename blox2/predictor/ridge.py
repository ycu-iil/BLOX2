import numpy as np
from sklearn.linear_model import Ridge
from blox2 import Predictor, DataPoint

class RidgePointPredictor(Predictor):
    def __init__(self, alpha: float=1.0, fit_intercept: bool=True, random_state: int=0):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.random_state = random_state

    def fit(self, observed_samples: list[DataPoint]):
        X, Y = self.prep_data(observed_samples)
        
        # multi-target regression
        self._model = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept, random_state=self.random_state)
        self._model.fit(X, Y)
        return self

    def pred(self, x: DataPoint) -> list[np.ndarray]:
        if self._model is None:
            raise RuntimeError("Call fit() before pred().")
        fv = np.asarray(x.feature_vector(), float).ravel().reshape(1, -1)
        y_hat = np.asarray(self._model.predict(fv)[0], float)  # (n_objectives,)
        return [y_hat]