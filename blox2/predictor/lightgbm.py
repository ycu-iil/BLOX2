import warnings
from lightgbm import LGBMRegressor
import numpy as np
from blox2 import Predictor

class LightGBMPredictor(Predictor):
    def __init__(self, n_estimators: int=100, learning_rate: float=0.05, num_leaves: int=31, seed: int=0, lgbm_kwargs: dict=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.seed = seed
        self.lgbm_kwargs = lgbm_kwargs or {}

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self._models = []

        base_params = dict(n_estimators=self.n_estimators, learning_rate=self.learning_rate, num_leaves=self.num_leaves, seed=self.seed)
        base_params.update(self.lgbm_kwargs)
        base_params.setdefault("verbose", -1)
        if Y.shape[0] < 60:
            base_params.setdefault("min_child_samples", 1 + Y.shape[0] / 3) # will result in the same predicted values otherwise

        for j in range(Y.shape[1]):
            m = LGBMRegressor(**base_params)
            m.fit(X, Y[:, j])
            self._models.append(m)

        return self

    def pred(self, X: np.ndarray) -> np.ndarray:
        if self._models is None:
            raise RuntimeError("Call fit() before pred().")

        preds = []
        for m in self._models:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"X does not have valid feature names, but .* was fitted with feature names",
                    category=UserWarning,
                )
                preds.append(np.asarray(m.predict(X), float))

        return np.stack(preds, axis=1)