import warnings
from lightgbm import LGBMRegressor
import numpy as np
from blox2 import Predictor

class LightGBMPredictor(Predictor):
    def __init__(self, n_estimators: int=100, learning_rate: float=0.05, num_leaves: int=31, use_uncertainty: bool=False, quantiles: tuple[float, float]=(0.1, 0.9), seed: int=0, lgbm_kwargs: dict=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.use_uncertainty = use_uncertainty
        self.quantiles = quantiles
        self.seed = seed
        self.lgbm_kwargs = lgbm_kwargs or {}

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self._models = []

        base_params = dict(n_estimators=self.n_estimators, learning_rate=self.learning_rate, num_leaves=self.num_leaves, seed=self.seed)
        base_params.update(self.lgbm_kwargs)
        base_params.setdefault("verbose", -1)
        if Y.shape[0] < 60:
            base_params.setdefault("min_child_samples", int(1 + Y.shape[0] / 3)) # will result in the same predicted values otherwise

        for j in range(Y.shape[1]):
            m = LGBMRegressor(**base_params)
            m.fit(X, Y[:, j])
            self._models.append(m)

        if self.use_uncertainty:
            self._q_models_lo = []
            self._q_models_hi = []

            qlo = self.quantiles[0]
            qhi = self.quantiles[1]
            if not (0.0 < qlo < qhi < 1.0):
                raise ValueError("Require 0 < quantiles[0] < quantiles[1] < 1")

            for j in range(Y.shape[1]):
                p_lo = dict(base_params)
                p_lo.update(objective="quantile", alpha=qlo)
                m_lo = LGBMRegressor(**p_lo)
                m_lo.fit(X, Y[:, j])
                self._q_models_lo.append(m_lo)

                p_hi = dict(base_params)
                p_hi.update(objective="quantile", alpha=qhi)
                m_hi = LGBMRegressor(**p_hi)
                m_hi.fit(X, Y[:, j])
                self._q_models_hi.append(m_hi)

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
    
    def uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Uncertainty: qhi_pred - qlo_pred
        """
        if not self.use_uncertainty:
            raise RuntimeError("Initialize LightGBMPredictor with use_uncertainty=True to use uncertainty().")
        if self._q_models_lo is None or self._q_models_hi is None:
            raise RuntimeError("Call fit() before pred_with_uncertainty().")

        qlo_pred = []
        qhi_pred = []
        for m_lo, m_hi in zip(self._q_models_lo, self._q_models_hi):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"X does not have valid feature names, but .* was fitted with feature names",
                    category=UserWarning,
                )
                qlo_pred.append(np.asarray(m_lo.predict(X), float))
                qhi_pred.append(np.asarray(m_hi.predict(X), float))

        qlo_pred = np.stack(qlo_pred, axis=1)
        qhi_pred = np.stack(qhi_pred, axis=1)

        uncertainty = np.maximum(0.0, qhi_pred - qlo_pred)
        return uncertainty