import numpy as np
import time
from .base import Selector, Predictor
    
class DummyPredictor(Predictor):
    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.d_obj = Y.shape[1]
        return self

    def pred(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        m = 1 if X.ndim == 1 else X.shape[0]
        return np.full((m, self.d_obj), 0.0, dtype=float)
    
class RandomSelector(Selector):
    def __init__(self, observed_features, observed_values, unobserved_features, predictor=None, squared_sigma: float=1.0, normalize_features: bool=False, normalize_values: bool=False):
        # dummy args for easier YAML implementation TODO: clean this up?
        super().__init__(observed_features, observed_values, unobserved_features, DummyPredictor(), False, False)
        
    def next_candidates(self, n: int) -> list[int]:
        total_t0 = time.perf_counter()
        if n <= 0:
            return []

        unobs_ids = self.unobs_ids()
        if unobs_ids.size == 0:
            return []

        k = min(n, int(unobs_ids.size))
        selected = np.random.choice(unobs_ids, size=k, replace=False).astype(int).tolist()

        for cid in selected:
            self.candidate_id_history.append(cid)

        self.passed_times_selection.append(time.perf_counter() - total_t0)
        self.passed_times_train.append(0.0)
        self.passed_times_pred.append(0.0)
        self.passed_times_total.append(time.perf_counter() - total_t0)

        return selected