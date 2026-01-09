import time
import numpy as np
import pandas as pd
from .base import Selector, Predictor
from .utils import stein_novelty_repli

class BLOX2Selector(Selector):
    def __init__(self, observed_features: pd.DataFrame, observed_values: pd.DataFrame, unobserved_features: pd.DataFrame, predictor: Predictor, normalize_features: bool=True, normalize_values: bool=True, squared_sigma: float=1, use_distribution: bool=False, verbose: bool=False, compare_selection_time=False):
        super().__init__(observed_features, observed_values, unobserved_features, predictor, normalize_features, normalize_values, verbose)
        self.squared_sigma = squared_sigma
        self._use_distribution = use_distribution
        self.compare_selection_time = compare_selection_time
        if verbose:
            self.best_scores = []
        if compare_selection_time:
            self.passed_times_blox2 = []
            self.passed_times_repli = []
            
    def use_distribution(self):
        return self._use_distribution
            
    def best_id(self, X_pred: np.ndarray) -> int:
        t0 = time.perf_counter()
        Y = self.Y_obs
        _, d = Y.shape
        sigma = self.squared_sigma
        unobs_ids = self.unobs_ids()

        best_id = -1
        best_score = -np.inf

        # TODO: batch with chunking
        for i, cid in enumerate(unobs_ids):
            if self.use_distribution():
                d_stein_equivs = []
                for x in X_pred[i]:
                    diff = Y - x # (n, d_obj)
                    dist = (diff * diff).sum(axis=1) # (n,)
                    d_stein_equiv = np.sum((dist - d * sigma) * np.exp(-dist / (2 * sigma)))
                    d_stein_equivs.append(d_stein_equiv)

                score = np.mean(d_stein_equivs) # TODO: compare
                if score > best_score:
                    best_score = score
                    best_id = int(cid)
            else:
                x = X_pred[i] # (d_obj,)
                diff = Y - x # (n, d_obj)
                dist = (diff * diff).sum(axis=1) # (n,)
                score = np.sum((dist - d * sigma) * np.exp(-dist / (2 * sigma)))

                if score > best_score:
                    best_score = score
                    best_id = int(cid)
        
        if self.verbose:
            self.best_scores.append(best_score)

        if self.compare_selection_time:
            self.passed_times_blox2.append(time.perf_counter() - t0)
            if not self.use_distribution():
                best_id_valid = self.best_id_blox_replication(X_pred)
                if best_id != best_id_valid:
                    print(f"WARNING: Different best point at {len(self.obs_ids)} observed points.")

        return best_id

    def best_id_blox_replication(self, X_pred: np.ndarray) -> int:
        """For validation purpose. Not used for the selection."""
        t0 = time.perf_counter()
        Y = self.Y_obs
        unobs_ids = self.unobs_ids()

        best_id = -1
        best_score = -np.inf
        for i, cid in enumerate(unobs_ids):
            s = stein_novelty_repli(X_pred[i], Y, self.squared_sigma)
            if s > best_score:
                best_score = s
                best_id = int(cid)

        if self.compare_selection_time:
            self.passed_times_repli.append(time.perf_counter() - t0)

        return best_id