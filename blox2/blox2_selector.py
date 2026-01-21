import time
import numpy as np
import pandas as pd
from .base import Selector, Predictor
from .utils import stein_novelty_repli

class BLOX2Selector(Selector):
    def __init__(self, observed_features: pd.DataFrame, observed_values: pd.DataFrame, unobserved_features: pd.DataFrame, predictor: Predictor, normalize_features: bool=True, value_normalization: str="default", pred_clip: list[tuple[float | None, float | None]]=None, sigma: float=1.0, n_obs_samples: int=None, n_chunks: int=256, use_distribution: bool=False, pooling: str="mean", compare_selection_time=False, verbose_plot_dir: str=None):
        """
        Args:
            value_normalization: 
                - default: apply after prediction, using the scaler fitted before prediction
                - before_pred: fit and apply before prediction
                - after_pred: fit and apply after prediction
                - diable: disable
            pred_clip: Valid value range of objectives. This cannot be used with "before_pred".
            n_obs_samples: When the number of observed points are greater than this value, samples n_obs_samples points for Stein novelty calculation instead of using all of the observed points.
            pooling: How to use Stein novelty scores of predicted samples when 'use_distribution'=True. Can be one of: "mean" / "max"
        """
        super().__init__(observed_features, observed_values, unobserved_features, predictor, sigma=sigma, normalize_features=normalize_features, value_normalization=value_normalization, pred_clip=pred_clip, verbose_plot_dir=verbose_plot_dir)
        self._use_distribution = use_distribution
        self.compare_selection_time = compare_selection_time
        self.n_obs_samples = n_obs_samples
        self.n_chunks = n_chunks
        self.pooling = pooling
        
        if compare_selection_time:
            self.passed_times_blox2 = []
            self.passed_times_repli = []
            
    def use_distribution(self):
        return self._use_distribution
            
    def best_id(self, X_pred: np.ndarray, Y_obs: np.ndarray) -> int:
        t0 = time.perf_counter()
        
        Y_full = Y_obs
        n, d = Y_full.shape
        sigma2 = self.squared_sigma
        unobs_ids = self.unobs_ids()

        if self.n_obs_samples is not None and self.n_obs_samples > 0 and self.n_obs_samples < n:
            idx = np.random.choice(n, self.n_obs_samples, replace=False)
            Y = Y_full[idx]
        else:
            Y = Y_full

        best_id = -1
        best_score = -np.inf
        
        if self.use_distribution(): # X_pred: (n_unobs, n_samples, d)
            for s in range(0, len(unobs_ids), self.n_chunks):
                e = min(s + self.n_chunks, len(unobs_ids))

                Xc = X_pred[s:e] # (c, n_samples, d)
                n_samples = Xc.shape[1]
                c = e - s

                scores_per_sample = np.zeros((n_samples, c)) # (n_samples, c)

                for k in range(n_samples):
                    x = Xc[:, k, :] # (c, d)
                    diff = Y[None, :, :] - x[:, None, :] # (c, n_obs, d)
                    dist = np.sum(diff * diff, axis=2) # (c, n_obs)

                    scores_per_sample[k] = np.sum((dist - d * sigma2) * np.exp(-dist / (2 * sigma2)), axis=1)

                if self.pooling == "mean":
                    scores = scores_per_sample.mean(axis=0) # (c,)
                elif self.pooling == "max":
                    scores = scores_per_sample.max(axis=0)
                else:
                    raise ValueError(f"Unknown pooling_type: {self.pooling}")

                j = np.argmax(scores)
                if scores[j] > best_score:
                    best_score = scores[j]
                    best_id = int(unobs_ids[s + j])
        else: # X_pred: (n_unobs, d)
            for s in range(0, len(unobs_ids), self.n_chunks):
                e = min(s + self.n_chunks, len(unobs_ids))

                Xc = X_pred[s:e] # (c, d)
                diff = Y[None, :, :] - Xc[:, None, :] # (c, n_obs, d)
                dist = np.sum(diff * diff, axis=2) # (c, n_obs)

                scores = np.sum((dist - d * sigma2) * np.exp(-dist / (2 * sigma2)), axis=1)

                j = np.argmax(scores)
                if scores[j] > best_score:
                    best_score = scores[j]
                    best_id = int(unobs_ids[s + j])

        if self.compare_selection_time:
            self.passed_times_blox2.append(time.perf_counter() - t0)
            if not self.use_distribution():
                best_id_valid = self.best_id_blox_replication(X_pred, Y_obs)
                if best_id != best_id_valid:
                    print(f"WARNING: Different best point at {len(self.obs_ids)} observed points.")

        return best_id

    def best_id_blox_replication(self, X_pred: np.ndarray, Y_obs: np.ndarray) -> int:
        """For validation purpose. Not used for the selection."""
        t0 = time.perf_counter()
        Y = Y_obs
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