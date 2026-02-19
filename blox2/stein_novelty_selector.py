import time
import numpy as np
import pandas as pd
from .base import Selector, Predictor
from .utils import stein_novelty_repli

class SteinNoveltySelector(Selector):
    def __init__(self, observed_features: pd.DataFrame, observed_values: pd.DataFrame, unobserved_features: pd.DataFrame, predictor: Predictor, normalize_features: bool=True, value_normalization: str="default", pred_clip: list[tuple[float | None, float | None]]=None, sigma: float=1.0, n_obs_samples: int=None, chunk_size: int=256, use_uncertainty=False, uncertainty_ratio: float=0.2, uncertainty_aggregation_type: str="mean", print_uncertainty: bool=False, use_distribution: bool=False, pooling: str="mean", use_batch_penalty=False, batch_penalty_ratio: float=0.5, batch_penalty_type: str="stein", batch_penalty_stein_sigma: float | str="auto", batch_penalty_auto_sigma_max_samples: int=10**5, batch_penalty_cutoff_ratio: float=0.0, compare_selection_time=False, verbose_plot_dir: str=None):
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
            batch_penalty_cutoff_ratio: skip batch penalty calculation of bad candidates (per chunk)
        """
        super().__init__(observed_features, observed_values, unobserved_features, predictor, sigma=sigma, normalize_features=normalize_features, value_normalization=value_normalization, pred_clip=pred_clip, verbose_plot_dir=verbose_plot_dir)     

        self._use_distribution = use_distribution
        self._use_uncertainty = use_uncertainty
        self._use_batch_penalty = use_batch_penalty
        if use_distribution and use_batch_penalty:
            raise ValueError("'use_batch_penalty' with 'use_distribution' is not supported.")
        self.uncertainty_ratio = uncertainty_ratio
        if uncertainty_ratio > 1:
            raise ValueError("'uncertainty_ratio' must be <= 1.0.")
        self.batch_penalty_ratio = batch_penalty_ratio
        if not batch_penalty_type in ["stein", "distance"]:
            raise ValueError("'batch_penalty_type' must be 'stein' or 'distance'")
        self.batch_penalty_type = batch_penalty_type
        self.batch_penalty_cutoff_ratio = batch_penalty_cutoff_ratio
        if not (0.0 <= self.batch_penalty_cutoff_ratio < 1.0):
            raise ValueError("'batch_penalty_cutoff_ratio' must be in [0, 1).")
        
        if use_batch_penalty: # standardize input space for batch penalty
            if not normalize_features:
                mu = self.X_all.mean(axis=0)
                sd = self.X_all.std(axis=0)
                sd = np.where(sd > 1e-12, sd, 1.0) # avoid /0
                self.X_all_normalized = (self.X_all - mu[None, :]) / sd[None, :]
            else:
                self.X_all_normalized = self.X_all
        
        if isinstance(batch_penalty_stein_sigma, str):
            if batch_penalty_stein_sigma != "auto":
                raise ValueError(f"batch_penalty_stein_sigma must be float or 'auto', got {batch_penalty_stein_sigma}")

            X = self.X_all_normalized
            n = X.shape[0]

            # subsample pairs if too large (avoid O(N^2))
            max_pairs = batch_penalty_auto_sigma_max_samples
            if n * (n - 1) // 2 > max_pairs:
                rng = np.random.default_rng(0)
                idx1 = rng.integers(0, n, size=max_pairs)
                idx2 = rng.integers(0, n, size=max_pairs)
                mask = idx1 != idx2
                dists = np.linalg.norm(X[idx1[mask]] - X[idx2[mask]], axis=1)
            else:
                diff = X[:, None, :] - X[None, :, :]
                dists = np.linalg.norm(diff, axis=-1)
                dists = dists[np.triu_indices(n, k=1)]

            sigma = dists.mean()
            print(f"[Stein batch penalty] Set sigma to: {sigma:.6f}")

            self.batch_penalty_stein_sigma2 = sigma ** 2
        else:
            self.batch_penalty_stein_sigma2 = batch_penalty_stein_sigma ** 2

        self.uncertainty_aggregation_type = uncertainty_aggregation_type
        self.print_uncertainty = print_uncertainty
        self.compare_selection_time = compare_selection_time
        self.n_obs_samples = n_obs_samples
        self.chunk_size = chunk_size
        self.pooling = pooling
        
        if compare_selection_time:
            self.passed_times_blox2 = []
            self.passed_times_repli = []
            
    def use_distribution(self):
        return self._use_distribution
    
    def use_uncertainty(self):
        return self._use_uncertainty
            
    def best_id(self, X_pred: np.ndarray, Y_obs: np.ndarray, uncertainty: np.ndarray=None) -> int:
        t0 = time.perf_counter()
        
        Y_full = Y_obs
        n, dim = Y_full.shape
        sigma2 = self.squared_sigma()
        unobs_ids = self.unobs_ids()

        if self.n_obs_samples is not None and self.n_obs_samples > 0 and self.n_obs_samples < n:
            idx = np.random.choice(n, self.n_obs_samples, replace=False)
            Y = Y_full[idx]
            Y = np.asfortranarray(Y) # enforce Fortran order to avoid selection-time bloat (tested)
        else:
            Y = Y_full
            
        if self._use_batch_penalty:
            self._bp_mean = None
            self._bp_std = None

        best_id = -1
        best_score = -np.inf
        
        if self.use_distribution(): # X_pred: (n_unobs, n_samples, d)
            for s in range(0, len(unobs_ids), self.chunk_size):
                e = min(s + self.chunk_size, len(unobs_ids))

                Xc = X_pred[s:e] # (c, n_samples, d)
                n_samples = Xc.shape[1]
                c = e - s

                scores_per_sample = np.zeros((n_samples, c)) # (n_samples, c)

                for k in range(n_samples):
                    x = Xc[:, k, :] # (c, d)
                    diff = Y[None, :, :] - x[:, None, :] # (c, n_obs, d)
                    dist = np.sum(diff * diff, axis=2) # (c, n_obs)

                    scores_per_sample[k] = np.sum((dist - dim * sigma2) * np.exp(-dist / (2 * sigma2)), axis=1) # * - sigma2^2 (from the original)

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
            for s in range(0, len(unobs_ids), self.chunk_size):
                e = min(s + self.chunk_size, len(unobs_ids))

                Xc = X_pred[s:e] # (c, d)
                diff = Y[None, :, :] - Xc[:, None, :] # (c, n_obs, d)
                dist = np.sum(diff * diff, axis=2) # (c, n_obs)

                scores = np.sum((dist - dim * sigma2) * np.exp(-dist / (2 * sigma2)), axis=1)
                    
                if (s == 0) and (self.use_uncertainty() or self._use_batch_penalty):
                    # fix Stein novelty scaling based on the first chunk
                    sn_m = scores.mean()
                    sn_s = scores.std()
                    if sn_s <= 1e-12:
                        sn_s = 1.0

                if self.use_uncertainty():
                    if s == 0:
                        # standardize uncertainty from all predicted values per objective
                        U_all = np.asarray(uncertainty, float) # (n_unobs, d_obj)

                        Um = U_all.mean(axis=0) # (d_obj,)
                        Us = U_all.std(axis=0) # (d_obj,)
                        Us = np.where(Us > 1e-12, Us, 1.0) # avoid /0

                        U_all_z = (U_all - Um[None, :]) / Us[None, :] # (n_unobs, d_obj)
                        uc_all_z = self._aggregate_uncertainty(U_all_z) # (n_unobs,)

                    # slice precomputed aggregated uncertainty (already z-scored per objective then aggregated)
                    uc_z = uc_all_z[s:e]  # (c,)

                    # fixed z-score for Stein novelty using first-chunk stats
                    z_scores = (scores - sn_m) / sn_s
                        
                    # combine
                    final_scores = (1 - self.uncertainty_ratio) * z_scores + self.uncertainty_ratio * uc_z
                elif self._use_batch_penalty:
                    final_scores = (scores - sn_m) / sn_s
                else:
                    final_scores = scores
                    
                if self._use_batch_penalty and len(self.temp_added_ids) > 0:
                    chunk_ids = unobs_ids[s:e].astype(int)
                    c = int(chunk_ids.size)

                    # Skip calculation of bad candidates
                    cutoff = self.batch_penalty_cutoff_ratio
                    n_keep = max(1, int(np.ceil(c * (1.0 - cutoff)))) # always keep >= 1
                    if n_keep >= c:
                        keep_idx = np.arange(c, dtype=int)
                    else:
                        keep_idx = np.argpartition(final_scores, -n_keep)[-n_keep:]

                    raw_penalty_keep = self.batch_penalty(chunk_ids[keep_idx])

                    if self._bp_mean is None:
                        self._bp_mean = raw_penalty_keep.mean()
                        self._bp_std = raw_penalty_keep.std()
                        if self._bp_std <= 1e-12:
                            self._bp_std = 1.0

                    penalty_z = np.empty(c, dtype=float)

                    penalty_skip_z = 1000.0
                    penalty_z.fill(penalty_skip_z)

                    penalty_z[keep_idx] = (raw_penalty_keep - self._bp_mean) / self._bp_std

                    final_scores = (1 - self.batch_penalty_ratio) * final_scores - self.batch_penalty_ratio * penalty_z # + ratio * (-penalty)
                    
                j = np.argmax(final_scores)
                score_j = final_scores[j]

                if score_j > best_score:
                    # for testing
                    if self.use_uncertainty() and self.print_uncertainty:
                        print("SN, u, Combined: ", z_scores[j], score_j-z_scores[j], score_j)

                    best_score = score_j
                    best_id = int(unobs_ids[s + j])
                    
        # print("n_full=", Y_full.shape[0], "Y=", Y.shape, "dtype", Y.dtype, "C", Y.flags['C_CONTIGUOUS'], "F", Y.flags['F_CONTIGUOUS'])

        if self.compare_selection_time:
            self.passed_times_blox2.append(time.perf_counter() - t0)
            if not self.use_distribution():
                best_id_valid = self.best_id_blox_replication(X_pred, Y_obs)
                if best_id != best_id_valid:
                    print(f"WARNING: Different best point at {len(self.obs_ids)} observed points.")

        return best_id
    
    def batch_penalty(self, candidate_ids: np.ndarray) -> np.ndarray:
        if (not self._use_batch_penalty) or (self.batch_penalty_ratio <= 0.0):
            return np.zeros(int(candidate_ids.size), dtype=float)

        cand = np.asarray(candidate_ids, dtype=int).ravel()
        selected = np.asarray(self.temp_added_ids, dtype=int).ravel() # len(self.temp_added_ids) > 0 if called

        Xc = self.X_all_normalized[cand] # (c, d_feat)
        Xs = self.X_all_normalized[selected] # (k, d_feat)
    
        # pairwise squared distances in input space
        d2 = np.sum((Xc[:, None, :] - Xs[None, :, :]) ** 2, axis=2) # (c, k)
        eps = 1e-12
        
        if self.batch_penalty_type == "distance":
            min_d = np.sqrt(np.maximum(d2.min(axis=1), 0.0)) # (c,)
            return 1.0 / (min_d + eps)
        elif self.batch_penalty_type == "stein":
            sigma2 = self.batch_penalty_stein_sigma2
            dim = Xc.shape[1]
            stein_scores = np.sum((d2 - dim * sigma2) * np.exp(-d2 / (2.0 * sigma2)), axis=1) # (c,)
            return -stein_scores
        else:
            raise RuntimeError(f"Unexpected batch_penalty_type: {self.batch_penalty_type}")

    def best_id_blox_replication(self, X_pred: np.ndarray, Y_obs: np.ndarray) -> int:
        """For validation purpose. Not used for the selection."""
        t0 = time.perf_counter()
        Y = Y_obs
        unobs_ids = self.unobs_ids()

        best_id = -1
        best_score = -np.inf
        for i, cid in enumerate(unobs_ids):
            s = stein_novelty_repli(X_pred[i], Y, self.squared_sigma())
            if s > best_score:
                best_score = s
                best_id = int(cid)

        if self.compare_selection_time:
            self.passed_times_repli.append(time.perf_counter() - t0)

        return best_id
    
    def _aggregate_uncertainty(self, U: np.ndarray) -> np.ndarray:
        """
        Args:
            U: (m, d_obj) uncertainty per objective

        Returns:
            u: (m,) aggregated uncertainty
        """
        U = np.asarray(U, float)
        if U.ndim != 2:
            raise ValueError(f"U must be 2D (m, d_obj), got shape={U.shape}")

        if self.uncertainty_aggregation_type == "mean":
            return U.mean(axis=1)
        elif self.uncertainty_aggregation_type == "max":
            return U.max(axis=1)
        elif self.uncertainty_aggregation_type == "l2":
            return np.sqrt(np.sum(U * U, axis=1))
        else:
            raise ValueError(f"Unknown uncertainty_agg: {self.uncertainty_aggregation_type}")