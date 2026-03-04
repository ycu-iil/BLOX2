import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from .base import Selector, Predictor
from .utils import stein_novelty_repli

class SteinNoveltySelector(Selector):
    def __init__(self, observed_features: pd.DataFrame, observed_values: pd.DataFrame, unobserved_features: pd.DataFrame, predictor: Predictor, normalize_features: bool=True, value_normalization: str="before_pred", pred_clip: list[tuple[float | None, float | None]]=None, sigma: float=1.0, n_obs_samples: int=None, use_uncertainty=False, uncertainty_ratio: float=0.5, uncertainty_aggregation_type: str="mean", print_uncertainty: bool=False, use_input_stein_novelty: bool=False, input_stein_novelty_ratio: float=0.5, use_distribution: bool=False, distribution_pooling_type: str="mean", use_batch_penalty=False, batch_penalty_ratio: float=0.5, batch_penalty_type: str="stein", batch_penalty_cutoff_ratio: float=0.0, batch_penalty_simhash_bits: int=8, input_stein_pca_dim: int=None, input_stein_sigma: float | str="auto", input_stein_auto_n_samples: int=10**5, chunk_size: int=256, compare_selection_time=False, verbose_plot_dir: str=None):
        """
        Args:
            normalize_features: Whether to normalize input feature values for predictions
            value_normalization: 
                - before_pred: fit and apply before prediction
                - after_pred: fit and apply after prediction
                - mixed: apply after prediction, using the scaler fitted before prediction
                - diable: disable
            pred_clip: Valid value range of objectives. This cannot be used with "before_pred".
            
            n_obs_samples: When the number of observed points are greater than this value, samples n_obs_samples points for Stein novelty calculation instead of using all of the observed points.
            
            use_uncertainty: Whether to combine uncertainty score when selecting candidates.
            uncertainty_ratio: Maximize (1 - uncertainty_ratio) * standardized Stein novelty + uncertainty_ratio * standardized uncertainty score
            uncertainty_aggregation_type: How to aggregate uncertainty values over features. "mean", "max" or "l2".
            
            use_input_stein_novelty: Whether to combine Stein novelty in input feature space when selecting candidates.
            input_stein_novelty_ratio: Maximize (1 - input_stein_novelty_ratio) * Stein novelty in output space + input_stein_novelty_ratio * Stein novelty in input space
            
            use_distribution: Whether to use distribution ('pred_samples()' defined in predictor class)
            distribution_pooling_type: How to use Stein novelty scores of predicted samples when 'use_distribution'=True. Can be one of: "mean" / "max"
            
            use_batch_penalty: Whether to use proximity penalty in batch candidates
            batch_penalty_ratio: Maximize (1 - batch_penalty_ratio) * standardized score for non-batch selections (i.e. standardized Stein novelty, + uncertainty if used) + batch_penalty_ratio * (- standardized batch penalty score) when selecting batch candidates.
            batch_penalty_type: 'stein', 'distance', 'simhash' or 'simhash_min_hamming'
            batch_penalty_cutoff_ratio: skip batch penalty calculation of bad candidates (per chunk)
            batch_penalty_simhash_bits: number of SimHash bits (<= 64). Used when batch_penalty_type="simhash" or "simhash_min_hamming".
            
            input_stein_pca_dim: If set to a positive int, apply PCA to X_all_processed (input feature values, used for batch penalty with 'stein' type or input Stein novelty) and reduce its feature dimension to this value.
            input_stein_sigma: σ value of Gaussian kernel used for Stein novelty calculation in input space (batch penalty with 'stein' type or input Stein novelty). Set to 'auto' to use mean pairwise distance.
            input_stein_auto_n_samples: The maximum number of sample pairs when calculating mean pairwise distance.
            
            chunk_size: The number of candidates in one chunk (for chunked Stein novelty calculation)
            verbose_plot_dir: If set, saves verbose plots of predicted values and chosen points in each selection step.
        """
        super().__init__(observed_features, observed_values, unobserved_features, predictor, sigma=sigma, normalize_features=normalize_features, value_normalization=value_normalization, pred_clip=pred_clip, verbose_plot_dir=verbose_plot_dir)     

        self._use_distribution = use_distribution
        self._use_uncertainty = use_uncertainty
        self._use_input_stein_novelty = use_input_stein_novelty
        if use_uncertainty and use_input_stein_novelty:
            raise ValueError("'use_input_stein_novelty' with 'use_uncertainty' is not supported.")
        self.input_stein_novelty_ratio = input_stein_novelty_ratio
        if not (0.0 <= input_stein_novelty_ratio <= 1.0):
            raise ValueError("'input_stein_novelty_ratio' must be in [0, 1].")
        self._use_batch_penalty = use_batch_penalty
        if use_distribution and use_batch_penalty:
            raise ValueError("'use_batch_penalty' with 'use_distribution' is not supported.")
        self.uncertainty_ratio = uncertainty_ratio
        if not (0.0 <= uncertainty_ratio <= 1.0):
            raise ValueError("'uncertainty_ratio' must be in [0, 1].")
        self.batch_penalty_ratio = batch_penalty_ratio
        if not batch_penalty_type in ["stein", "distance", "simhash", "simhash_min_hamming"]:
            raise ValueError("'batch_penalty_type' must be 'stein', 'distance', 'simhash' or 'simhash_min_hamming'")
        self.batch_penalty_type = batch_penalty_type
        self.batch_penalty_cutoff_ratio = batch_penalty_cutoff_ratio
        if not (0.0 <= self.batch_penalty_cutoff_ratio < 1.0):
            raise ValueError("'batch_penalty_cutoff_ratio' must be in [0, 1).")
        
        if use_batch_penalty or use_input_stein_novelty: # standardize input space for batch penalty or input stein novelty
            if not normalize_features:
                mu = self.X_all.mean(axis=0)
                sd = self.X_all.std(axis=0)
                sd = np.where(sd > 1e-12, sd, 1.0) # avoid /0
                self.X_all_processed = (self.X_all - mu[None, :]) / sd[None, :]
            else:
                self.X_all_processed = self.X_all
                
            if input_stein_pca_dim is not None:
                if input_stein_pca_dim <= 0:
                    raise ValueError("'input_stein_pca_dim' must be a positive int or None.")

                if input_stein_pca_dim < self.X_all_processed.shape[1]:
                    pca = PCA(n_components=input_stein_pca_dim, svd_solver="auto", random_state=0)
                    self.X_all_processed = pca.fit_transform(self.X_all_processed) # (n, d)
                else: # already d <= batch_penalty_pca_dim
                    pass
            elif batch_penalty_type in ["simhash", "simhash_min_hamming"]:
                self.batch_penalty_simhash_samples = batch_penalty_simhash_bits
                if self.batch_penalty_simhash_samples <= 0:
                    raise ValueError("'batch_penalty_simhash_samples' must be a positive int.")
                if self.batch_penalty_simhash_samples > 64:
                    raise ValueError("'batch_penalty_simhash_samples' must be <= 64 (packed into uint64).")
                # precompute simhash codes
                rng = np.random.default_rng(0)
                d_feat = self.X_all_processed.shape[1]
                b = self.batch_penalty_simhash_samples
                self._simhash_R = rng.standard_normal(size=(d_feat, b)).astype(np.float32, copy=False) # random hyperplanes: (d_feat, b)
                self._simhash_codes_all = self._simhash_packbits(self.X_all_processed @ self._simhash_R) # (n,)
                
        if (use_batch_penalty and batch_penalty_type == "stein") or use_input_stein_novelty:
            if isinstance(input_stein_sigma, str):
                if input_stein_sigma != "auto":
                    raise ValueError(f"'input_stein_sigma' must be float or 'auto', got {input_stein_sigma}")

                X = self.X_all_processed
                n = X.shape[0]

                # subsample pairs if too large (avoid O(N^2))
                max_pairs = input_stein_auto_n_samples
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
                print(f"Set sigma value of input Stein novelty to: {sigma:.6f}")

                self.input_stein_sigma2 = sigma ** 2
            else:
                self.input_stein_sigma2 = input_stein_sigma ** 2

        self.uncertainty_aggregation_type = uncertainty_aggregation_type
        self.print_uncertainty = print_uncertainty
        self.compare_selection_time = compare_selection_time
        self.n_obs_samples = n_obs_samples
        self.chunk_size = chunk_size
        self.distribution_pooling_type = distribution_pooling_type
        
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

                if self.distribution_pooling_type == "mean":
                    scores = scores_per_sample.mean(axis=0) # (c,)
                elif self.distribution_pooling_type == "max":
                    scores = scores_per_sample.max(axis=0)
                else:
                    raise ValueError(f"Unknown pooling_type: {self.distribution_pooling_type}")

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
                    
                if (s == 0) and (self.use_uncertainty() or self._use_batch_penalty or self._use_input_stein_novelty):
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
                elif self._use_input_stein_novelty:
                    r = self.input_stein_novelty_ratio
                    X_obs_full = self.X_all_processed[np.asarray(self.obs_ids, dtype=int)] # (n_obs, d_feat)

                    # apply sampling if set (same as output Stein novelty)
                    if self.n_obs_samples is not None and self.n_obs_samples > 0 and self.n_obs_samples < n:
                        X_obs = X_obs_full[idx]
                        X_obs = np.asfortranarray(X_obs)
                    else:
                        X_obs = X_obs_full

                    # candidate features in processed input space for this chunk
                    chunk_ids = unobs_ids[s:e].astype(int)
                    Xc_in = self.X_all_processed[chunk_ids] # (c, d_feat)

                    # Stein novelty in input space
                    sigma2_in = self.input_stein_sigma2
                    dim_in = Xc_in.shape[1]
                    diff_in = X_obs[None, :, :] - Xc_in[:, None, :] # (c, n_obs, d_feat)
                    d2_in = np.sum(diff_in * diff_in, axis=2) # (c, n_obs)
                    scores_in = np.sum((d2_in - dim_in * sigma2_in) * np.exp(-d2_in / (2.0 * sigma2_in)), axis=1) # (c,)

                    if s == 0:
                        # fix input Stein novelty scaling based on the first chunk
                        in_m = scores_in.mean()
                        in_s = scores_in.std()
                        if in_s <= 1e-12:
                            in_s = 1.0

                    # z-scores
                    z_out = (scores - sn_m) / sn_s
                    z_in = (scores_in - in_m) / in_s

                    # mix
                    final_scores = (1 - r) * z_out + r * z_in
                elif self._use_batch_penalty: # use_batch_penalty but not use_uncertainty or use_input_stein_novelty (cases where final scores are already normalized)
                    final_scores = (scores - sn_m) / sn_s
                else:
                    final_scores = scores
                    
                if self._use_batch_penalty and len(self.temp_added_ids) > 0:
                    chunk_ids = unobs_ids[s:e].astype(int)
                    c = chunk_ids.size

                    # skip calculation of bad candidates
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

        # simhash
        if self.batch_penalty_type == "simhash_min_hamming":
            cand_codes = self._simhash_codes_all[cand] # (c,)
            sel_codes = self._simhash_codes_all[selected] # (k,)

            # pairwise hamming distances: d(i,j) = popcount(cand_i XOR sel_j)
            ham = np.bitwise_count(cand_codes[:, None] ^ sel_codes[None, :]) # (c, k)

            # nearest selected point in Hamming space
            min_ham = ham.min(axis=1) # (c,)
            eps = 1e-12
            penalty = 1.0 / (min_ham + eps) # closer => larger penalty
            return penalty
        elif self.batch_penalty_type == "simhash":
            cand_codes = self._simhash_codes_all[cand] # (c,)
            sel_codes = self._simhash_codes_all[selected] # (k,)

            # count selected codes
            uniq, cnt = np.unique(sel_codes, return_counts=True)
            code2cnt = dict(zip(uniq.tolist(), cnt.tolist()))

            # Penalty = how many selected points share the same code
            penalty = np.fromiter((code2cnt.get(int(cc), 0) for cc in cand_codes), dtype=float, count=cand_codes.size)
            return penalty

        Xc = self.X_all_processed[cand] # (c, d_feat)
        Xs = self.X_all_processed[selected] # (k, d_feat)
    
        # pairwise squared distances in input space
        d2 = np.sum((Xc[:, None, :] - Xs[None, :, :]) ** 2, axis=2) # (c, k)
        
        if self.batch_penalty_type == "distance":
            min_d = np.sqrt(np.maximum(d2.min(axis=1), 0.0)) # (c,)
            eps = 1e-12
            return 1.0 / (min_d + eps)
        elif self.batch_penalty_type == "stein":
            sigma2 = self.input_stein_sigma2
            dim = Xc.shape[1]
            stein_scores = np.sum((d2 - dim * sigma2) * np.exp(-d2 / (2.0 * sigma2)), axis=1) # (c,)
            return -stein_scores
        else:
            raise RuntimeError(f"Unexpected batch_penalty_type: {self.batch_penalty_type}")

    def _simhash_packbits(self, proj: np.ndarray) -> np.ndarray:
        """
        Pack sign(proj) into uint64 codes.

        Args:
            proj: (n, b) projection values. b <= 64

        Returns:
            codes: (n,) uint64 codes
        """
        P = np.asarray(proj)
        if P.ndim != 2:
            raise ValueError(f"proj must be 2D (n, b), got shape={P.shape}")
        n, b = P.shape
        if b > 64:
            raise ValueError(f"SimHash bits must be <= 64, got b={b}")

        bits = (P >= 0.0) # (n, b) bool
        codes = np.zeros(n, dtype=np.uint64)
        # pack bit j into (1<<j) (LSB-first)
        for j in range(b):
            codes |= (bits[:, j].astype(np.uint64) << np.uint64(j))
        return codes

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