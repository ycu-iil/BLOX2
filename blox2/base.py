from abc import ABC, abstractmethod
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .utils import PointCurve

class Predictor(ABC):
    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        Fit on observed data.
        X: (n_obs, d_feat)
        Y: (n_obs, d_obj)
        """
        raise NotImplementedError
    
    def pred(self, X: np.ndarray) -> np.ndarray:
        """
        Predict objective values for candidates.
        Returns point estimates as ndarray of shape (m, d_obj)
        """
        raise NotImplementedError

    def pred_samples(self, X: np.ndarray) -> np.ndarray:
        """
        (Optional) Predict objective samples for candidates.

        Args:
            X: (m, d_feat) or (d_feat,)
            n_samples: Number of samples to draw.

        Returns:
            samples: (m, n_samples, d_obj)
        """
        raise NotImplementedError
    
    def uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        (Optional) Estimate uncertainty. Called after pred(), so it might be better to cache this value on pred() call depending on the predictor.

        Args:
            X: (m, d_feat)

        Returns:
            uncertainty: (m, d_obj)
        """
        raise NotImplementedError

class Selector(ABC):    
    def __init__(self, observed_features: pd.DataFrame, observed_values: pd.DataFrame, unobserved_features: pd.DataFrame, predictor: Predictor, sigma: float | list[tuple[int, float]]=1.0, normalize_features: bool=True, value_normalization: str="before_pred", pred_clip: list[tuple[float | None, float | None]]=None, verbose_plot_dir: str=None):
        """
        value_normalization: 
            - before_pred: fit and apply before prediction
            - after_pred: fit and apply after prediction
            - mixed: apply after prediction, using the scaler fitted before prediction
            - diable: disable
        pred_clip: Valid value range of objectives. This cannot be used with "before_pred".
        """
        n_obs = len(observed_features)
        n_unobs = len(unobserved_features)

        if n_obs != len(observed_values):
            raise ValueError(f"observed_features ({n_obs}) and observed_values ({len(observed_values)}) must have same length.")
        self.initial_n_obs = n_obs
        
        X_obs_raw = observed_features.to_numpy(dtype=float, copy=True)
        X_unobs_raw = unobserved_features.to_numpy(dtype=float, copy=True)
        self.Y_obs_raw = observed_values.to_numpy(dtype=float, copy=True)
        
        if normalize_features:
            self.x_scaler = StandardScaler()
            X_all_raw = np.vstack([X_obs_raw, X_unobs_raw])
            self.x_scaler.fit(X_all_raw)
            X_obs = self.x_scaler.transform(X_obs_raw)
            X_unobs = self.x_scaler.transform(X_unobs_raw)
        else:
            X_obs, X_unobs = X_obs_raw, X_unobs_raw

        self.value_normalization = value_normalization
        self.y_scaler = None
        
        self.pred_clip = pred_clip
        if self.pred_clip is not None:
            if self.value_normalization == "before_pred":
                raise ValueError("pred_clip cannot be used with value_normalization='before_pred'.")
            for low, high in self.pred_clip:
                if low is not None and high is not None and low > high:
                    raise ValueError(f"Invalid pred_clip range: lo={low} > hi={high}")
        
        self.X_all: np.ndarray = np.vstack([X_obs, X_unobs])
        self.obs_ids: list[int] = list(range(n_obs))

        self.unchecked_mask: np.ndarray = np.ones(n_obs + n_unobs, dtype=bool)
        self.unchecked_mask[:n_obs] = False
        
        self.predictor = predictor
        if type(sigma) == list:
            self._sigma = PointCurve(sigma)
        else:
            self._sigma = sigma
        
        self.verbose_plot_dir = verbose_plot_dir
        if self.verbose_plot_dir is not None:
            os.makedirs(self.verbose_plot_dir, exist_ok=True)
        self.candidate_id_history = list(range(n_obs)) # contains ids of initial points
        self.passed_times_selection = []
        self.passed_times_train = []
        self.passed_times_pred = []
        self.passed_times_total = []
        
    def sigma(self) -> float:
        if type(self._sigma) == PointCurve:
            current_iter = len(self.candidate_id_history) - self.initial_n_obs + 1 # 1-indexed
            return self._sigma.curve(current_iter)
        else:
            return self._sigma
        
    def squared_sigma(self) -> float:
        return self.sigma() ** 2

    def best_id(self, X_pred: np.ndarray, Y_obs: np.ndarray, uncertainty: np.ndarray=None) -> int:
        raise NotImplementedError
    
    # needs to be overridden to use posterior distributuon for acquisition functions
    def use_distribution(self) -> bool:
        return False
    
    # needs to be overridden to use uncertainty for acquisition functions
    def use_uncertainty(self) -> bool:
        return False

    def next_candidate(self) -> int:
        return self.next_candidates(n=1)[0]
    
    def next_candidates(self, n: int) -> list[int]:
        total_t0 = time.perf_counter()
        if n <= 0:
            return []

        if self.Y_obs_raw.size == 0:
            print("No observed point.")
            return []

        X_obs = self.X_obs()
        
        if self.value_normalization == "mixed":
            self.y_scaler = StandardScaler()
            self.y_scaler.fit(self.Y_obs_raw)
            Y_obs = self.Y_obs_raw
        elif self.value_normalization == "before_pred":
            self.y_scaler = StandardScaler()
            self.y_scaler.fit(self.Y_obs_raw)
            Y_obs = self.y_scaler.transform(self.Y_obs_raw)
        else:
            Y_obs = self.Y_obs_raw

        t0 = time.perf_counter()
        self.predictor.fit(X_obs, Y_obs)
        self.passed_times_train.append(time.perf_counter() - t0)

        unobs_ids0 = self.unobs_ids()
        X_unobs0 = self.X_unobs(unobs_ids0)
        
        t0 = time.perf_counter()
        if self.use_distribution():
            Y_pred0 = self.predictor.pred_samples(X_unobs0)
        else:
            Y_pred0 = self.predictor.pred(X_unobs0)
        self.passed_times_pred.append(time.perf_counter() - t0)
        Y_pred0_raw = Y_pred0
        
        if self.pred_clip is not None:
            Y_pred0 = self._clip_raw(Y_pred0)
        
        if self.value_normalization == "mixed":
            if self.use_distribution():
                m, s, d = Y_pred0.shape
                Y_pred_2d = Y_pred0.reshape(m * s, d)
                Y_obs = self.y_scaler.transform(self.Y_obs_raw)
                Y_pred0 = self.y_scaler.transform(Y_pred_2d).reshape(m, s, d)
            else:
                Y_obs = self.y_scaler.transform(self.Y_obs_raw)
                Y_pred0 = self.y_scaler.transform(Y_pred0)
        elif self.value_normalization == "after_pred":
            self.y_scaler = StandardScaler()
            if self.use_distribution():
                m, s, d = Y_pred0.shape
                Y_pred_2d = Y_pred0.reshape(m * s, d)
                Y_all = np.vstack([self.Y_obs_raw, Y_pred_2d])
                self.y_scaler.fit(Y_all)
                Y_obs = self.y_scaler.transform(self.Y_obs_raw)
                Y_pred0 = self.y_scaler.transform(Y_pred_2d).reshape(m, s, d)
            else:
                Y_all = np.vstack([self.Y_obs_raw, Y_pred0])
                self.y_scaler.fit(Y_all)
                Y_obs = self.y_scaler.transform(self.Y_obs_raw)
                Y_pred0 = self.y_scaler.transform(Y_pred0)
        else:
            pass
        
        if self.use_uncertainty():
            if self.use_distribution():
                raise RuntimeError("use_uncertainty() with use_distribution() is not supported.")
            if not hasattr(self.predictor, "uncertainty"):
                raise RuntimeError("Selector.use_uncertainty() is True but predictor has no uncertainty().")

            t_u = time.perf_counter()
            U0 = self.predictor.uncertainty(X_unobs0) # (m, d_obj)
            self.passed_times_pred[-1] += (time.perf_counter() - t_u)

            if U0.ndim != 2 or U0.shape[0] != len(unobs_ids0):
                raise ValueError(f"uncertainty must be (m, d_obj) with m=len(unobs_ids0)={len(unobs_ids0)}, got {U0.shape}")

            unc_by_id = {int(cid): U0[i] for i, cid in enumerate(unobs_ids0)}

        # cache predictions by id to rebuild X_pred in the current unobs_ids() order
        pred_by_id = {int(cid): Y_pred0[i] for i, cid in enumerate(unobs_ids0)}
        pred_raw_by_id = {int(cid): Y_pred0_raw[i] for i, cid in enumerate(unobs_ids0)}

        selected_ids = []
        self.temp_added_ids = []
        selection_time = 0
        
        for _ in range(min(n, int(unobs_ids0.size))):
            cur_unobs_ids = self.unobs_ids()
            if cur_unobs_ids.size == 0:
                break

            if self.use_distribution():
                X_pred_cur = np.stack([pred_by_id[int(cid)] for cid in cur_unobs_ids], axis=0)
            else:
                X_pred_cur = np.vstack([pred_by_id[int(cid)] for cid in cur_unobs_ids])
            
            if self.use_uncertainty():
                U_cur = np.vstack([unc_by_id[int(cid)] for cid in cur_unobs_ids]) # (n_cur, d_obj)
            else:
                U_cur = None

            t0 = time.perf_counter()
            cid = self.best_id(X_pred_cur, Y_obs, uncertainty=U_cur)
            selection_time += time.perf_counter() - t0
            self.candidate_id_history.append(cid)

            selected_ids.append(cid)
            self.temp_added_ids.append(cid)

            # virtual observation                
            if self.use_distribution():
                y_virtual_for_obs = np.mean(pred_raw_by_id[cid], axis=0)
            else:
                y_virtual_for_obs = pred_raw_by_id[cid]
                
            if self.value_normalization == "before_pred": #invert scaling
                y_virtual_for_obs = self.y_scaler.inverse_transform(y_virtual_for_obs.reshape(1, -1))[0]

            self.observe(cid, y_virtual_for_obs)
            
        self.passed_times_selection.append(selection_time)

        # revert virtual observation
        for cid in reversed(self.temp_added_ids):
            self.unobserve(cid)
        self.temp_added_ids = []

        self.passed_times_total.append(time.perf_counter() - total_t0)
        
        if self.verbose_plot_dir is not None:
            try:
                if self.y_scaler is not None:
                    Y_obs_plot = self.y_scaler.inverse_transform(np.asarray(Y_obs))
                else:
                    Y_obs_plot = np.asarray(Y_obs)
                    
                if Y_obs_plot.ndim == 2 and Y_obs_plot.shape[1] == 2:
                    if self.use_distribution():
                        Yp = np.mean(np.asarray(Y_pred0), axis=1)
                    else:
                        Yp = np.asarray(Y_pred0)
                        
                    if self.y_scaler is not None:
                        Y_pred_plot = self.y_scaler.inverse_transform(Yp)
                    else:
                        Y_pred_plot = Yp

                    selected_set = set(int(x) for x in selected_ids)
                    sel_idx = [i for i, cid in enumerate(unobs_ids0) if int(cid) in selected_set]
                    Y_sel_plot = Y_pred_plot[sel_idx] if len(sel_idx) > 0 else np.empty((0, 2))

                    plt.figure()
                    plt.scatter(Y_pred_plot[:, 0], Y_pred_plot[:, 1], c="C1", s=10, alpha=0.6, label="Predicted")
                    plt.scatter(Y_obs_plot[:, 0], Y_obs_plot[:, 1], c="C0", s=30, alpha=0.8, label="Observed")
                    if Y_sel_plot.size > 0:
                        plt.scatter(Y_sel_plot[:, 0], Y_sel_plot[:, 1], c="C2", s=80, alpha=0.9, label="Selected")

                    plt.xlabel("objective 1")
                    plt.ylabel("objective 2")
                    plt.legend()
                    plt.tight_layout()
                    step = len(self.passed_times_total)
                    out_path = os.path.join(self.verbose_plot_dir, f"scatter_{step}.png")
                    plt.savefig(out_path, dpi=200)
                    plt.close()
            except Exception:
                pass
            
        return selected_ids

    def observe(self, id: int, observed_values: np.ndarray):
        if not self.unchecked_mask[id]:
            raise ValueError(f"id={id} is already observed, or not in unchecked set.")

        y = np.asarray(observed_values, float).ravel()

        self.obs_ids.append(id)
        if self.Y_obs_raw.size == 0:
            self.Y_obs_raw = y[None, :]
        else:
            self.Y_obs_raw = np.vstack([self.Y_obs_raw, y[None, :]])

        self.unchecked_mask[id] = False
        
    def unobserve(self, id: int):
        if self.unchecked_mask[id]:
            return

        try:
            idx = self.obs_ids.index(id)
        except ValueError:
            raise RuntimeError(f"Inconsistent state: id={id} not found in obs_ids.")

        self.obs_ids.pop(idx)

        if self.Y_obs_raw.size == 0:
            raise RuntimeError("Inconsistent state: Y_obs is empty.")
        elif self.Y_obs_raw.shape[0] == 1:
            self.Y_obs_raw = np.empty((0, self.Y_obs_raw.shape[1]))
        else:
            self.Y_obs_raw = np.delete(self.Y_obs_raw, idx, axis=0)

        self.unchecked_mask[id] = True
            
    def unobs_ids(self) -> np.ndarray:
        return np.flatnonzero(self.unchecked_mask)
    
    def n_unobs(self) -> int:
        return int(self.unobs_ids().size)
    
    def X_obs(self):
        return self.X_all[np.asarray(self.obs_ids, int)]
    
    def X_unobs(self, unobs_ids=None):
        unobs_ids = self.unobs_ids() if unobs_ids is None else unobs_ids
        if unobs_ids.size == 0:
            raise ValueError("No unobserved points.")
        return self.X_all[unobs_ids]
    
    def _clip_raw(self, Y_raw: np.ndarray) -> np.ndarray:
        """
        Clip objective values in raw space using self.pred_clip.
        """
        if self.pred_clip is None:
            return Y_raw

        Y = np.asarray(Y_raw, dtype=float)

        d = Y.shape[-1]
        if len(self.pred_clip) != d:
            raise ValueError(f"pred_clip must have length of the number of objectives {d}, but got {len(self.pred_clip)}")

        lo = np.array([(-np.inf if a is None else float(a)) for a, _ in self.pred_clip], dtype=float)
        hi = np.array([( np.inf if b is None else float(b)) for _, b in self.pred_clip], dtype=float)

        if Y.ndim == 2:
            return np.minimum(np.maximum(Y, lo[None, :]), hi[None, :])

        if Y.ndim == 3:
            return np.minimum(np.maximum(Y, lo[None, None, :]), hi[None, None, :])

        raise ValueError(f"Invalid Y_raw.ndim ({Y.ndim}) in predicted value clipping.")
    
    def inverse_transform_y(self, y_scaled: np.ndarray) -> np.ndarray:
        if self.y_scaler is None:
            return y_scaled
        return self.y_scaler.inverse_transform(y_scaled)
    
    def make_observation_history(self) -> np.ndarray:
        id_to_row = {int(cid): i for i, cid in enumerate(self.obs_ids)}
        rows = []
        for cid in self.candidate_id_history:
            if cid not in id_to_row:
                raise ValueError(f"cid={cid} is not currently observed, so Y cannot be reconstructed from current state.")
            rows.append(self.Y_obs_raw[id_to_row[cid]])

        return np.asarray(rows)