from abc import ABC, abstractmethod
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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
        Predict objective samples for candidates.

        Args:
            X: (m, d_feat) or (d_feat,)
            n_samples: Number of samples to draw.

        Returns:
            samples: (m, n_samples, d_obj)
        """
        raise NotImplementedError

class Selector(ABC):    
    def __init__(self, observed_features: pd.DataFrame, observed_values: pd.DataFrame, unobserved_features: pd.DataFrame, predictor: Predictor, normalize_features: bool=True, normalize_values: bool=True):
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

        self.normalize_values = normalize_values
        self.y_scaler = None
        
        self.X_all: np.ndarray = np.vstack([X_obs, X_unobs])
        self.obs_ids: list[int] = list(range(n_obs))

        self.unchecked_mask: np.ndarray = np.ones(n_obs + n_unobs, dtype=bool)
        self.unchecked_mask[:n_obs] = False
        
        self.predictor = predictor
        
        self.candidate_id_history = list(range(n_obs)) # contains ids of initial points
        self.passed_times_selection = []
        self.passed_times_train = []
        self.passed_times_pred = []

    def best_id(self, X_pred: np.ndarray, Y_obs: np.ndarray) -> int:
        raise NotImplementedError
    
    # needs to be overridden to use posterior distributuon for acquisition functions
    def use_distribution(self) -> bool:
        return False

    def next_candidate(self) -> int:
        return self.next_candidates(n=1)[0]
    
    def next_candidates(self, n: int) -> list[int]:
        if n <= 0:
            return []

        if self.Y_obs_raw.size == 0:
            print("No observed point.")
            return []

        X_obs = self.X_obs()
        
        if self.normalize_values:
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
            X_pred0 = self.predictor.pred_samples(X_unobs0)
        else:
            X_pred0 = self.predictor.pred(X_unobs0)
        self.passed_times_pred.append(time.perf_counter() - t0)

        # cache predictions by id to rebuild X_pred in the current unobs_ids() order
        pred_by_id = {int(cid): X_pred0[i] for i, cid in enumerate(unobs_ids0)}

        selected_ids = []
        temp_added_ids = []
        
        for _ in range(min(n, int(unobs_ids0.size))):
            cur_unobs_ids = self.unobs_ids()
            if cur_unobs_ids.size == 0:
                break

            # X_pred_cur = np.vstack([pred_by_id[int(cid)] for cid in cur_unobs_ids])
            if self.use_distribution():
                X_pred_cur = np.stack([pred_by_id[int(cid)] for cid in cur_unobs_ids], axis=0)
            else:
                X_pred_cur = np.vstack([pred_by_id[int(cid)] for cid in cur_unobs_ids])
                
            t0 = time.perf_counter()
            cid = self.best_id(X_pred_cur, Y_obs)
            
            self.passed_times_selection.append(time.perf_counter() - t0)
            self.candidate_id_history.append(cid)

            selected_ids.append(cid)
            temp_added_ids.append(cid)

            # virtual observation
            if self.use_distribution():
                y_virtual = np.mean(pred_by_id[cid], axis=0)
                self.observe(cid, y_virtual)
            else:
                self.observe(cid, pred_by_id[cid])

        # revert virtual observation
        for cid in reversed(temp_added_ids):
            self.unobserve(cid)

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