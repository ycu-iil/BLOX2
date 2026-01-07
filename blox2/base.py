from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class Selector(ABC):        
    def __init__(self, observed_features: pd.DataFrame, observed_values: pd.DataFrame, unobserved_features: pd.DataFrame):
        n_obs = len(observed_features)
        n_unobs = len(unobserved_features)

        if n_obs != len(observed_values):
            raise ValueError(f"observed_features ({n_obs}) and observed_values ({len(observed_values)}) must have same length.")

        X_obs = observed_features.to_numpy(dtype=float, copy=True)
        X_unobs = unobserved_features.to_numpy(dtype=float, copy=True)
        
        self.X_all: np.ndarray = np.vstack([X_obs, X_unobs])
        self.obs_ids: list[int] = list(range(n_obs))
        self.Y_obs: np.ndarray = observed_values.to_numpy(dtype=float, copy=True)

        self.unchecked_mask: np.ndarray = np.ones(n_obs + n_unobs, dtype=bool)
        self.unchecked_mask[:n_obs] = False

    def next_candidate_id(self) -> int:
        raise NotImplementedError

    def observe(self, id: int, observed_values: np.ndarray):
        if not self.unchecked_mask[id]:
            raise ValueError(f"id={id} is already observed, or not in unchecked set.")

        y = np.asarray(observed_values, float).ravel()

        self.obs_ids.append(id)
        if self.Y_obs.size == 0:
            self.Y_obs = y[None, :]
        else:
            self.Y_obs = np.vstack([self.Y_obs, y[None, :]])

        self.unchecked_mask[id] = False
        
    def unobs_ids(self) -> np.ndarray:
        return np.flatnonzero(self.unchecked_mask)
    
    def X_obs(self):
        return self.X_all[np.asarray(self.obs_ids, int)]
    
    def X_unobs(self, unobs_ids=None):
        unobs_ids = self.unobs_ids() if unobs_ids is None else unobs_ids
        if unobs_ids.size == 0:
            raise ValueError("No unobserved points.")
        return self.X_all[unobs_ids]

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
        Returns point estimates as ndarray of shape (m, d_obj) when X is (m, d_feat),
        or shape (d_obj,) when X is (d_feat,).
        """
        raise NotImplementedError

    def pred_samples(self, X: np.ndarray, n_samples: int=1) -> np.ndarray:
        """
        Predict objective samples for candidates.

        Args:
            X: (m, d_feat) or (d_feat,)
            n_samples: Number of samples to draw.

        Returns:
            samples: (n_samples, m, d_obj) if X is (m, d_feat)
                     (n_samples, d_obj)    if X is (d_feat,)
        """
        raise NotImplementedError
