import time
import numpy as np
import pandas as pd
from .base import Selector, Predictor
from .utils import stein_novelty_repli

class BLOX2Selector(Selector):
    def __init__(self, observed_features: pd.DataFrame, observed_values: pd.DataFrame, unobserved_features: pd.DataFrame, predictor: Predictor, squared_sigma: float=0.1, verbose: bool=False):
        super().__init__(observed_features, observed_values, unobserved_features)
        self.predictor = predictor
        self.squared_sigma = squared_sigma
        self.verbose = verbose
        if verbose:
           self.passed_time_blox2 = 0
           self.passed_time_repli = 0
           
    def best_stein_novelty_id(self) -> int:
        t0 = time.perf_counter()
        
        Y = self.Y_obs

        n, d = Y.shape
        sigma = self.squared_sigma

        unobs_ids = self.unobs_ids()
        X_unobs = self.X_unobs(unobs_ids)
        X_pred = self.predictor.pred(X_unobs)

        best_id = -1
        best_score = -np.inf

        # TODO: batch with chunking
        for i, cid in enumerate(unobs_ids):
            x = X_pred[i]  # (d_obj,)

            diff = Y - x               # (n, d_obj)
            dist = (diff * diff).sum(axis=1)  # (n,)
            score = np.sum((dist - d * sigma) * np.exp(-dist / (2 * sigma)))

            if score > best_score:
                best_score = score
                best_id = int(cid)

        if self.verbose:
            self.passed_time_blox2 += time.perf_counter() - t0

        return best_id

    def best_stein_novelty_repli_id(self) -> int:
        """For validation purpose. Not used for the selection."""
        t0 = time.perf_counter()
        
        Y = self.Y_obs
        unobs_ids = self.unobs_ids()
        X_unobs = self.X_unobs(unobs_ids)
        X_pred = self.predictor.pred(X_unobs)

        best_id = -1
        best_score = -np.inf
        for i, cid in enumerate(unobs_ids):
            s = stein_novelty_repli(X_pred[i], Y, self.squared_sigma)
            if s > best_score:
                best_score = s
                best_id = int(cid)

        if self.verbose:
            self.passed_time_repli += time.perf_counter() - t0

        return best_id
                
    def next_candidate_id(self) -> int:
        if self.Y_obs.size == 0:
            print("No observed point.")
            return None

        X_obs = self.X_obs()
        Y_obs = self.Y_obs
        self.predictor.fit(X_obs, Y_obs)

        best_id = self.best_stein_novelty_id()

        if self.verbose:
            best_id_valid = self.best_stein_novelty_repli_id()
            if best_id == best_id_valid:
                print(f"Same best point at {len(self.obs_ids)} observed points.")
            else:
                print(f"WARNING: Different best point at {len(self.obs_ids)} observed points.")
        return best_id