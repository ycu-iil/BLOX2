import time
import numpy as np
from .base import DataPoint, Selector, Predictor
from .utils import stein_novelty_repli

class BLOX2Selector(Selector):
    def __init__(self, observed_points: list[DataPoint], unchecked_points: list[DataPoint], predictor: Predictor, squared_sigma: float=0.1, verbose: bool=False):
        super().__init__(observed_points, unchecked_points)
        self.squared_sigma = squared_sigma
        self.predictor = predictor
        self.verbose = verbose
        if verbose:
           self.passed_time_blox2 = 0
           self.passed_time_repli = 0 
        
    def best_stein_novelty(self) -> DataPoint:
        Y = np.vstack([q.observed_values for q in self.observed_points])
        t0 = time.perf_counter()
        n, d = Y.shape

        best_p = None
        best_score = -np.inf

        for p in self.unchecked_points:
            x = self.predictor.pred(p)[0]
            diff = Y - x
            dist = (diff * diff).sum(axis=1)
            score = np.sum((dist - d * self.squared_sigma) * np.exp(-dist / (2 * self.squared_sigma)))
            if score > best_score:
                best_score = score
                best_p = p

        if self.verbose:
            self.passed_time_blox2 += time.perf_counter() - t0
        return best_p
            
    def best_stein_novelty_repli(self) -> DataPoint:
        """
        Assumes deterministic point estimation for validation purposes.
        """
        data_list = np.vstack([q.observed_values for q in self.observed_points])
        t0 = time.perf_counter()
        best_point = max(self.unchecked_points, key=lambda p: stein_novelty_repli(self.predictor.pred(p)[0], data_list, self.squared_sigma))
        
        if self.verbose:
            self.passed_time_repli += time.perf_counter() - t0
        return best_point
                
    def next_candidate(self) -> DataPoint:
        if self.observed_points[-1].observed_values is None:
            print("The last entry of observed_points is not observed yet.") # TODO: use logger
            return None
        else:
            self.predictor.fit(self.observed_points)
            best_point = self.best_stein_novelty()
            if self.verbose:
                best_point_valid = self.best_stein_novelty_repli()
                if best_point == best_point_valid:
                    print(f"Same best point at {len(self.observed_points)} observed points.")
                else:
                    print(f"WARNING: Different best point at {len(self.observed_points)} observed points.")
            return best_point
