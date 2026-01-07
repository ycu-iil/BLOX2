import numpy as np
from .base import DataPoint, Selector, Predictor
from .utils import stein_novelty_repli

class BLOX2Selector(Selector):
    def __init__(self, observed_points: list[DataPoint], unchecked_points: list[DataPoint], predictor: Predictor, squared_sigma: float=0.1, verbose: bool=False):
        super().__init__(observed_points, unchecked_points)
        self.squared_sigma = squared_sigma
        self.predictor = predictor
        self.verbose = verbose
        
        # initialize stein novelty
        self.initialize_stein_novelty_equivs()
    
    def negative_hesgau_equiv(self, x, y):
        dist = np.sum((x - y)**2)
        d = x.shape[0]
        return (dist - d * self.squared_sigma) * np.exp(-dist/(2*self.squared_sigma))
    
    def refresh_stein_novelty_equivs(self):
        self.predictor.fit(self.observed_points)
        for p in self.unchecked_points:
            x_pred = self.predictor.pred(p)[0] # TODO: distribution compat
            y = self.observed_points[-1].observed_values
            p.stein_novelty_equiv += self.negative_hesgau_equiv(x_pred, y)
            
    def initialize_stein_novelty_equivs(self):
        self.predictor.fit(self.observed_points)
        for p in self.unchecked_points:
            x_pred = self.predictor.pred(p)[0] # TODO: distribution compat
            for q in self.observed_points:
                y = q.observed_values
                p.stein_novelty_equiv += self.negative_hesgau_equiv(x_pred, y)
            
    def best_stein_novelty_repli(self) -> DataPoint:
        """
        Assumes deterministic point estimation for validation purposes.
        predictor.fit() is called in refresh_stein_novelty_equivs(), which should be called before this method.
        """
        data_list = np.vstack([q.observed_values for q in self.observed_points])
        best_point = max(self.unchecked_points, key=lambda p: stein_novelty_repli(self.predictor.pred(p)[0], data_list, self.squared_sigma))
        return best_point
                
    def next_candidate(self) -> DataPoint:
        if self.observed_points[-1].observed_values is None:
            print("The last entry of observed_points is not observed yet.") # TODO: use logger
            return None
        else:
            self.refresh_stein_novelty_equivs()
            best_point = max(self.unchecked_points, key=lambda p: p.stein_novelty_equiv)
            if self.verbose:
                best_point_valid = self.best_stein_novelty_repli()
                if best_point == best_point_valid:
                    print(f"Same best point at {len(self.observed_points)} observed points.")
                else:
                    print(f"WARNING: Different best point at {len(self.observed_points)} observed points.")
                    data_list = np.vstack([q.observed_values for q in self.observed_points])
                    for p in self.unchecked_points:
                        print(f"used: {p.stein_novelty_equiv}, repli: {stein_novelty_repli(self.predictor.pred(p)[0], data_list, self.squared_sigma)}")
            return best_point
