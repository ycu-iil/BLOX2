import numpy as np
from blox2 import DataPoint, Selector, Predictor

class BLOX2Selector(Selector):
    def __init__(self, squared_sigma: float, observed: list[DataPoint], unchecked: list[DataPoint], predictor: Predictor, sn_validation: bool=False):
        super().__init__(observed, unchecked)
        self.squared_sigma = squared_sigma
        self.predictor = predictor
        self.sn_validation = sn_validation
        
        # initialize stein novelty
        self.initialize_stein_novelty_equivs()
    
    def negative_hesgau_equiv(self, x, y):
        dist = np.sum(np.power(x-y, 2))
        return dist * np.exp(-dist/(2 * self.squared_sigma))
        
    def refresh_stein_novelty_equivs(self):
        self.predictor.fit(self.observed)
        for p in self.unchecked:
            x_pred = self.predictor.pred(p)[0] # TODO: distribution compat
            y = self.observed[-1].evaluated_values
            p.stein_novelty_equiv += self.negative_hesgau_equiv(x_pred, y)
            
    def initialize_stein_novelty_equivs(self):
        self.predictor.fit(self.observed)
        for p in self.unchecked:
            x_pred = self.predictor.pred(p)[0] # TODO: distribution compat
            for q in self.observed:
                y = q.evaluated_values
                p.stein_novelty_equiv += self.negative_hesgau_equiv(x_pred, y)
            
    def best_stein_novelty_debug(self) -> DataPoint:
        """
        Assumes deterministic point estimation for validation purposes.
        From the original BLOX: https://github.com/tsudalab/BLOX/blob/master/
        """
        def hesgau_repli(x, y, sigma):
            dim = len(x)
            dist = np.sum(np.power(x-y, 2))
            return (dim/sigma - dist/sigma**2)*np.exp(-dist/(2*sigma))
        
        def stein_novelty_repli(point, data_list, sigma):
            n = len(data_list)
            score = 0
            score = np.sum([hesgau_repli(point, data_list[k,:], sigma) for k in range(n)])
            score = score/(n*(n+1)/2)
            return -score
        
        data_list = [q.evaluated_values for q in self.observed]
        best_point = max(self.unchecked, key=lambda p: stein_novelty_repli(self.predictor.pred(p)[0], data_list, self.squared_sigma))
        return best_point
                
    def next_candidate(self) -> DataPoint:
        if self.observed[-1].evaluated_values is None:
            print("The last candidate is not evaluated yet.") # TODO: use logger
            return None
        else:
            self.refresh_stein_novelty_equivs()
            best_point = max(self.unchecked, key=lambda p: p.stein_novelty_equiv)
            if self.sn_validation:
                best_point_valid = self.best_stein_novelty_debug()
                if best_point == best_point_valid:
                    print(f"Same best point at {len(self.observed)} observed points.")
                else:
                    print(f"WARNING: Different best point at {len(self.observed)} observed points.")
            return best_point
