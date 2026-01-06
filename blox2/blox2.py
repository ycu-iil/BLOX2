import numpy as np
from blox2 import Point, Selector

class BLOX2(Selector):
    def __init__(self, squared_sigma: float, observed_samples: list[Point], unchecked_samples: list[Point]):
        super().__init__(observed_samples, unchecked_samples)
        self.squared_sigma = squared_sigma
        
        # initialize stein novelty
        for p in unchecked_samples:
            for q in observed_samples[:-1]: # regard the last one as a new point
                p.refresh_stein_novelty_equiv(q, squared_sigma)
    
    def negative_hesgau_equiv(self, x, y):
        dist = np.sum(np.power(x-y, 2))
        return dist * np.exp(-dist/(2 * self.squared_sigma))
        
    def refresh_stein_novelty_equivs(self):
        for p in self.unchecked_samples:
            x_pred = self.pred(p)
            y = self.observed_samples[-1].evaluation_results
            p.stein_novelty_equiv += self.negative_hesgau_equiv(x_pred, y)
                
    def next_candidate(self) -> Point:
        if self.observed_samples[-1].evaluation_results is None:
            print("The last candidate is not evaluated yet.") # TODO: use logger
            return None
        else:
            self.refresh_stein_novelty_equivs()
            best_point = max(self.unchecked_samples, key=lambda p: p.stein_novelty_equiv)
            return best_point
        
    def pred(p: Point):
        pass
