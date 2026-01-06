from abc import ABC, abstractmethod

class Point(ABC):
    def __init__(self, evaluation_results=None):
        self.stein_novelty_equiv = 0
        self.evaluation_results = evaluation_results

    @abstractmethod
    def feature_vector():
        pass

class Selector(ABC):
    def __init__(self, observed_samples: list[Point], unchecked_samples: list[Point]):
        self.observed_samples = observed_samples
        self.unchecked_samples = unchecked_samples
    
    @abstractmethod
    def next_candidate() -> Point:
        pass