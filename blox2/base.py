from abc import ABC, abstractmethod
import numpy as np

class DataPoint(ABC):
    def __init__(self, evaluated_values=None):
        self.stein_novelty_equiv = 0
        self.evaluated_values: np.ndarray = evaluated_values

    @abstractmethod
    def feature_vector(self) -> np.ndarray:
        pass
    
class RawFeature(DataPoint):
    def __init__(self, feature_vector: np.ndarray, evaluated_values=None):
        super().__init__(evaluated_values)
        self._feature_vector = feature_vector
        
    def feature_vector(self):
        return self.feature_vector

class Selector(ABC):
    def __init__(self, observed: list[DataPoint], unchecked: list[DataPoint]):
        self.observed = observed
        self.unchecked = unchecked
    
    @abstractmethod
    def next_candidate() -> DataPoint:
        pass
    
class Predictor(ABC):
    @abstractmethod
    def fit(observed_samples: list[DataPoint]):
        pass
    
    @abstractmethod
    def pred(x: DataPoint) -> list[np.ndarray]:
        """Return samples of predicted objective values. For point estimation, return [np.ndarray] (single entry list)."""
        pass
    
    def prep_data(observed_samples: list[DataPoint]):
        X = np.vstack([np.asarray(s.feature_vector(), float).ravel() for s in observed_samples])
        Y = np.vstack([np.asarray(s.evaluated_values, float).ravel() for s in observed_samples])
        return X, Y
