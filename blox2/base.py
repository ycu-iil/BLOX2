from abc import ABC, abstractmethod
import numpy as np

class DataPoint(ABC):
    def __init__(self, observed_values=None):
        self.stein_novelty_equiv = 0
        self.observed_values: np.ndarray = observed_values

    @abstractmethod
    def feature_vector(self) -> np.ndarray:
        pass
    
class RawFeature(DataPoint):
    def __init__(self, feature_vector: np.ndarray, id: int=None, observed_values=None):
        super().__init__(observed_values)
        self.id = id
        self._feature_vector = feature_vector
        
    def feature_vector(self):
        return self.feature_vector

class Selector(ABC):
    def __init__(self, observed_points: list[DataPoint], unchecked_points: list[DataPoint]):
        self.observed_points = observed_points
        self.unchecked_points = unchecked_points
    
    @abstractmethod
    def next_candidate() -> DataPoint:
        pass
    
class Predictor(ABC):
    @abstractmethod
    def fit(observed_points: list[DataPoint]):
        pass
    
    @abstractmethod
    def pred(x: DataPoint) -> list[np.ndarray]:
        """Return samples of predicted objective values. For point estimation, return [np.ndarray] (single entry list)."""
        pass
    
    def prep_data(observed_points: list[DataPoint]):
        X = np.vstack([np.asarray(s.feature_vector(), float).ravel() for s in observed_points])
        Y = np.vstack([np.asarray(s.observed_values, float).ravel() for s in observed_points])
        return X, Y
