from abc import ABC, abstractmethod
import numpy as np

class DataPoint(ABC):
    def __init__(self, id: int, observed_values=None):
        self.id = id
        self.observed_values: np.ndarray = observed_values

    @abstractmethod
    def feature_vector(self) -> np.ndarray:
        pass
    
class RawFeature(DataPoint):
    def __init__(self, feature_vector: np.ndarray, id: int, observed_values=None):
        super().__init__(id, observed_values)
        self._feature_vector = feature_vector
        
    def feature_vector(self):
        return self._feature_vector

class Selector(ABC):
    def __init__(self, observed_points: list[DataPoint], unchecked_points: list[DataPoint]):
        self.observed_points = observed_points
        self.unchecked_points = unchecked_points
    
    @abstractmethod
    def next_candidate() -> DataPoint:
        pass
    
    def observe(self, id: int, observed_values: np.ndarray):
        for i, p in enumerate(self.unchecked_points):
            if p.id == id:
                p.observed_values = np.asarray(observed_values)
                self.observed_points.append(p)
                del self.unchecked_points[i]
                return
        raise ValueError(f"DataPoint ID {id} not found in unchecked points.")
    
class Predictor(ABC):
    @abstractmethod
    def fit(observed_points: list[DataPoint]):
        pass
    
    @abstractmethod
    def pred(x: DataPoint) -> list[np.ndarray]:
        """Return samples of predicted objective values. For point estimation, return [np.ndarray] (single entry list)."""
        pass
    
    @staticmethod
    def prep_data(observed_points: list[DataPoint]):
        X = np.vstack([np.asarray(p.feature_vector(), float).ravel() for p in observed_points])
        Y = np.vstack([np.asarray(p.observed_values, float).ravel() for p in observed_points])
        return X, Y
