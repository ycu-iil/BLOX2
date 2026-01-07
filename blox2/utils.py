import numpy as np
import pandas as pd
from blox2 import RawFeature

def make_datapoints_from_features(observed_features: pd.DataFrame, observed_values: pd.DataFrame, unchecked_features: pd.DataFrame):
    observed_points = []
    unchecked_points = []
    n_observed = len(observed_features)
    
    for i in range(n_observed):
        feature_vector = np.asarray(observed_features.iloc[i])
        values = np.asarray(observed_values.iloc[i])
        observed_points.append(RawFeature(feature_vector=feature_vector, id=i, observed_values=values))
        
    for i in range(len(unchecked_features)):
        feature_vector = np.asarray(unchecked_features.iloc[i])
        unchecked_points.append(RawFeature(feature_vector=feature_vector, id=i+n_observed))
        
    return observed_points, unchecked_points