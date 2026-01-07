import numpy as np
import pandas as pd
from .base import RawFeature

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

def hesgau_repli(x, y, sigma):
    """Defined for validation purpose, and not used in actual selection. From the original BLOX: https://github.com/tsudalab/BLOX/blob/master/"""
    dim = len(x)
    dist = np.sum(np.power(x-y, 2))
    return (dim/sigma - dist/sigma**2)*np.exp(-dist/(2*sigma))

def stein_novelty_repli(point, data_list, sigma):
    """Defined for validation purpose, and not used in actual selection. From the original BLOX: https://github.com/tsudalab/BLOX/blob/master/"""
    n = len(data_list)
    score = 0
    score = np.sum([hesgau_repli(point, data_list[k,:], sigma) for k in range(n)])
    score = score/(n*(n+1)/2)
    return -score