from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def hesgau_repli(x, y, sigma):
    """Ported for validation purpose, and not used in actual selection. From the original BLOX: https://github.com/tsudalab/BLOX/blob/master/"""
    dim = len(x)
    dist = np.sum(np.power(x-y, 2))
    return (dim/sigma - dist/sigma**2)*np.exp(-dist/(2*sigma))

def stein_novelty_repli(point, data_list, sigma):
    """Ported for validation purpose, and not used in actual selection. From the original BLOX: https://github.com/tsudalab/BLOX/blob/master/"""
    n = len(data_list)
    score = 0
    score = np.sum([hesgau_repli(point, data_list[k,:], sigma) for k in range(n)])
    score = score/(n*(n+1)/2)
    return -score

def split_df_by_n_rows(df: pd.DataFrame, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n < 0:
        raise ValueError("n must be non-negative")

    df_head = df.iloc[:n].copy()
    df_tail = df.iloc[n:].copy()
    return df_head, df_tail

def load_features(path: str, header=None) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path, header=header)

    elif suffix == ".npz":
        z = np.load(path, allow_pickle=True)

        # needs to be "features_1", "features_2", ...
        feature_keys = sorted(
            [k for k in z.files if k.startswith("features_")],
            key=lambda s: int(s.split("_")[1]),
        )
        
        if not feature_keys:
            raise ValueError("NPZ has no features_* arrays.")
        
        feats = [z[k] for k in feature_keys]

        X = np.hstack(feats)
        return pd.DataFrame(X)
    else:
        raise ValueError(f"Unsupported feature format: {suffix}")
    
def make_scaler(scale: np.ndarray | StandardScaler, d: int) -> StandardScaler:
    if isinstance(scale, StandardScaler):
        return scale

    S = np.asarray(scale)
    if S.ndim != 2:
        raise ValueError(f"scale ndarray must be 2D (n_scale, d); got shape {S.shape}")
    if S.shape[1] != d:
        raise ValueError(f"scale has d={S.shape[1]} but observation_history has d={d}")

    return StandardScaler().fit(S)

def make_scaled_trajectory(initial_observed_properties_path, observation_histories_path, all_properties_path) -> np.ndarray:
    """
    Returns ndarray of shape (n_observed, d_properties): History of observed properties (including initial points), scaled with all properties (including unobserved ones).
    """
    # read data
    df1 = pd.read_csv(initial_observed_properties_path, header=None)
    df2 = pd.read_csv(observation_histories_path, header=None)
    df = pd.concat([df1, df2], axis=0, ignore_index=True)
    
    # scale with all props
    all_props = pd.read_csv(all_properties_path).to_numpy()
    scaler = make_scaler(all_props, d=2)
    trajectory = df.to_numpy()
    trajectory = scaler.transform(trajectory)

    return trajectory