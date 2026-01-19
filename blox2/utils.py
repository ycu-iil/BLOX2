from pathlib import Path
import numpy as np
import pandas as pd

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
        return pd.read_csv(path, header=header).to_numpy()

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