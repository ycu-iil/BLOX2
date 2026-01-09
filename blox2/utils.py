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