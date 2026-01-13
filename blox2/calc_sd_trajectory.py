import numpy as np
from sklearn.preprocessing import StandardScaler

def calc_stein_discrepancy_trajectory(observation_history: np.ndarray, scale: np.ndarray | StandardScaler, squared_sigma: float, copy: bool=True) -> np.ndarray:
    """
    Compute the trajectory of the estimate of Stein discrepancy under a Gaussian kernel with a fixed scaler.

    Args:
        observe_history : Observation history as an array of shape (n_steps, d).
        scale : Either a fitted sklearn.preprocessing.StandardScaler, or an array
                of shape (n_scale, d) used to fit a fixed StandardScaler.
        squared_sigma : Gaussian kernel bandwidth^2.
        copy : If True, copy input arrays when casting.

    Returns:
        (n_steps,) array, with NaN for n<2.
    """
    
    Y_hist = np.asarray(observation_history)
    if Y_hist.ndim != 2:
        raise ValueError(f"observe_history must be 2D (n_steps, d); got shape {Y_hist.shape}")

    n_steps, d = Y_hist.shape
    if n_steps == 0:
        return np.empty((0,))

    if not (np.isfinite(squared_sigma) and squared_sigma > 0):
        raise ValueError(f"squared_sigma must be a positive finite float; got {squared_sigma}")

    scaler = _make_scaler(scale, d)
    Y = scaler.transform(Y_hist.copy() if copy else Y_hist)

    sigma2 = squared_sigma
    inv_sigma2 = 1.0 / sigma2
    inv_sigma4 = inv_sigma2 * inv_sigma2

    # maintain sum over unordered pairs i<j of: f(i,j) = k_ij * (d/σ^2 - dist2_ij/σ^4), symmetric in i,j
    pair_sum = 0.0
    stein_hat = np.full((n_steps,), np.nan)

    for n in range(2, n_steps + 1):
        y_new = Y[n - 1] # (d,)
        Y_prev = Y[: n - 1] # (n-1, d)

        diff = Y_prev - y_new # (n-1, d)
        dist2 = np.einsum("ij,ij->i", diff, diff) # (n-1,)

        k = np.exp(-dist2 * (0.5 * inv_sigma2)) # (n-1,)
        f = k * (d * inv_sigma2 - dist2 * inv_sigma4) # (n-1,)

        pair_sum += f.sum()

        # uses ordered pairs i != j, so ordered_sum = 2 * pair_sum
        ordered_sum = 2.0 * pair_sum
        stein_hat[n - 1] = ordered_sum / (n * (n - 1))

    return stein_hat

def _make_scaler(scale: np.ndarray | StandardScaler, d: int) -> StandardScaler:
    if isinstance(scale, StandardScaler):
        return scale

    S = np.asarray(scale)
    if S.ndim != 2:
        raise ValueError(f"scale ndarray must be 2D (n_scale, d); got shape {S.shape}")
    if S.shape[1] != d:
        raise ValueError(f"scale has d={S.shape[1]} but observation_history has d={d}")

    return StandardScaler().fit(S)