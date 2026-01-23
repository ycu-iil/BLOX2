import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull, QhullError
from .utils import make_scaler

def stein_discrepancy_trajectory(observation_history: np.ndarray, scale: np.ndarray | StandardScaler, sigma: float, copy: bool=True) -> np.ndarray:
    """
    Compute the trajectory of the estimate of Stein discrepancy under a Gaussian kernel with a fixed scaler.

    Args:
        observe_history : Observation history as an array of shape (n_steps, d).
        scale : Either a fitted sklearn.preprocessing.StandardScaler, or an array
                of shape (n_scale, d) used to fit a fixed StandardScaler.
        sigma : Gaussian kernel bandwidth.
        copy : If True, copy input arrays when casting.

    Returns:
        (n_steps,) array.
    """
    # TODO: integrate with other metrics?
    if not (np.isfinite(sigma) and sigma > 0):
        raise ValueError(f"sigma must be a positive finite float; got {sigma}")
    
    sigma2 = sigma**2
    
    Y_hist = np.asarray(observation_history)
    if Y_hist.ndim != 2:
        raise ValueError(f"observe_history must be 2D (n_steps, d); got shape {Y_hist.shape}")

    n_steps, d = Y_hist.shape
    if n_steps == 0:
        return np.empty((0,))

    scaler = make_scaler(scale, d)
    Y = scaler.transform(Y_hist.copy() if copy else Y_hist)

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

def convex_hull_area_trajectory(X: np.ndarray, qhull_options: str="QJ") -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"X must be shape (N,2), got {X.shape}")

    N = X.shape[0]
    areas = np.zeros(N, dtype=float)

    for k in range(3, N + 1):
        pts = X[:k]
        try:
            hull = ConvexHull(pts, qhull_options=qhull_options)
            areas[k - 1] = float(hull.volume)  # 2D: area
        except QhullError: # collinear/degenerate etc. -> treat as area 0
            areas[k - 1] = 0.0

    return areas

def convex_hull_perimeter_trajectory(X: np.ndarray, qhull_options: str="QJ") -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"X must be shape (N,2), got {X.shape}")

    N = X.shape[0]
    perims = np.zeros(N, dtype=float)

    for k in range(3, N + 1):
        pts = X[:k]
        try:
            hull = ConvexHull(pts, qhull_options=qhull_options)
            perims[k - 1] = float(hull.area) # says area, but actually perimeter in 2D
        except QhullError:
            perims[k - 1] = 0.0

    return perims
