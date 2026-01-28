import numpy as np
from scipy.spatial import ConvexHull, Delaunay, QhullError
from .utils import make_scaler

def stein_discrepancy_trajectory(X: np.ndarray, sigma: float) -> np.ndarray:
    """
    Compute the trajectory of the estimate of Stein discrepancy under a Gaussian kernel.
    """
    if not (np.isfinite(sigma) and sigma > 0):
        raise ValueError(f"sigma must be a positive finite float; got {sigma}")
    
    sigma2 = sigma**2

    if X.ndim != 2:
        raise ValueError(f"observe_history must be 2D (n_steps, d); got shape {X.shape}")

    n_steps, d = X.shape
    if n_steps == 0:
        return np.empty((0,))

    inv_sigma2 = 1.0 / sigma2
    inv_sigma4 = inv_sigma2 * inv_sigma2

    # maintain sum over unordered pairs i<j of: f(i,j) = k_ij * (d/σ^2 - dist2_ij/σ^4), symmetric in i,j
    pair_sum = 0.0
    stein_hat = np.full((n_steps,), np.nan)

    for n in range(2, n_steps + 1):
        y_new = X[n - 1] # (d,)
        Y_prev = X[: n - 1] # (n-1, d)

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

def occupancy_trajectory(X: np.ndarray, X_all: np.ndarray=None, bins: int=50, bounds: tuple[tuple[float, float], tuple[float, float]]=None) -> np.ndarray:
    """
    (# unique grid cells hit by points) / (total grid cells), where the grid bounds are fixed by:
      - bounds if provided, else
      - X_all if provided, else
      - X (only observed data; not recommended for comparisons)

    Args:
        X : (N, 2) points in time/order.
        X_all : (M, 2) points used to fix the grid bounds (recommended).
        bins : number of bins per axis (bins x bins grid).
        bounds : ((xmin, xmax), (ymin, ymax)) explicit bounds (overrides X_all).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"X must be shape (N,2), got {X.shape}")
    if X_all is None:
        X_ref = X
    else:
        X_ref = np.asarray(X_all, dtype=float)
        if X_ref.ndim != 2 or X_ref.shape[1] != 2:
            raise ValueError(f"X_all must be shape (M,2), got {X_ref.shape}")

    if bounds is None:
        xmin, ymin = np.min(X_ref, axis=0)
        xmax, ymax = np.max(X_ref, axis=0)
    else:
        (xmin, xmax), (ymin, ymax) = bounds

    # range
    eps = 1e-12
    xr = max(xmax - xmin, eps)
    yr = max(ymax - ymin, eps)

    N = X.shape[0]
    occ = np.zeros(N, dtype=float)

    # map points -> integer cell indices (0..bins-1)
    # clip if points fall outside bounds
    def to_cell_ids(pts: np.ndarray) -> np.ndarray:
        u = (pts[:, 0] - xmin) / xr
        v = (pts[:, 1] - ymin) / yr
        ix = np.clip((u * bins).astype(int), 0, bins - 1)
        iy = np.clip((v * bins).astype(int), 0, bins - 1)
        # combine into single id
        return ix * bins + iy

    total_cells = bins * bins

    for k in range(1, N + 1):
        ids = to_cell_ids(X[:k])
        occ[k - 1] = np.unique(ids).size / total_cells

    return occ

def alpha_concave_hull_area_trajectory(X: np.ndarray, alpha: float=1.0, print_interval: int=None) -> np.ndarray:
    """
    Requires: alphashape, shapely
    Alpha concave hull area trajectory. Ref: https://arxiv.org/abs/1309.7829

    Uses Delaunay triangulation and keeps triangles whose circumradius R satisfies R <= alpha.
    The alpha-shape area is the sum of areas of kept Delaunay triangles (triangles are interior-disjoint in the triangulation).

    Args:
        X: (N, 2) array of points.
        alpha: Circumradius threshold. Larger -> closer to convex hull.
        qhull_options: Passed to scipy.spatial.Delaunay.
        treat_degenerate_as_zero: If True, degenerate steps return 0 area instead of raising.
    """
    try:
        import alphashape
    except:
        raise AttributeError("To use alpha_concave_hull_area_trajectory(), please install alphashape.")
    
    X = np.asarray(X, float)
    N = X.shape[0]
    areas = np.zeros(N)

    for k in range(3, N + 1):
        if print_interval is not None and k % print_interval==0:
            print(f"Step {k} finished.")

        shape = alphashape.alphashape(X[:k], alpha)
        areas[k - 1] = shape.area if not shape.is_empty else 0.0

    return areas