import matplotlib.pyplot as plt
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

def _buffer_union_geometry(pts: np.ndarray, radius: float, resolution: int=4, cap_style: int=1, join_style: int=1):
    """
    Build the union of circular buffers around points.

    Args:
        pts : Array of points of shape (k, 2).
        radius : Buffer radius (same units as pts).
        resolution : Buffer resolution (higher -> smoother circle, slower).
        cap_style : Shapely cap style (1=round, 2=flat, 3=square).
        join_style : Shapely join style (1=round, 2=mitre, 3=bevel).

    Returns:
        shapely geometry: Polygon or MultiPolygon representing the union.
    """
    try:
        from shapely.geometry import Point, Polygon
        from shapely.ops import unary_union
    except Exception as e:
        raise ImportError("This code requires shapely. Install with `uv pip install shapely`.") from e
    pts = np.asarray(pts, dtype=float)
    if pts.size == 0:
        return Polygon() # empty polygon

    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")

    # Create buffers for each point and union them.
    geoms = [Point(float(x), float(y)).buffer(radius, resolution=resolution, cap_style=cap_style, join_style=join_style) for x, y in pts]
    return unary_union(geoms)


def _iter_polygons(geom):
    """Yield polygons from Polygon / MultiPolygon."""
    try:
        from shapely.geometry import Polygon, MultiPolygon
        from shapely.ops import unary_union
    except Exception as e:
        raise ImportError("This code requires shapely. Install with `uv pip install shapely`.") from e
    
    if geom.is_empty:
        return
    if isinstance(geom, Polygon):
        yield geom
    elif isinstance(geom, MultiPolygon):
        for g in geom.geoms:
            yield g
    else:
        if hasattr(geom, "geoms"):
            for g in geom.geoms:
                yield from _iter_polygons(g)

def buffer_union_area_trajectory(X: np.ndarray, radius: float, resolution: int=4, cap_style: int=1, join_style: int=1, start_k: int=1, step: int=1) -> np.ndarray:
    """
    Compute area trajectory of union-of-buffers over prefixes of X.

    Args:
        X : Input points of shape (N, 2). Interpreted as prefixes: X[:k].
        radius : Buffer radius around each point.
        resolution : Buffer resolution (higher -> smoother circle, slower).
        cap_style : Shapely cap style (1=round, 2=flat, 3=square).
        join_style : Shapely join style (1=round, 2=mitre, 3=bevel).
        start_k : First k to start computing from (default 1). For k < start_k -> 0.
        step: Calculation interval.
    """
    try:
        from shapely.geometry.base import BaseGeometry
        from shapely.geometry import Point
        from shapely.ops import unary_union
    except Exception as e:
        raise ImportError("This code requires shapely. Install with `pip install shapely`.") from e

    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"X must be shape (N,2), got {X.shape}")

    N = X.shape[0]
    areas = np.zeros(N, dtype=float)
    if N == 0:
        return areas

    if start_k < 1 or start_k > N:
        raise ValueError(f"start_k must be in [1, N], got {start_k} with N={N}")
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")

    union_geom: BaseGeometry = None
    last_area = 0.0
    last_added_i = start_k - 1

    for i in range(N):
        k = i + 1
        if k < start_k:
            areas[i] = 0.0
            continue

        do_update = ((k - start_k) % step == 0) or (k == N)
        if do_update:
            pts = X[last_added_i : i + 1] # Add points from last_added_i .. i inclusive
            buffers = [Point(px, py).buffer(radius, resolution=resolution, cap_style=cap_style, join_style=join_style) for px, py in pts]
            batch = unary_union(buffers) # union within the batch first
            union_geom = batch if union_geom is None else union_geom.union(batch)

            last_added_i = i + 1
            last_area = float(getattr(union_geom, "area", 0.0))

        areas[i] = last_area

    return areas

def plot_buffer_union(X: np.ndarray, radius: float, k: int=None, ax=None, resolution: int=4, cap_style: int=1, join_style: int=1,
    show_points: bool=True, show_boundary: bool=True, show_fill: bool=True,
    fill_alpha: float=0.25, boundary_lw: float=2.0,
    point_kwargs: dict=None, boundary_kwargs: dict=None, fill_kwargs: dict=None, equal_aspect: bool=True) -> tuple:
    """
    Visualize the union-of-buffers region for X[:k].

    Args:
        X : Input points of shape (N, 2).
        radius : Buffer radius.
        k : If given, plot using only the prefix X[:k]. If None, use all points.
        ax : Matplotlib axes. If None, creates a new figure+axes.
        resolution : Buffer resolution for circles.
        cap_style : Shapely cap style (1=round, 2=flat, 3=square).
        join_style : Shapely join style (1=round, 2=mitre, 3=bevel).
        show_points : Whether to scatter points.
        show_boundary : Whether to draw polygon boundary lines.
        show_fill : Whether to fill polygon interiors.
        fill_alpha : Alpha for fill.
        boundary_lw : Line width for boundaries.
        point_kwargs : kwargs forwarded to ax.scatter.
        boundary_kwargs : kwargs forwarded to ax.plot for boundaries.
        fill_kwargs : kwargs forwarded to ax.fill for fills.
        equal_aspect : If True, set ax.set_aspect("equal", adjustable="box").

    Returns:
        (ax, geom): Matplotlib axes and the shapely union geometry plotted.
    """
    N = X.shape[0]

    if k is None:
        k = N
    if not (0 <= k <= N):
        raise ValueError(f"k must be in [0, N], got {k} with N={N}")

    pts = X[:k]

    if ax is None:
        _, ax = plt.subplots()

    geom = _buffer_union_geometry(pts, radius, resolution=resolution, cap_style=cap_style, join_style=join_style)

    # Defaults
    if point_kwargs is None:
        point_kwargs = dict(s=12)
    if boundary_kwargs is None:
        boundary_kwargs = dict()
    if fill_kwargs is None:
        fill_kwargs = dict()

    if not geom.is_empty:
        for poly in _iter_polygons(geom):
            # Fill exterior
            if show_fill:
                x, y = poly.exterior.xy
                ax.fill(x, y, alpha=fill_alpha, **fill_kwargs)
                for ring in poly.interiors:
                    hx, hy = ring.xy
                    if show_boundary:
                        ax.plot(hx, hy, lw=boundary_lw, **boundary_kwargs)

            # Boundary exterior
            if show_boundary:
                x, y = poly.exterior.xy
                ax.plot(x, y, lw=boundary_lw, **boundary_kwargs)

    # Plot points
    if show_points and k > 0:
        ax.scatter(pts[:, 0], pts[:, 1], **point_kwargs)

    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    ax.set_title(f"Union of buffers (ε={radius})")
    return ax, geom