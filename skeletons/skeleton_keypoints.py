# skeleton_keypoints.py
import numpy as np


def compute_discrete_curvature(points):
    """
    Compute a simple discrete curvature measure along a 3D polyline.

    points: (N, 3) array, ordered along the branch.
    Returns:
        kappa: (N,) array of curvature values (0 at endpoints).
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    kappa = np.zeros(n, dtype=float)
    if n < 3:
        return kappa

    # For each interior point i, measure change of direction
    for i in range(1, n - 1):
        v1 = points[i] - points[i - 1]
        v2 = points[i + 1] - points[i]

        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            continue

        # Normalize
        v1 /= n1
        v2 /= n2

        # Curvature proxy: magnitude of change in direction
        # = ||v1 - v2||, in [0, 2]
        kappa[i] = np.linalg.norm(v1 - v2)

    return kappa


def detect_curvature_keypoints(
    points,
    curvature=None,
    percentile=80.0,
    min_index_distance=2,
):
    """
    Detect curvature keypoints on a branch.

    points: (N, 3) array, ordered along the branch.
    curvature: optional (N,) array. If None, it will be computed.
    percentile: keep only points whose curvature is above this percentile
                among all interior points (e.g., 80 means top 20%).
    min_index_distance: minimal index distance between two keypoints
                        (non-maxima suppression along the chain).

    Returns:
        key_indices: list of indices in [0, N-1]
    """
    points = np.asarray(points, dtype=float)
    if curvature is None:
        curvature = compute_discrete_curvature(points)

    n = len(points)
    if n < 3:
        return []

    # Use only interior points for thresholding
    interior_kappa = curvature[1:-1]
    valid = interior_kappa > 0
    if not np.any(valid):
        return []

    thresh = np.percentile(interior_kappa[valid], percentile)

    # Candidate indices: local maxima above threshold
    candidates = []
    for i in range(1, n - 1):
        if curvature[i] < thresh:
            continue
        if curvature[i] >= curvature[i - 1] and curvature[i] >= curvature[i + 1]:
            candidates.append(i)

    # Non-maximum suppression along index axis
    key_indices = []
    for idx in candidates:
        if len(key_indices) == 0 or idx - key_indices[-1] >= min_index_distance:
            key_indices.append(idx)
        else:
            # If two candidates are too close, keep the one with larger curvature
            if curvature[idx] > curvature[key_indices[-1]]:
                key_indices[-1] = idx

    return key_indices


def detect_radius_change_keypoints(
    radii,
    percentile=80.0,
    min_index_distance=2,
):
    """
    Detect radius-change keypoints along a branch.

    radii: (N,) array of radius values at each point along the branch.
           For example, distance from skeleton point to the surface.
    percentile: keep only points whose |dr/ds| is above this percentile.
    min_index_distance: minimal index distance between two keypoints
                        (non-maxima suppression).

    Returns:
        key_indices: list of indices in [0, N-1]
    """
    radii = np.asarray(radii, dtype=float)
    n = len(radii)
    if n < 3:
        return []

    # Finite difference derivative along the chain (first-order)
    # We center it such that grad[i] roughly corresponds to point i
    grad = np.zeros(n, dtype=float)
    for i in range(1, n - 1):
        grad[i] = (radii[i + 1] - radii[i - 1]) * 0.5

    abs_grad = np.abs(grad[1:-1])
    valid = abs_grad > 0
    if not np.any(valid):
        return []

    thresh = np.percentile(abs_grad[valid], percentile)

    candidates = []
    for i in range(1, n - 1):
        if abs(grad[i]) < thresh:
            continue
        # local maximum of |grad| (strictly or equal)
        if abs(grad[i]) >= abs(grad[i - 1]) and abs(grad[i]) >= abs(grad[i + 1]):
            candidates.append(i)

    # Non-maximum suppression along index axis
    key_indices = []
    for idx in candidates:
        if len(key_indices) == 0 or idx - key_indices[-1] >= min_index_distance:
            key_indices.append(idx)
        else:
            if abs(grad[idx]) > abs(grad[key_indices[-1]]):
                key_indices[-1] = idx

    return key_indices


def merge_keypoints_by_proximity(
    keypoints_dict,
    merge_radius=0.02,
    priority_order=None,
):
    """
    Merge keypoints from different categories if they are spatially close.

    keypoints_dict:
        dict from type_name -> list/array of (3,) point coordinates
        e.g. {
            "endpoint": [(...), (...), ...],
            "junction": [...],
            "curvature": [...],
            "radius_change": [...],
        }

    merge_radius:
        If two keypoints are within this Euclidean distance, they will be merged
        into a single "group" with a type set containing both types.

    priority_order:
        Optional list specifying a priority ordering of types, e.g.
        ["junction", "endpoint", "radius_change", "curvature"].
        This can be used later to choose a single "display type".

    Returns:
        merged:
            list of dicts, each dict has:
            {
                "position": np.array(3),
                "types": set([...]),
            }

        If priority_order is provided, each dict also has:
            "display_type": str
    """
    # Flatten all keypoints into (position, type)
    all_points = []
    for t, pts in keypoints_dict.items():
        if pts is None:
            continue
        pts = np.asarray(pts, dtype=float)
        for p in pts:
            all_points.append((p, t))

    if not all_points:
        return []

    clusters = []  # each: {"position": np.array(3), "types": set([...]), "count": int}

    for p, t in all_points:
        assigned = False
        for cluster in clusters:
            center = cluster["position"]
            dist = np.linalg.norm(p - center)
            if dist <= merge_radius:
                # Merge into this cluster, update running average of position
                c = cluster["count"]
                cluster["position"] = (center * c + p) / (c + 1)
                cluster["count"] = c + 1
                cluster["types"].add(t)
                assigned = True
                break

        if not assigned:
            clusters.append(
                {
                    "position": p.copy(),
                    "types": {t},
                    "count": 1,
                }
            )

    # Convert to final representation
    merged = []
    for cluster in clusters:
        entry = {
            "position": cluster["position"],
            "types": set(cluster["types"]),
        }
        merged.append(entry)

    # Optionally decide a "display_type" based on priority_order
    if priority_order is not None:
        for entry in merged:
            display_type = None
            for t in priority_order:
                if t in entry["types"]:
                    display_type = t
                    break
            # If none of the priority types appear, just pick one arbitrarily
            if display_type is None and len(entry["types"]) > 0:
                display_type = sorted(list(entry["types"]))[0]
            entry["display_type"] = display_type

    return merged
