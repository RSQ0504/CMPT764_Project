import os
import argparse
import numpy as np
import open3d as o3d

# Assume this file is in the same directory as branch_utils.py
from branch_utils import (
    read_skeleton_segments,
    extract_skeleton_branches,
)
from skeleton_postprocess import get_clean_keypoints_from_segments


def segments_to_point_cloud(segments):
    """
    Convert a list of skeleton segments into a point cloud (unique endpoints).
    segments: [((x1, y1, z1), (x2, y2, z2)), ...]
    """
    points = []
    for p1, p2 in segments:
        points.append(p1)
        points.append(p2)
    points = np.array(points, dtype=np.float64)

    # Remove duplicate points
    points_unique = np.unique(points, axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_unique)
    return pcd


def spheres_from_points(points, radius=0.02, color=(1.0, 0.0, 0.0)):
    """
    Create small sphere meshes at given points for highlighting.
    points: list or array of shape (N, 3)
    radius: sphere radius
    color: (r, g, b), each in [0, 1]
    """
    points = np.asarray(points, dtype=float)
    meshes = []
    for p in points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(p)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color(color)
        meshes.append(sphere)
    return meshes


# ---------- Curvature & Radius Keypoints Utilities ----------

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

    for i in range(1, n - 1):
        v1 = points[i] - points[i - 1]
        v2 = points[i + 1] - points[i]

        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            continue

        v1 /= n1
        v2 /= n2

        # Curvature proxy: magnitude of change in direction, in [0, 2]
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
    grad = np.zeros(n, dtype=float)
    for i in range(1, n - 1):
        grad[i] = 0.5 * (radii[i + 1] - radii[i - 1])

    abs_grad = np.abs(grad[1:-1])
    valid = abs_grad > 0
    if not np.any(valid):
        return []

    thresh = np.percentile(abs_grad[valid], percentile)

    candidates = []
    for i in range(1, n - 1):
        if abs(grad[i]) < thresh:
            continue
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


# ---------- Main Visualization ----------

def visualize_skeleton_with_seg_points(
    folder,
    skeleton_txt_name="voxel_64_mc.txt",
    max_gap=0.05,
    min_nodes=5,
    min_length=0.03,
    curvature_percentile=80.0,
    radius_percentile=80.0,
):
    """
    Visualize the skeleton and its keypoints in a single Open3D window:
    - junctions (topological, from cleaned graph)
    - endpoints (topological, from cleaned graph)
    - high-curvature keypoints (geometric)
    - radius-change keypoints (geometric, based on distance to surface mesh)
    """
    txt_path = os.path.join(folder, skeleton_txt_name)
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Skeleton txt file not found: {txt_path}")

    print(f"[Info] Using skeleton file: {txt_path}")

    # 1. Read skeleton segments from txt
    segments = read_skeleton_segments(txt_path)

    # 2. Build clean graph and find junctions & endpoints
    clean_graph, junctions, endpoints = get_clean_keypoints_from_segments(
        segments,
        max_gap=max_gap,
        min_nodes=min_nodes,
        min_length=min_length,
    )

    print(f"[Info] Number of junctions (clean): {len(junctions)}")
    print(f"[Info] Number of endpoints (clean): {len(endpoints)}")

    # 3. Convert the whole skeleton to a point cloud (gray background)
    ske_pcd = segments_to_point_cloud(segments)
    ske_pcd.paint_uniform_color([0.7, 0.7, 0.7])
    geoms = [ske_pcd]

    # 4. Junctions -> red spheres
    if len(junctions) > 0:
        junctions_np = np.array(junctions, dtype=np.float64)
        junction_meshes = spheres_from_points(
            junctions_np,
            radius=0.5,
            color=(1.0, 0.0, 0.0),  # red
        )
        geoms.extend(junction_meshes)

    # 5. Endpoints -> green spheres
    if len(endpoints) > 0:
        endpoints_np = np.array(endpoints, dtype=np.float64)
        endpoint_meshes = spheres_from_points(
            endpoints_np,
            radius=0.5,
            color=(0.0, 1.0, 0.0),  # green
        )
        geoms.extend(endpoint_meshes)

    # ---------- New: curvature & radius-change keypoints ----------

    # 6. Reconstruct segments from clean_graph so that branches reflect cleaning
    clean_segments = []
    for u, nbrs in clean_graph.items():
        for v in nbrs:
            # avoid duplicate edges by enforcing an ordering
            if u < v:
                clean_segments.append((u, v))

    # 7. Extract branches from the cleaned segments
    branches = extract_skeleton_branches(clean_segments)
    print(f"[Info] Number of branches (clean): {len(branches)}")

    # 8. Prepare surface mesh and KDTree for radius computation
    mesh_path = os.path.join(folder, "voxel_64_mc.off")
    kdtree = None
    surface_pcd = None
    if os.path.exists(mesh_path):
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if mesh.has_triangles():
            mesh.compute_vertex_normals()
            # You can adjust the number of sample points as needed
            surface_pcd = mesh.sample_points_uniformly(number_of_points=10000)
            kdtree = o3d.geometry.KDTreeFlann(surface_pcd)
            print("[Info] Surface mesh loaded, KDTree built for radius estimation.")
        else:
            print("[Warn] Mesh has no triangles, skip radius-based keypoints.")
    else:
        print(f"[Warn] Mesh file not found: {mesh_path}. Skip radius-based keypoints.")

    curvature_keypoints = []
    radius_keypoints = []

    # 9. For each branch, detect curvature & radius-change keypoints
    for b_idx, branch in enumerate(branches):
        branch_points = np.asarray(branch, dtype=float)
        n = len(branch_points)
        if n < 3:
            continue

        # 9.1 Curvature keypoints
        kappa = compute_discrete_curvature(branch_points)
        curv_indices = detect_curvature_keypoints(
            branch_points,
            curvature=kappa,
            percentile=curvature_percentile,
            min_index_distance=2,
        )
        if len(curv_indices) > 0:
            curvature_keypoints.append(branch_points[curv_indices])

        # 9.2 Radius-change keypoints (only if KDTree is available)
        if kdtree is not None:
            radii = []
            for p in branch_points:
                # Find nearest surface point
                _, idx, dists = kdtree.search_knn_vector_3d(p, 1)
                # dists is squared distances
                radius = np.sqrt(dists[0])
                radii.append(radius)
            radii = np.asarray(radii, dtype=float)

            radius_indices = detect_radius_change_keypoints(
                radii,
                percentile=radius_percentile,
                min_index_distance=2,
            )
            if len(radius_indices) > 0:
                radius_keypoints.append(branch_points[radius_indices])

    # Concatenate keypoints from all branches
    if len(curvature_keypoints) > 0:
        curvature_keypoints = np.vstack(curvature_keypoints)
    else:
        curvature_keypoints = np.zeros((0, 3))

    if len(radius_keypoints) > 0:
        radius_keypoints = np.vstack(radius_keypoints)
    else:
        radius_keypoints = np.zeros((0, 3))

    print(f"[Info] Number of curvature keypoints: {len(curvature_keypoints)}")
    print(f"[Info] Number of radius-change keypoints: {len(radius_keypoints)}")

    # 10. Visualize curvature keypoints -> blue spheres
    if len(curvature_keypoints) > 0:
        curv_meshes = spheres_from_points(
            curvature_keypoints,
            radius=0.4,
            color=(0.0, 0.0, 1.0),  # blue
        )
        geoms.extend(curv_meshes)

    # 11. Visualize radius-change keypoints -> yellow spheres
    if len(radius_keypoints) > 0:
        rad_meshes = spheres_from_points(
            radius_keypoints,
            radius=0.4,
            color=(1.0, 1.0, 0.0),  # yellow
        )
        geoms.extend(rad_meshes)

    # 12. Visualize everything in one window
    o3d.visualization.draw_geometries(
        geoms,
        window_name=f"Skeleton Keypoints - {os.path.basename(folder)}",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize skeleton keypoints (junctions, endpoints, curvature, radius-change)."
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="./reference_models_processed/sofa",
        help="Folder path of the model, e.g. ./reference_models_processed/pot",
    )
    parser.add_argument(
        "--txt_name",
        type=str,
        default="voxel_64_mc.txt",
        help="Skeleton txt file name (default: voxel_64_mc.txt)",
    )
    parser.add_argument(
        "--max_gap",
        type=float,
        default=0.05,
        help="Max distance between endpoints to bridge gaps (default: 0.05).",
    )
    parser.add_argument(
        "--min_nodes",
        type=int,
        default=5,
        help="Minimum number of nodes to keep a component (default: 5).",
    )
    parser.add_argument(
        "--min_length",
        type=float,
        default=0.03,
        help="Minimum total edge length to keep a component (default: 0.03).",
    )
    parser.add_argument(
        "--curv_pct",
        type=float,
        default=80.0,
        help="Percentile for curvature keypoints (default: 80.0).",
    )
    parser.add_argument(
        "--radius_pct",
        type=float,
        default=80.0,
        help="Percentile for radius-change keypoints (default: 80.0).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualize_skeleton_with_seg_points(
        folder=args.folder,
        skeleton_txt_name=args.txt_name,
        max_gap=args.max_gap,
        min_nodes=args.min_nodes,
        min_length=args.min_length,
        curvature_percentile=args.curv_pct,
        radius_percentile=args.radius_pct,
    )
