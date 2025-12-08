import os
import random
import argparse
import numpy as np
import open3d as o3d


def spheres_from_points(points, radius=0.5, color=(1.0, 0.0, 0.0)):
    """
    Create small sphere meshes at given points for highlighting.
    points: (N, 3)
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


def build_skeleton_lineset_from_segments(segments, color=(0.0, 0.0, 1.0)):
    """
    Build an Open3D LineSet from skeleton_segments.
    segments: (M, 2, 3) numpy array
    """
    segments = np.asarray(segments, dtype=float)
    if segments.ndim != 3 or segments.shape[1:] != (2, 3):
        raise ValueError(f"segments shape should be (M, 2, 3), got {segments.shape}")

    # Flatten and get unique points to avoid duplicates
    flat_points = segments.reshape(-1, 3)  # (M*2, 3)
    unique_points, inverse = np.unique(flat_points, axis=0, return_inverse=True)
    lines = inverse.reshape(-1, 2)  # (M, 2)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(unique_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(
        np.tile(np.array(color, dtype=float), (lines.shape[0], 1))
    )
    return line_set


def load_object_point_cloud_from_npz(data, threshold=0.5):
    """
    Try to construct an object point cloud from npz data.

    Priority:
    1) 'points' field: assume (N, 3) point cloud
    2) 'xyz' field:    assume (N, 3)
    3) 'voxels' field: (D, H, W) occupancy grid, use occupied voxels as points

    Modify this function if your actual keys are different.
    """
    if "points" in data.files:
        pts = np.asarray(data["points"], dtype=float)
    elif "xyz" in data.files:
        pts = np.asarray(data["xyz"], dtype=float)
    elif "voxels" in data.files:
        vox = np.asarray(data["voxels"], dtype=float)
        # Use all voxels above a threshold as points
        idx = np.argwhere(vox > threshold)  # (N, 3) indices
        pts = idx.astype(float)
    else:
        raise KeyError(
            f"Cannot find 'points' or 'xyz' or 'voxels' in npz. "
            f"Available keys: {list(data.files)}"
        )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.paint_uniform_color([0.7, 0.7, 0.7])  # gray
    return pcd


def pick_random_npz(root):
    """
    Randomly pick one npz from train_with_skeleton structure:
        root/object_name/*.npz
    """
    all_npz = []
    for obj_name in os.listdir(root):
        obj_dir = os.path.join(root, obj_name)
        if not os.path.isdir(obj_dir):
            continue
        for fname in os.listdir(obj_dir):
            if fname.endswith(".npz"):
                all_npz.append(os.path.join(obj_dir, fname))

    if not all_npz:
        raise RuntimeError(f"No npz files found under {root}")

    npz_path = random.choice(all_npz)
    return npz_path


def debug_visualize_one_npz(npz_path):
    """
    Debug visualization for one npz file:

    1) Show object only (point cloud)
    2) Show object + skeleton + junction/endpoints
    """
    print(f"[Info] Debug visualize npz: {npz_path}")

    with np.load(npz_path) as data:
        keys = list(data.files)
        print("[Info] npz keys:", keys)

        # 1) Object point cloud
        obj_pcd = load_object_point_cloud_from_npz(data)

        # 2) Skeleton & keypoints (if available)
        if "skeleton_segments" in keys:
            segments = np.asarray(data["skeleton_segments"], dtype=float)
        else:
            segments = None
            print("[Warn] 'skeleton_segments' not found in npz.")

        if "skeleton_points" in keys:
            skeleton_points = np.asarray(data["skeleton_points"], dtype=float)
        else:
            skeleton_points = None
            print("[Warn] 'skeleton_points' not found in npz.")

        if "junction_points" in keys:
            junction_points = np.asarray(data["junction_points"], dtype=float)
        else:
            junction_points = np.zeros((0, 3), dtype=float)
            print("[Warn] 'junction_points' not found in npz.")

        if "endpoint_points" in keys:
            endpoint_points = np.asarray(data["endpoint_points"], dtype=float)
        else:
            endpoint_points = np.zeros((0, 3), dtype=float)
            print("[Warn] 'endpoint_points' not found in npz.")

    # ---------- Visualization ----------

    # A. Show object only
    print("[Info] Showing object only...")
    o3d.visualization.draw_geometries(
        [obj_pcd],
        window_name=f"Object only - {os.path.basename(npz_path)}",
    )

    # B. Show object + skeleton + keypoints
    geoms = [obj_pcd]

    # Skeleton lines (blue)
    if segments is not None:
        line_set = build_skeleton_lineset_from_segments(
            segments,
            color=(0.0, 0.0, 1.0),  # blue
        )
        geoms.append(line_set)

    # Junction points (red spheres)
    if junction_points.shape[0] > 0:
        jp_meshes = spheres_from_points(
            junction_points,
            radius=0.5,               # adjust according to your scale
            color=(1.0, 0.0, 0.0),    # red
        )
        geoms.extend(jp_meshes)

    # Endpoint points (green spheres)
    if endpoint_points.shape[0] > 0:
        ep_meshes = spheres_from_points(
            endpoint_points,
            radius=0.5,
            color=(0.0, 1.0, 0.0),    # green
        )
        geoms.extend(ep_meshes)

    print("[Info] Showing object + skeleton + keypoints...")
    o3d.visualization.draw_geometries(
        geoms,
        window_name=f"Object + Skeleton + Keypoints - {os.path.basename(npz_path)}",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Debug visualization for train_with_skeleton npz files."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="./train_with_skeleton",
        help="Root folder of processed npz files (default: ./train_with_skeleton).",
    )
    parser.add_argument(
        "--npz",
        type=str,
        default=None,
        help="Path to a specific npz file. If not provided, one will be chosen randomly.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.npz is not None:
        npz_path = args.npz
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"npz file not found: {npz_path}")
    else:
        npz_path = pick_random_npz(args.root)

    debug_visualize_one_npz(npz_path)
