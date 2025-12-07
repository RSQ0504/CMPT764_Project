import os
import argparse
import numpy as np
import open3d as o3d

# Assume this file is in the same directory as branch_utils.py
from branch_utils import (
    read_skeleton_segments,
    build_skeleton_graph,
    find_junction_and_endpoint_nodes,
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
    meshes = []
    for p in points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(p)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color(color)
        meshes.append(sphere)
    return meshes


def visualize_skeleton_with_seg_points(
    folder,
    skeleton_txt_name="voxel_64_mc.txt",
    max_gap=0.05,
    min_nodes=5,
    min_length=0.03,
):
    """
    Visualize the skeleton and its segmentation points
    (junctions and endpoints) in a single Open3D window.

    folder: directory containing the skeleton txt file
    skeleton_txt_name: skeleton text file name
    """
    txt_path = os.path.join(folder, skeleton_txt_name)
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Skeleton txt file not found: {txt_path}")

    print(f"[Info] Using skeleton file: {txt_path}")

    # 1. Read skeleton segments from txt
    segments = read_skeleton_segments(txt_path)

    # 2. Build graph and find junctions & endpoints
    clean_graph, junctions, endpoints = get_clean_keypoints_from_segments(
        segments,
        max_gap=max_gap,
        min_nodes=min_nodes,
        min_length=min_length,
    )
    # graph = build_skeleton_graph(segments)
    # junctions, endpoints = find_junction_and_endpoint_nodes(graph)

    print(f"[Info] Number of junctions: {len(junctions)}")
    print(f"[Info] Number of endpoints: {len(endpoints)}")

    # 3. Convert the whole skeleton to a point cloud (gray)
    ske_pcd = segments_to_point_cloud(segments)
    ske_pcd.paint_uniform_color([0.7, 0.7, 0.7])

    geoms = [ske_pcd]

    # 4. Junctions -> red spheres
    if len(junctions) > 0:
        junctions_np = np.array(junctions, dtype=np.float64)
        junction_meshes = spheres_from_points(
            junctions_np,
            radius=0.5,           # you can adjust this according to your scale
            color=(1.0, 0.0, 0.0)  # red
        )
        geoms.extend(junction_meshes)

    # 5. Endpoints -> green spheres
    if len(endpoints) > 0:
        endpoints_np = np.array(endpoints, dtype=np.float64)
        endpoint_meshes = spheres_from_points(
            endpoints_np,
            radius=0.5,          # slightly smaller
            color=(0.0, 1.0, 0.0)  # green
        )
        geoms.extend(endpoint_meshes)

    # 6. Visualize everything in one window
    o3d.visualization.draw_geometries(
        geoms,
        window_name=f"Skeleton Segmentation Points - {os.path.basename(folder)}",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize skeleton segmentation points (junctions & endpoints)."
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualize_skeleton_with_seg_points(args.folder, args.txt_name)
