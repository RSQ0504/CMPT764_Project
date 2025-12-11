import os
import time
import argparse
import subprocess
import numpy as np
from skimage import measure

from branch_utils import (
    read_skeleton_segments,
)
from skeleton_postprocess import get_clean_keypoints_from_segments

# Optional progress bar
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ============================================================
# NOTE:
# Curvature / radius-change keypoint related code is temporarily
# disabled. If you want to re-enable it in the future, you can
# add the corresponding functions back here.
# ============================================================

"""
# ----------------- curvature & radius keypoints (DISABLED) ----------------- #

def compute_discrete_curvature(points):
    ...
def detect_curvature_keypoints(...):
    ...
def detect_radius_change_keypoints(...):
    ...
"""

# ----------------- basic helpers ----------------- #

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
    points_unique = np.unique(points, axis=0)
    return points_unique  # (K, 3)


def write_off(vertices, faces, out_path):
    """
    Write a simple OFF file given vertices and triangular faces.
    vertices: (V, 3)
    faces: (F, 3) integer indices
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)

    with open(out_path, "w") as f:
        f.write("OFF\n")
        f.write(f"{len(vertices)} {len(faces)} 0\n")
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


# ----------------- mesh & skeleton generation (per npz) ----------------- #

def generate_mesh_and_skeleton_for_npz(
    npz_path,
    skeleton_bin="./MCF_Skeleton_example",
    level=0.5,
):
    """
    For a given npz file, generate its corresponding OFF mesh and skeleton TXT.

    It will create:
        <basename>_mc.off
        <basename>_mc.txt
    in the same directory as npz_path.

    Returns:
        off_path, txt_path
    """
    folder = os.path.dirname(npz_path)
    base = os.path.splitext(os.path.basename(npz_path))[0]
    off_path = os.path.join(folder, f"{base}_mc.off")
    txt_path = os.path.join(folder, f"{base}_mc.txt")

    off_exists = os.path.exists(off_path)
    txt_exists = os.path.exists(txt_path)

    if off_exists and txt_exists:
        # Already generated before, just reuse
        return off_path, txt_path

    with np.load(npz_path) as data:
        # 🔴 Change "voxels" to your actual voxel key if different.
        if "voxels" in data.files:
            voxels = data["voxels"]
        else:
            raise KeyError(
                f"'voxels' key not found in {npz_path}. "
                f"Available keys: {list(data.files)}"
            )

    voxels = np.asarray(voxels, dtype=float)

    # Marching cubes to generate mesh
    verts, faces, _, _ = measure.marching_cubes(
        volume=voxels,
        level=level,
        spacing=(1.0, 1.0, 1.0),
    )

    # Write OFF mesh
    write_off(verts, faces, off_path)

    # Call external skeletonization binary
    cmd = [skeleton_bin, off_path, txt_path]
    subprocess.run(cmd, check=True)

    return off_path, txt_path


# ----------------- skeleton keypoints extraction (per npz) ----------------- #

def compute_skeleton_and_keypoints_for_npz(
    npz_path,
    skeleton_bin="./MCF_Skeleton_example",
    max_gap=0.05,
    min_nodes=5,
    min_length=0.03,
):
    """
    For a given npz file, ensure its mesh & skeleton exist, then compute:
      - skeleton_segments: (M, 2, 3)
      - skeleton_points:   (K, 3), unique endpoints of segments
      - junction_points:   (J, 3)
      - endpoint_points:   (E, 3)

    Returns a dict of numpy arrays ready to be merged into this npz.
    """
    off_path, txt_path = generate_mesh_and_skeleton_for_npz(
        npz_path=npz_path,
        skeleton_bin=skeleton_bin,
    )

    obj_name = os.path.basename(os.path.dirname(npz_path))
    base_name = os.path.basename(npz_path)

    segments = read_skeleton_segments(txt_path)
    if len(segments) == 0:
        raise RuntimeError(f"No skeleton segments found for {obj_name}/{base_name}.")

    # Clean graph + topology keypoints
    clean_graph, junctions, endpoints = get_clean_keypoints_from_segments(
        segments,
        max_gap=max_gap,
        min_nodes=min_nodes,
        min_length=min_length,
    )

    # Topological keypoints
    junction_points = np.array(junctions, dtype=np.float64) if len(junctions) > 0 else np.zeros((0, 3))
    endpoint_points = np.array(endpoints, dtype=np.float64) if len(endpoints) > 0 else np.zeros((0, 3))

    # Skeleton segments as (M, 2, 3)
    skeleton_segments = np.array(segments, dtype=np.float64)  # shape (M, 2, 3)
    skeleton_points = segments_to_point_cloud(segments)       # (K, 3)

    print(f"[Info] {obj_name}/{base_name}: "
          f"#segments={len(skeleton_segments)}, "
          f"#junctions={len(junction_points)}, "
          f"#endpoints={len(endpoint_points)}")

    # Pack into a dict to be merged into npz
    result = {
        "skeleton_segments": skeleton_segments,  # (M, 2, 3)
        "skeleton_points": skeleton_points,      # (K, 3)
        "junction_points": junction_points,      # (J, 3)
        "endpoint_points": endpoint_points,      # (E, 3)
    }
    return result


# ----------------- main loop over train folder (per npz) ----------------- #

def process_train_folder(train_root,
                         out_root,
                         max_gap=0.05,
                         min_nodes=5,
                         min_length=0.03,
                         skeleton_bin="./MCF_Skeleton_example"):
    """
    For each npz file under train_root/<object>/, compute:
        - mesh (.off)
        - skeleton (.txt)
        - skeleton & topological keypoints (junctions & endpoints)
    and save new npz files under out_root with the same relative structure.

    This is a per-npz pipeline: one OFF/TXT per npz.
    """
    os.makedirs(out_root, exist_ok=True)

    obj_names = sorted(
        d for d in os.listdir(train_root)
        if os.path.isdir(os.path.join(train_root, d))
    )

    print(f"[Info] Found {len(obj_names)} object folders in train.")

    for obj_name in obj_names:
        obj_dir = os.path.join(train_root, obj_name)
        out_obj_dir = os.path.join(out_root, obj_name)
        os.makedirs(out_obj_dir, exist_ok=True)

        npz_files = sorted(
            f for f in os.listdir(obj_dir) if f.endswith(".npz")
        )
        if not npz_files:
            print(f"[Warn] No npz files found in {obj_dir}, skip.")
            continue

        print(f"\n[Info] Object: {obj_name}, #npz = {len(npz_files)}")

        # Progress bar / time status
        if tqdm is not None:
            iterator = tqdm(npz_files, desc=f"{obj_name}", unit="file")
        else:
            iterator = npz_files

        start_time = time.time()

        for idx, fname in enumerate(iterator, start=1):
            in_npz_path = os.path.join(obj_dir, fname)
            out_npz_path = os.path.join(out_obj_dir, fname)

            if tqdm is None:
                # Simple textual progress with elapsed time
                elapsed = time.time() - start_time
                print(f"[{obj_name}] {idx}/{len(npz_files)} "
                      f"Elapsed: {elapsed:.1f}s  File: {fname}")

            try:
                # Compute skeleton & keypoints for THIS npz
                skel_info = compute_skeleton_and_keypoints_for_npz(
                    npz_path=in_npz_path,
                    skeleton_bin=skeleton_bin,
                    max_gap=max_gap,
                    min_nodes=min_nodes,
                    min_length=min_length,
                )
            except Exception as e:
                print(f"[Error] Failed on {obj_name}/{fname}: {e}")
                continue

            # Load existing data
            with np.load(in_npz_path) as data:
                data_dict = {k: data[k] for k in data.files}

            # Merge skeleton info
            data_dict.update(skel_info)

            # Save to new npz (compressed)
            np.savez_compressed(out_npz_path, **data_dict)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build train_with_skeleton npz files (junctions & endpoints only, one mesh & skeleton per npz)."
    )
    parser.add_argument(
        "--train_root",
        type=str,
        default="../data/2025_test_models_processed",
        help="Root folder of training npz files (default: ../data/2025_test_models_processed).",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="../data/train_with_skeleton_test_data",
        help="Output root folder (default: ../data/train_with_skeleton_test_data).",
    )
    parser.add_argument(
        "--skeleton_bin",
        type=str,
        default="./MCF_Skeleton_example",
        help="Path to the skeletonization binary (default: ./MCF_Skeleton_example).",
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_train_folder(
        train_root=args.train_root,
        out_root=args.out_root,
        max_gap=args.max_gap,
        min_nodes=args.min_nodes,
        min_length=args.min_length,
        skeleton_bin=args.skeleton_bin,
    )
