import numpy as np
import trimesh
import pyvista as pv
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree




def sample_points_from_mesh(mesh: trimesh.Trimesh, n_points=5000):
    points, _ = trimesh.sample.sample_surface_even(mesh, n_points)
    return points



def fit_cuboid(points: np.ndarray, eps=0.03, min_samples=10, ransac_iters=100):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    cuboids = []
    
    unique_labels = [l for l in np.unique(labels) if l != -1]
    
    for label in unique_labels:
        cluster_pts = points[labels == label]
        
        best_inliers = 0
        best_cuboid = None
        
        for _ in range(ransac_iters):
            if cluster_pts.shape[0] < 8:
                sample_pts = cluster_pts
            else:
                sample_pts = cluster_pts[np.random.choice(cluster_pts.shape[0], 8, replace=False)]
            
            aabb_min = sample_pts.min(axis=0)
            aabb_max = sample_pts.max(axis=0)
            
            inlier_mask = np.all((cluster_pts >= aabb_min) & (cluster_pts <= aabb_max), axis=1)
            inliers = inlier_mask.sum()
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_cuboid = {
                    "min": aabb_min,
                    "max": aabb_max,
                    "center": (aabb_min + aabb_max) / 2,
                    "size": aabb_max - aabb_min,
                    "label": label
                }
        
        if best_cuboid is not None:
            cuboids.append(best_cuboid)
    
    return cuboids


def visualize_meshes_and_cuboid_subplots(meshes, cuboids):
    # 1 row, 2 columns
    plotter = pv.Plotter(shape=(1, 2), title="PointCloud + Solid Cuboids")

    colors = ["red", "blue", "green", "yellow", "purple", "cyan", "orange"]


    plotter.subplot(0, 0)
    plotter.add_text("co-segment", font_size=12)

    for i, mesh in enumerate(meshes):
        if mesh is None:
            continue
        plotter.add_mesh(
            mesh,
            color=colors[i % len(colors)],
        )


    plotter.subplot(0, 1)
    plotter.add_text("Solid Cuboids", font_size=12)

    for i, cub in enumerate(cuboids):
        if cub is None:
            continue

        bounds = (
            cub["min"][0], cub["max"][0],
            cub["min"][1], cub["max"][1],
            cub["min"][2], cub["max"][2]
        )

        # solid cuboid mesh
        box = pv.Box(bounds=bounds)

        plotter.add_mesh(
            box,
            color=colors[i % len(colors)],
            opacity=1,
            show_edges=True
        )

    plotter.show()
    plotter.close()


def visualize_pointcloud_and_cuboid_subplots(sampled_points_by_label, cuboids):
    # 1 row, 2 columns
    plotter = pv.Plotter(shape=(1, 2), title="PointCloud + Solid Cuboids")

    colors = ["red", "blue", "green", "yellow", "purple", "cyan", "orange"]


    plotter.subplot(0, 0)
    plotter.add_text("Dense Point Cloud", font_size=12)

    for i, pts in enumerate(sampled_points_by_label):
        if pts is None:
            continue
        plotter.add_points(
            pts,
            color=colors[i % len(colors)],
            point_size=5
        )


    plotter.subplot(0, 1)
    plotter.add_text("Solid Cuboids", font_size=12)

    for i, cub in enumerate(cuboids):
        if cub is None:
            continue

        bounds = (
            cub["min"][0], cub["max"][0],
            cub["min"][1], cub["max"][1],
            cub["min"][2], cub["max"][2]
        )

        # solid cuboid mesh
        box = pv.Box(bounds=bounds)

        plotter.add_mesh(
            box,
            color=colors[i % len(colors)],
            opacity=1,
            show_edges=True
        )

    plotter.show()
    plotter.close()



def process_segmentation_meshes(vertices_list, triangles_list, sample_points=5000):

    all_meshes = []
    all_dense_points = []
    all_cuboids = []

    for v, t in zip(vertices_list, triangles_list):

        if v is None or len(v) == 0:
            all_meshes.append(None)
            all_dense_points.append(None)
            all_cuboids.append(None)
            continue

        mesh = trimesh.Trimesh(vertices=v, faces=t)
        all_meshes.append(mesh)

        pts = sample_points_from_mesh(mesh, sample_points)
        all_dense_points.append(pts)

        cub = fit_cuboid(pts)
        all_cuboids.append(cub)

    return all_meshes, all_dense_points, all_cuboids



def union_meshes(mesh_list):

    meshes = [m for m in mesh_list if m is not None]
    if len(meshes) == 0:
        return None
    return trimesh.util.concatenate(meshes)


def sample_mesh_points(mesh, n=5000):
    pts, _ = trimesh.sample.sample_surface_even(mesh, n)
    return pts


def sample_points_from_cuboid(cuboid, n=2000):
    mn = cuboid["min"]
    mx = cuboid["max"]
    return np.random.rand(n, 3) * (mx - mn) + mn


def sample_union_cuboids(cuboids, total_points=6000):
    each = max(total_points // len(cuboids), 50)
    pts_list = []
    for cub in cuboids:
        pts_list.append(sample_points_from_cuboid(cub, each))
    return np.vstack(pts_list)

def chamfer_l2(pts_a, pts_b):
    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)

    d_a_to_b, _ = tree_b.query(pts_a)
    d_b_to_a, _ = tree_a.query(pts_b)

    return np.mean(d_a_to_b**2) + np.mean(d_b_to_a**2)


def compute_union_cd(mesh_union, cuboids, n_points=6000):
    pts_mesh = sample_mesh_points(mesh_union, n_points)
    pts_pred = sample_union_cuboids(cuboids, n_points)
    return chamfer_l2(pts_mesh, pts_pred)

def cuboid_contains_points(cuboid, pts):
    mn, mx = cuboid["min"], cuboid["max"]
    return np.all((pts >= mn) & (pts <= mx), axis=1)



def compute_union_iou(mesh_union, cuboids, resolution=80):
    mins = [mesh_union.bounds[0]] + [c["min"] for c in cuboids]
    maxs = [mesh_union.bounds[1]] + [c["max"] for c in cuboids]
    mn = np.min(mins, axis=0)
    mx = np.max(maxs, axis=0)

    xs = np.linspace(mn[0], mx[0], resolution)
    ys = np.linspace(mn[1], mx[1], resolution)
    zs = np.linspace(mn[2], mx[2], resolution)
    grid = np.stack(np.meshgrid(xs, ys, zs), axis=-1).reshape(-1, 3)

    pred_mask = np.zeros(len(grid), dtype=bool)
    for cub in cuboids:
        pred_mask |= cuboid_contains_points(cub, grid)


    pitch = (mx - mn).max() / resolution
    vox = mesh_union.voxelized(pitch=pitch)

    mat = vox.matrix.astype(bool)
    dims = mat.shape

    origin = vox.transform[:3, 3]
    pitch = vox.transform[0, 0]  # override pitch to match voxel grid

    xs_v = origin[0] + (np.arange(dims[0]) + 0.5) * pitch
    ys_v = origin[1] + (np.arange(dims[1]) + 0.5) * pitch
    zs_v = origin[2] + (np.arange(dims[2]) + 0.5) * pitch

    gx = (grid[:, 0] - origin[0]) / pitch - 0.5
    gy = (grid[:, 1] - origin[1]) / pitch - 0.5
    gz = (grid[:, 2] - origin[2]) / pitch - 0.5 

    ix = np.floor(gx).astype(int)
    iy = np.floor(gy).astype(int)
    iz = np.floor(gz).astype(int)

    valid = (
        (ix >= 0) & (ix < dims[0]) &
        (iy >= 0) & (iy < dims[1]) &
        (iz >= 0) & (iz < dims[2])
    )

    mesh_mask = np.zeros(len(grid), dtype=bool)
    mesh_mask[valid] = mat[ix[valid], iy[valid], iz[valid]]


    inter = np.sum(pred_mask & mesh_mask)
    union = np.sum(pred_mask | mesh_mask)

    if union == 0:
        return 0.0
    return inter / union

def evaluate_union_multiple_meshes(mesh_list, cuboids, iou_resolution=64, cd_points=6000):
    mesh_union = union_meshes(mesh_list)

    iou = compute_union_iou(mesh_union, cuboids, resolution=iou_resolution)
    print(iou)

    cd = compute_union_cd(mesh_union, cuboids, n_points=cd_points)
    print(cd)

    return {
        "iou": float(iou),
        "cd": float(cd)
    }