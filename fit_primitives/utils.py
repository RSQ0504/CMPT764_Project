import numpy as np
import trimesh
import pyvista as pv
from sklearn.cluster import DBSCAN




def sample_points_from_mesh(mesh: trimesh.Trimesh, n_points=5000):
    points, _ = trimesh.sample.sample_surface_even(mesh, n_points)
    return points



def fit_cuboid(points: np.ndarray, eps=0.05, min_samples=10, ransac_iters=100):
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
