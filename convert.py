import numpy as np
import pyvista as pv
from skimage import measure
from skimage.morphology import skeletonize, medial_axis



npz_path = "./reference_models_processed/dog/voxel_and_sdf.npz"
data = np.load(npz_path)
voxels = data['voxels']
print(voxels.shape)
sdf_points = data['sdf_points']
print(sdf_points.shape)
sdf_values = data['sdf_values']
print(sdf_values.shape)
centroid = data['centroid']
print(centroid)
scale = data['scale']
print(scale)

verts, faces, normals, values = measure.marching_cubes(voxels.astype(float), level=0.5)
faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
mesh_smooth = pv.PolyData(verts, faces)

plotter1 = pv.Plotter(title="1")
plotter1.add_mesh(mesh_smooth, color="white", opacity=0.5)
plotter1.show()

# # Voxel Mesh
# x, y, z = np.where(voxels)
# cubes = [pv.Cube(center=(xi, yi, zi), x_length=1, y_length=1, z_length=1) for xi, yi, zi in zip(x, y, z)]
# mesh_voxel = cubes[0].merge(cubes[1:]) if cubes else None

# plotter2 = pv.Plotter(title="2")
# if mesh_voxel:
#     plotter2.add_mesh(mesh_voxel, color="lightgray", opacity=0.3)
# plotter2.show()

# Skeleton using skeletonize
skeleton_sk = skeletonize(voxels > 0)
sx, sy, sz = np.where(skeleton_sk)
skeleton_sk_points = np.vstack([sx, sy, sz]).T

plotter3 = pv.Plotter(title="3")
if skeleton_sk_points.size > 0:
    skel_cloud_sk = pv.PolyData(skeleton_sk_points)
    plotter3.add_points(skel_cloud_sk, color="blue", point_size=5, render_points_as_spheres=True)
plotter3.show()
