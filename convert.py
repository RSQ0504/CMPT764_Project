import numpy as np
import pyvista as pv
from skimage import measure
import h5py


npz_path = "voxel_and_sdf.npz"
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

mesh_smooth.save("mesh_smooth.obj")
print("Smooth mesh saved as mesh_smooth.obj")


x, y, z = np.where(voxels)
plotter = pv.Plotter(off_screen=True)
cubes = []

for xi, yi, zi in zip(x, y, z):
    cube = pv.Cube(center=(xi, yi, zi), x_length=1, y_length=1, z_length=1)
    cubes.append(cube)


if cubes:
    mesh_voxel = cubes[0].merge(cubes[1:])
    mesh_voxel.save("mesh_voxel.obj")
    print("Voxel cube mesh saved as mesh_voxel.obj")
else:
    print("No voxels to save as cubes.")

# points = sdf_points / scale + centroid
points = sdf_points.copy()

point_cloud = pv.PolyData(points)
point_cloud["SDF"] = sdf_values

plotter = pv.Plotter()
plotter.add_points(point_cloud, scalars="SDF", render_points_as_spheres=True, point_size=3)

# plotter.add_mesh(pv.Sphere(center=centroid, radius=1.0), color="red")

plotter.show()