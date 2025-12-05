import numpy as np
import pyvista as pv

# 读取
points = np.load("segmentation.npy")  # shape (100000,4)
xyz = points[:, :3]   # x,y,z
labels = points[:, 3] # 标签或值
# print(points[0:100])

# 可视化
cloud = pv.PolyData(xyz)
# 可以根据 label 上色
plotter = pv.Plotter()
plotter.add_mesh(cloud, scalars=labels, render_points_as_spheres=True, point_size=3)
plotter.show()