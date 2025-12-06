import os
import subprocess
import trimesh
import open3d as o3d
import os
import numpy as np
import json
import branch_utils as branch_utils
gem = []
folder = "./reference_models_processed/pot"
for file in os.listdir(folder):
    if file.startswith("branch_") and file.endswith(".ply"):
        pcd = o3d.io.read_point_cloud(os.path.join(folder, file))
        color = np.random.rand(3)
        pcd.paint_uniform_color(color)
        gem.append(pcd)
o3d.visualization.draw_geometries(gem)

pcd = o3d.io.read_point_cloud(os.path.join(folder, "skeletal_prior.ply"))
o3d.visualization.draw_geometries([pcd])