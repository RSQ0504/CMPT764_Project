import numpy as np
import pyvista as pv
from skimage import measure
import torch
import os
import trimesh
import pyvista as pv


from model import Encoder3D, BAE_NET_Wrapper

class Config:
    def __init__(self):
        self.learning_rate = 1e-3
        self.beta1 = 0.9
        self.epoch = 200000


bae_net = BAE_NET_Wrapper(data_dir='./reference_models_processed/hand')
config = Config()
bae_net.load_checkpoint("checkpoint/checkpoint_epoch_190000.pth")

test_voxels = bae_net.data_voxels[0:1]
test_points = bae_net.data_points[0]
# print(test_voxels.shape, test_points.shape)
predictions = bae_net.test_segmentation(test_points, test_voxels)
print(predictions.shape)
np.save(os.path.join('segmentation.npy'), predictions)

test_voxels = bae_net.data_voxels[:1]
print(test_voxels.shape)

vertices_list, triangles_list = bae_net.generate_mesh(test_voxels)

plotter = pv.Plotter()

if vertices_list:
    for i, (vertices, triangles) in enumerate(zip(vertices_list, triangles_list)):
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        mesh.export(f'mesh_branch_{i}.ply')
        print(f"Mesh branch {i} saved with {len(vertices)} vertices, {len(triangles)} faces")
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, show_edges=True)
        plotter.show()
        plotter.close()
