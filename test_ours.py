import numpy as np
import pyvista as pv
from skimage import measure
import torch
import os
import trimesh
import pyvista as pv


from Ours_model.model_revise_with_keypoints_extra_loss import Encoder3D, BAE_NET_Wrapper

class Config:
    def __init__(self):
        self.learning_rate = 1e-3
        self.beta1 = 0.9
        self.epoch = 200000


bae_net = BAE_NET_Wrapper(data_dir='./data/train_with_skeleton/hand')
config = Config()
bae_net.load_checkpoint("checkpoint/model_revised_keypoint/checkpoint_epoch_180000.pth")

test_voxels = bae_net.data_voxels[0:1]
test_points = bae_net.data_points[0]
test_junctions = bae_net.data_junctions
test_endpoints = bae_net.data_endpoints
# print(test_voxels.shape, test_points.shape)
predictions = bae_net.test_segmentation(test_points, test_voxels, test_junctions, test_endpoints)
print(predictions.shape)
# np.save(os.path.join('segmentation.npy'), predictions)

test_voxels = bae_net.data_voxels[:1]
print(test_voxels.shape)

vertices_list, triangles_list, all_vertices, all_triangles = bae_net.generate_mesh(test_voxels, test_junctions, test_endpoints)

plotter = pv.Plotter()

if vertices_list:
    mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_triangles)
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, show_edges=True)
    plotter.show()
    plotter.close()
    for i, (vertices, triangles) in enumerate(zip(vertices_list, triangles_list)):
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        # mesh.export(f'mesh_branch_{i}.ply')
        print(f"Mesh branch {i} saved with {len(vertices)} vertices, {len(triangles)} faces")
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, show_edges=True)
        plotter.show()
        plotter.close()
