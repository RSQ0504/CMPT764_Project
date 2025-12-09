import numpy as np
import pyvista as pv
from skimage import measure
import torch
import os
import trimesh

from model_revise_with_keypoints import Encoder3D, BAE_NET_Wrapper

class Config:
    def __init__(self):
        self.learning_rate = 1e-3 
        self.beta1 = 0.9   
        self.epoch = 200001

# npz_path = "./reference_models_processed/hand/voxel_and_sdf.npz"
# data = np.load(npz_path)
# voxels = data['voxels']

# voxels = voxels.astype(np.float32)
# x = torch.from_numpy(voxels).unsqueeze(0).unsqueeze(0)

# encoder = Encoder3D(z_dim=128)
# z = encoder(x)
# # print(z.shape)

# bae_net = BAE_NET_Wrapper(data_dir='./train_with_skeleton_new/sofa')
bae_net = BAE_NET_Wrapper(data_dir='./train_with_skeleton_new/dog')
config = Config()
bae_net._train_unsupervised(config)

# test_voxels = bae_net.data_voxels[1:2]
# test_points = bae_net.data_points[1]
# # print(test_voxels.shape, test_points.shape)
# predictions = bae_net.test_segmentation(test_points, test_voxels)
# np.save(os.path.join('segmentation.npy'), predictions)
#
# test_voxels = bae_net.data_voxels[:1]
#
# vertices_list, triangles_list = bae_net.generate_mesh(test_voxels)
#
# if vertices_list:
#     for i, (vertices, triangles) in enumerate(zip(vertices_list, triangles_list)):
#         mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
#         mesh.export(f'mesh_branch_{i}.ply')
#         print(f"Mesh branch {i} saved with {len(vertices)} vertices, {len(triangles)} faces")