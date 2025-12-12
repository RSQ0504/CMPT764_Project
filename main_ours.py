import numpy as np
import pyvista as pv
from skimage import measure
import torch
import os
import trimesh

from Ours_model.model_revise_with_keypoints_extra_loss import Encoder3D, NET_Wrapper

class Config:
    def __init__(self):
        self.learning_rate = 1e-3 
        self.beta1 = 0.9   
        self.epoch = 500001

# npz_path = "./reference_models_processed/hand/voxel_and_sdf.npz"
# data = np.load(npz_path)
# voxels = data['voxels']

# voxels = voxels.astype(np.float32)
# x = torch.from_numpy(voxels).unsqueeze(0).unsqueeze(0)

# encoder = Encoder3D(z_dim=128)
# z = encoder(x)
# # print(z.shape)

# bae_net = BAE_NET_Wrapper(data_dir='./train_with_skeleton_new/sofa')
net = NET_Wrapper(data_dir='data/train_with_skeleton_test_data/chair',checkpoint_dir='checkpoint/ours/chair_5p', gf_split=5)
config = Config()
net._train_unsupervised(config)

# test_voxels = net.data_voxels[1:2]
# test_points = net.data_points[1]
# # print(test_voxels.shape, test_points.shape)
# predictions = net.test_segmentation(test_points, test_voxels)
# np.save(os.path.join('segmentation.npy'), predictions)
#
# test_voxels = net.data_voxels[:1]
#
# vertices_list, triangles_list = net.generate_mesh(test_voxels)
#
# if vertices_list:
#     for i, (vertices, triangles) in enumerate(zip(vertices_list, triangles_list)):
#         mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
#         mesh.export(f'mesh_branch_{i}.ply')
#         print(f"Mesh branch {i} saved with {len(vertices)} vertices, {len(triangles)} faces")