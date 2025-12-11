import numpy as np
import pyvista as pv
from skimage import measure
import torch
import os
import trimesh
import pyvista as pv
import matplotlib.pyplot as plt




from model_revise import Encoder3D, BAE_NET_Wrapper

from .utils import process_segmentation_meshes
from .utils import visualize_meshes_and_cuboid_subplots, evaluate_union_multiple_meshes

class Config:
    def __init__(self):
        self.learning_rate = 1e-3
        self.beta1 = 0.9
        self.epoch = 200000


bae_net = BAE_NET_Wrapper(data_dir='data/train_with_skeleton/hand')
config = Config()
bae_net.load_checkpoint("checkpoint/model_revised/hand256-4/checkpoint_epoch_190000.pth")
idx = 0
test_voxels = bae_net.data_voxels[idx:idx + 1]
test_points = bae_net.data_points[idx]


# raw_junc = bae_net.data_junctions[idx]
# raw_end  = bae_net.data_endpoints[idx]

# mask_j = np.abs(raw_junc[:, 0] - 10.0) > 1e-4
# mask_e = np.abs(raw_end[:, 0]  - 10.0) > 1e-4

# fixed_junc = raw_junc[mask_j]
# fixed_end  = raw_end[mask_e]


# predictions = bae_net.test_segmentation(
#     test_points,
#     test_voxels,
#     junction_points=fixed_junc,
#     endpoint_points=fixed_end
# )
# predictions = torch.from_numpy(predictions)
# predictions, _ = torch.max(predictions, dim=1, keepdim=True)
# print(predictions)
# predictions[predictions < 0.5] = -1
# print(f"Predictions Shape: {predictions.shape}")
# print(test_points.shape)
# test_points = np.concatenate([test_points, predictions], axis=1)
# coords = test_points[:, :3]
# seg_id = test_points[:, 3]
# mask = seg_id != -1
# test_points = test_points[mask]
# print(test_points.shape) 
# np.save('segmentation.npy', test_points)

# unique_labels = np.unique(seg_id[mask])

# plotter = pv.Plotter(title="Final Data Verification")

colors = ["red", "blue", "green", "yellow", "purple", "cyan", "orange"]


vertices_list, triangles_list, _, _ = bae_net.generate_mesh(
        test_voxels
    )
# plotter = pv.Plotter(title="Final Data Verification")

# for i, (v, t) in enumerate(zip(vertices_list, triangles_list)):
#     if v is not None and len(v) > 0:
#         mesh = trimesh.Trimesh(vertices=v, faces=t)


#         plotter.add_mesh(mesh, color=colors[i % len(colors)], show_edges=False)
# plotter.show()
# plotter.close()

meshes, dense_points, cuboids = process_segmentation_meshes(vertices_list, triangles_list)
flat_cuboids = [c for cub_list in cuboids if cub_list is not None for c in cub_list]


visualize_meshes_and_cuboid_subplots(meshes, flat_cuboids)

print(evaluate_union_multiple_meshes(meshes, flat_cuboids))