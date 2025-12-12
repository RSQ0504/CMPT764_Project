import numpy as np
import pyvista as pv
from skimage import measure
import torch
import os
import trimesh
import pyvista as pv


from Ours_model.model_revise_with_keypoints_extra_loss import Encoder3D, NET_Wrapper

class Config:
    def __init__(self):
        self.learning_rate = 1e-3
        self.beta1 = 0.9
        self.epoch = 200000


net = NET_Wrapper(data_dir='data/train_with_skeleton_test_data/chair', gf_split=6)
config = Config()
net.load_checkpoint("checkpoint/ours/chair_6p/ckpt_epoch_110000.pth")

idx = 0
test_voxels = net.data_voxels[idx:idx + 1]
test_points = net.data_points[idx]


raw_junc = net.data_junctions[idx]
raw_end  = net.data_endpoints[idx]

mask_j = np.abs(raw_junc[:, 0] - 10.0) > 1e-4
mask_e = np.abs(raw_end[:, 0]  - 10.0) > 1e-4

fixed_junc = raw_junc[mask_j]
fixed_end  = raw_end[mask_e]

# print(test_voxels.shape, test_points.shape)
predictions = net.test_segmentation(test_points, test_voxels, junction_points = fixed_junc, endpoint_points = fixed_end)
print(predictions.shape)
# np.save(os.path.join('segmentation.npy'), predictions)

test_voxels = net.data_voxels[:1]
print(test_voxels.shape)

vertices_list, triangles_list, all_vertices, all_triangles = net.generate_mesh(test_voxels, fixed_junc, fixed_end)

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
