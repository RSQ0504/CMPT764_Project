import numpy as np
import pyvista as pv
from skimage import measure
import torch
import os
import trimesh
import pyvista as pv


from baseline_model.model_revise import Encoder3D, BAE_NET_Wrapper

class Config:
    def __init__(self):
        self.learning_rate = 1e-3
        self.beta1 = 0.9
        self.epoch = 200000


bae_net = BAE_NET_Wrapper(data_dir='data/reference_models_processed/pot')
config = Config()
bae_net.load_checkpoint("checkpoint/model_revised/pot256-4/checkpoint_epoch_200000.pth")
idx = 0
test_voxels = bae_net.data_voxels[idx:idx+1]
test_points = bae_net.data_points[idx]
# print(test_voxels.shape, test_points.shape)
predictions = bae_net.test_segmentation(test_points, test_voxels)
print(predictions.shape)
# np.save(os.path.join('segmentation.npy'), predictions)

test_voxels = bae_net.data_voxels[:1]
print(test_voxels.shape)

vertices_list, triangles_list,all_vertices,all_triangles = bae_net.generate_mesh(test_voxels)

plotter = pv.Plotter()

plotter = pv.Plotter()

if vertices_list:
    plotter = pv.Plotter(title="Segmentation Results", shape=(1, len(vertices_list) + 2))

    # --- 视口 0: 完整重建 ---
    plotter.subplot(0, 0)
    plotter.add_text("Total Reconstruction", font_size=10)
    if all_vertices is not None and len(all_vertices) > 0:
        mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_triangles)
        plotter.add_mesh(mesh, color='white', show_edges=False, opacity=1.0)
    else:
        plotter.add_text("Total Mesh Failed", color='red')
    plotter.subplot(0, 1)
    # --- 视口 1~N: 部件 ---
    colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
    for i, (vertices, triangles) in enumerate(zip(vertices_list, triangles_list)):
        plotter.subplot(0, i + 1 + 1)
        plotter.add_text(f"Branch {i}", font_size=10)

        if vertices is not None and len(vertices) > 0:
            mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
            # mesh.export(f'mesh_branch_{i}.ply')
            print(f"  - Branch {i}: {len(vertices)} verts")
            plotter.add_mesh(mesh, color=colors[i % len(colors)], show_edges=False)
        else:
            plotter.add_text("Empty", color='grey')

    plotter.show()
else:
    print("❌ No meshes generated.")