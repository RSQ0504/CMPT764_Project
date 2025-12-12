import numpy as np
import pyvista as pv
import torch
import os
import trimesh

# 引用你的模型文件
from Ours_model.model_revise_with_keypoints_extra_loss import NET_Wrapper
# import numpy as np
# import pyvista as pv
def evaluate_gt_sdf_vs_voxel_iou(bae_net, shape_idx=0):
    pts = bae_net.data_points[shape_idx]          # [N,3]  SDF 采样点
    occ_sdf = bae_net.data_occupancy[shape_idx]   # [N,]   SDF-based occupancy (0/1)

    vox = bae_net.data_voxels[shape_idx]          # [64,64,64]
    dim = vox.shape[0]

    # [-1,1] -> [0, dim-1] 映射到 voxel 索引
    coords = (pts + 1.0) / 2.0
    idx = np.clip((coords * (dim - 1)).astype(np.int32), 0, dim - 1)
    x, y, z = idx[:, 0], idx[:, 1], idx[:, 2]

    occ_vox = vox[x, y, z].astype(np.int32)       # 在同一批 SDF 点上，用 voxel 定义采 GT

    gt_sdf = occ_sdf.astype(np.int32)
    gt_vox = occ_vox

    inter = np.logical_and(gt_sdf == 1, gt_vox == 1).sum()
    union = np.logical_or(gt_sdf == 1, gt_vox == 1).sum()
    iou = inter / (union + 1e-8)

    print(f"[GT SDF vs GT Voxel] IoU={iou:.4f}, "
          f"pos_sdf={gt_sdf.sum()}, pos_vox={gt_vox.sum()}, N={len(pts)}")
    return iou


def triangles_to_pv_faces(triangles: np.ndarray) -> np.ndarray:

    triangles = np.asarray(triangles, dtype=np.int64)
    n_faces = triangles.shape[0]

    faces = np.hstack([
        np.full((n_faces, 1), 3, dtype=np.int64),
        triangles
    ]).ravel()
    return faces

class Config:
    def __init__(self):
        self.learning_rate = 1e-3
        self.beta1 = 0.9
        self.epoch = 200000



net = NET_Wrapper(data_dir='data/train_with_skeleton/rod', gf_split=2)
config = Config()
net.load_checkpoint("checkpoint/ours/rod_2p/ckpt_epoch_500000.pth")



idx = 0

test_voxels = net.data_voxels[idx:idx + 1]
test_points = net.data_points[idx]

raw_junc = net.data_junctions[idx]
raw_end  = net.data_endpoints[idx]
mask_j = np.abs(raw_junc[:, 0] - 10.0) > 1e-4
mask_e = np.abs(raw_end[:, 0]  - 10.0) > 1e-4

fixed_junc = raw_junc[mask_j]
fixed_end  = raw_end[mask_e]
print(f"\n[Data Loaded from Wrapper]")
print(f"Voxels: {test_voxels.shape}")
print(f"Points: {test_points.shape}")
print(f"Valid Junctions: {len(fixed_junc)}")
print(f"Valid Endpoints: {len(fixed_end)}")


print("\n[Visualizing Loaded Data]...")

plotter = pv.Plotter(title="Final Data Verification")

test_occ = net.data_occupancy[idx]
inside_points = test_points[test_occ > 0.5]
if len(inside_points) == 0: inside_points = test_points # Fallback

plotter.add_mesh(pv.PolyData(inside_points), color="red", point_size=2, opacity=0.6, label="Object")

if len(fixed_junc) > 0:
    plotter.add_mesh(pv.PolyData(fixed_junc), color="blue", point_size=15, render_points_as_spheres=True, label="Junctions")
if len(fixed_end) > 0:
    plotter.add_mesh(pv.PolyData(fixed_end), color="green", point_size=15, render_points_as_spheres=True, label="Endpoints")

plotter.add_legend()
plotter.show()


print("\n[Running Inference]...")

predictions = net.test_segmentation(
    test_points,
    test_voxels,
    junction_points=fixed_junc,
    endpoint_points=fixed_end
)
print(f"Predictions Shape: {predictions.shape}")
np.save('segmentation.npy', predictions)


print("\n[Generating Meshes]...")
gt_vertices, gt_triangles = net.generate_gt_mesh_from_voxels(test_voxels, threshold=0.5)

vertices_list, triangles_list, total_v, total_f = net.generate_mesh(
    test_voxels,
    junction_points=fixed_junc,
    endpoint_points=fixed_end
)


if vertices_list:
    plotter = pv.Plotter(title="Segmentation Results", shape=(1, len(vertices_list) + 2))

    plotter.subplot(0, 0)
    plotter.add_text("Total Reconstruction", font_size=10)
    if total_v is not None and len(total_v) > 0:
        mesh = trimesh.Trimesh(vertices=total_v, faces=total_f)
        plotter.add_mesh(mesh, color='white', show_edges=False, opacity=1.0)
    else:
        plotter.add_text("Total Mesh Failed", color='red')
    plotter.subplot(0, 1)
    if gt_vertices is not None and gt_triangles is not None:
        gt_faces = triangles_to_pv_faces(gt_triangles)
        mesh_gt = pv.PolyData(gt_vertices, gt_faces)
        plotter.add_text("GT (voxels)", font_size=14)
        plotter.add_mesh(mesh_gt, color="white")
    else:
        plotter.add_text("GT Mesh Failed", color="red", font_size=14)
    colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
    for i, (vertices, triangles) in enumerate(zip(vertices_list, triangles_list)):
        plotter.subplot(0, i + 2)
        plotter.add_text(f"Branch {i}", font_size=10)

        if vertices is not None and len(vertices) > 0:
            mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
            mesh.export(f'mesh_branch_{i}.ply')
            print(f"  - Branch {i}: {len(vertices)} verts")
            plotter.add_mesh(mesh, color=colors[i % len(colors)], show_edges=False)
        else:
            plotter.add_text("Empty", color='grey')
    print("\n[Evaluating IoU on voxel grid]...")
    _ = net.evaluate_shape_iou(shape_idx=idx, threshold=0.5)

    print("\n[Evaluating IoU in SDF space]...")
    _ = net.evaluate_shape_iou_sdf(shape_idx=idx, threshold=0.5)

    print("\n[Evaluating *GT* SDF vs *GT* Voxel IoU]...")
    evaluate_gt_sdf_vs_voxel_iou(net, shape_idx=idx)
    plotter.show()
