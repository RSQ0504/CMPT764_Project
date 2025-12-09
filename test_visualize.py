import numpy as np
import pyvista as pv
import torch
import os
import trimesh

# 引用你的模型文件
from model_revise_with_keypoints import BAE_NET_Wrapper
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

# def evaluate_shape_iou_sdf(bae_net, shape_idx=0, threshold=0.5, batch_size=8192):
#     """
#     使用 SDF-based occupancy (data_points + data_occupancy) 计算 IoU：
#       - GT: bae_net.data_points[shape_idx], bae_net.data_occupancy[shape_idx]
#       - Pred: 在同一批点上，用当前模型 + 骨骼预测 occupancy
#     """
#     device = bae_net.device
#     bae_net.model.eval()
#
#     with torch.no_grad():
#         # ---------- 1. 取 SDF 点和 GT occupancy ----------
#         pts = bae_net.data_points[shape_idx]          # [N, 3]
#         occ = bae_net.data_occupancy[shape_idx]       # [N,]
#         gt_bin = (occ > 0.5).astype(np.int32)        # 保险起见再二值化一下
#
#         N = pts.shape[0]
#
#         # ---------- 2. 编码 z (和训练完全一致) ----------
#         vox = bae_net.data_voxels[shape_idx]         # [64,64,64]
#         t_vox = torch.from_numpy(vox).float().unsqueeze(0).unsqueeze(0).to(device)
#         z = bae_net.model.encoder(t_vox)             # [1, z_dim]
#
#         # ---------- 3. 取骨骼 (用 raw 的，和训练一致，有 padding=10.0) ----------
#         raw_junc = bae_net.data_junctions[shape_idx]    # [J_pad, 3]
#         raw_end  = bae_net.data_endpoints[shape_idx]    # [E_pad, 3]
#
#         def to_tensor(arr):
#             if arr is None or arr.size == 0:
#                 return None
#             return torch.from_numpy(arr).float().unsqueeze(0).to(device)
#
#         j_tensor = to_tensor(raw_junc)
#         e_tensor = to_tensor(raw_end)
#
#         def get_dummy():
#             return torch.ones(1, 1, 3).to(device) * 10.0
#
#         if j_tensor is not None and e_tensor is None:
#             e_tensor = get_dummy()
#         if e_tensor is not None and j_tensor is None:
#             j_tensor = get_dummy()
#
#         # ---------- 4. 在 SDF 点上预测 occupancy ----------
#         preds = []
#         for i in range(0, N, batch_size):
#             batch_pts_np = pts[i:i + batch_size]                  # [B,3]
#             batch_pts = torch.from_numpy(batch_pts_np).float().unsqueeze(0).to(device)  # [1,B,3]
#
#             # 直接用 generator (和训练 forward 一致)
#             # generator 返回: h3(branch_pred), h3_max(pred_occupancy)
#             _, pred = bae_net.model.generator(
#                 batch_pts,
#                 z,
#                 junction_points=j_tensor,
#                 endpoint_points=e_tensor
#             )  # pred: [1, B, 1]
#
#             preds.append(pred.squeeze(0).cpu().numpy())           # [B,1] -> [B]
#
#         pred_full = np.concatenate(preds, axis=0)                 # [N,]
#         pred_full = pred_full.reshape(-1)
#
#         # ---------- 5. 在 SDF 空间上算 IoU + MSE ----------
#         pred_bin = (pred_full >= threshold).astype(np.int32)
#
#         inter = np.logical_and(gt_bin == 1, pred_bin == 1).sum()
#         union = np.logical_or(gt_bin == 1, pred_bin == 1).sum()
#         iou = inter / (union + 1e-8)
#
#         mse = ((pred_full - gt_bin.astype(np.float32)) ** 2).mean()
#
#         print(f"[SDF IoU] shape {shape_idx}: IoU={iou:.4f}, "
#               f"MSE={mse:.6f}, "
#               f"pred_pos={pred_bin.sum()}, gt_pos={gt_bin.sum()}, N={N}")
#         return iou, mse

# def evaluate_shape_iou(bae_net, shape_idx=0, threshold=0.5, batch_size=8192):
#     """
#     对指定 shape 计算 IoU：
#       - GT: 直接用 voxels 的 0/1
#       - Pred: 在同样的 64^3 网格上，用当前模型 + 骨骼 预测 occupancy，再和 GT 做 IoU
#     """
#     device = bae_net.device
#     bae_net.model.eval()
#
#     with torch.no_grad():
#         # ---------- 1. 取 GT 体素 ----------
#         vox = bae_net.data_voxels[shape_idx]        # [64,64,64]
#         gt = (vox > 0.5).astype(np.int32)
#
#         # ---------- 2. 得到 latent code z ----------
#         t_vox = torch.from_numpy(vox).float().unsqueeze(0).unsqueeze(0).to(device)
#         z = bae_net.model.encoder(t_vox)           # [1, z_dim]
#
#         # ---------- 3. 取骨骼并做和脚本中一致的过滤 ----------
#         raw_junc = bae_net.data_junctions[shape_idx]
#         raw_end  = bae_net.data_endpoints[shape_idx]
#
#         j_valid = raw_junc[raw_junc[:, 0] < 5.0]   # 和你上面 fixed_junc 一致
#         e_valid = raw_end[raw_end[:, 0] < 5.0]
#
#         def to_tensor(arr):
#             if arr is None or len(arr) == 0:
#                 return None
#             return torch.from_numpy(arr).float().unsqueeze(0).to(device)
#
#         j_tensor = to_tensor(j_valid)
#         e_tensor = to_tensor(e_valid)
#
#         # 和 test_segmentation / generate_mesh 保持同样 dummy 逻辑
#         def get_dummy():
#             return torch.ones(1, 1, 3).to(device) * 10.0
#
#         if j_tensor is not None and e_tensor is None:
#             e_tensor = get_dummy()
#         if e_tensor is not None and j_tensor is None:
#             j_tensor = get_dummy()
#
#         # ---------- 4. 在 [-1,1]^3 网格上预测 ----------
#         dim = 64
#         coords = np.linspace(-1.0, 1.0, dim)
#         gx, gy, gz = np.meshgrid(coords, coords, coords, indexing='ij')
#         grid_points = np.stack([gx.flatten(), gy.flatten(), gz.flatten()], axis=1)
#
#         preds = []
#         for i in range(0, len(grid_points), batch_size):
#             batch_pts = grid_points[i:i + batch_size]
#             batch_tensor = torch.from_numpy(batch_pts).float().unsqueeze(0).to(device)
#
#             # 直接用 generator（和 generate_mesh 一致）
#             _, pred = bae_net.model.generator(
#                 batch_tensor,
#                 z,
#                 junction_points=j_tensor,
#                 endpoint_points=e_tensor
#             )  # pred: [1, B, 1]
#
#             preds.append(pred.squeeze(0).cpu().numpy())  # [B]
#
#         pred_grid = np.concatenate(preds, axis=0).reshape(dim, dim, dim)
#
#         # ---------- 5. 二值化并计算 IoU ----------
#         pred_occ = (pred_grid >= threshold).astype(np.int32)
#
#         inter = np.logical_and(gt == 1, pred_occ == 1).sum()
#         union = np.logical_or(gt == 1, pred_occ == 1).sum()
#         iou = inter / (union + 1e-8)
#
#         print(f"[IoU] shape {shape_idx}: IoU={iou:.4f}, "
#               f"pred_vol={pred_occ.sum()}, gt_vol={gt.sum()}")
#         return iou

def triangles_to_pv_faces(triangles: np.ndarray) -> np.ndarray:
    """
    triangles: (N, 3) int 或 float
    返回: (N*4,) 的一维 faces array，格式 [3, i, j, k, 3, i2, j2, k2, ...]
    """
    triangles = np.asarray(triangles, dtype=np.int64)
    n_faces = triangles.shape[0]
    # 每一行前面加一个 3，然后 flatten
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

# ==========================================
# 1. 加载模型和数据
# ==========================================
print("正在初始化 BAE_NET_Wrapper...")
# 这里会自动调用你修改过的 _load_data (包含 Swap X-Y 和 Force Scale)
bae_net = BAE_NET_Wrapper(data_dir='./train_with_skeleton_new/dog')
config = Config()

checkpoint_path = "checkpoint/model_skeleton/ckpt_epoch_150000.pth"
if os.path.exists(checkpoint_path):
    bae_net.load_checkpoint(checkpoint_path)
else:
    print(f"[Error] Checkpoint not found: {checkpoint_path}")

# ==========================================
# 2. 获取数据 (Trust the Loader)
# ==========================================
idx = 0
# 这些数据已经被 _load_data 处理过了(缩放+换轴)，我们直接用！
test_voxels = bae_net.data_voxels[idx:idx + 1]
test_points = bae_net.data_points[idx]
# raw_junc = bae_net.data_junctions[idx]
# raw_end = bae_net.data_endpoints[idx]
#
# # 唯一需要做的是：过滤掉 Padding (那些等于 10.0 的点)
# # 因为 _load_data 为了对其 Batch 做了 Padding
# fixed_junc = raw_junc[raw_junc[:, 0] < 5.0]
# fixed_end = raw_end[raw_end[:, 0] < 5.0]
raw_junc = bae_net.data_junctions[idx]
raw_end  = bae_net.data_endpoints[idx]

mask_j = np.abs(raw_junc[:, 0] - 10.0) > 1e-4
mask_e = np.abs(raw_end[:, 0]  - 10.0) > 1e-4

fixed_junc = raw_junc[mask_j]
fixed_end  = raw_end[mask_e]
print(f"\n[Data Loaded from Wrapper]")
print(f"Voxels: {test_voxels.shape}")
print(f"Points: {test_points.shape}")
print(f"Valid Junctions: {len(fixed_junc)}")
print(f"Valid Endpoints: {len(fixed_end)}")

# ==========================================
# 3. 可视化验证 (Verification)
# ==========================================
print("\n[Visualizing Loaded Data]...")
print("如果 _load_data 是对的，这里的 蓝/绿点 应该完美贴合在 红点 上。")

plotter = pv.Plotter(title="Final Data Verification")

# 1. 过滤出物体内部点 (为了看清楚形状)
test_occ = bae_net.data_occupancy[idx]
inside_points = test_points[test_occ > 0.5]
if len(inside_points) == 0: inside_points = test_points # Fallback

# 2. 画物体
plotter.add_mesh(pv.PolyData(inside_points), color="red", point_size=2, opacity=0.6, label="Object")

# 3. 画骨骼 (直接用从 loader 拿出来的点)
if len(fixed_junc) > 0:
    plotter.add_mesh(pv.PolyData(fixed_junc), color="blue", point_size=15, render_points_as_spheres=True, label="Junctions")
if len(fixed_end) > 0:
    plotter.add_mesh(pv.PolyData(fixed_end), color="green", point_size=15, render_points_as_spheres=True, label="Endpoints")

plotter.add_legend()
plotter.show()

# ==========================================
# 4. 运行预测 (Inference)
# ==========================================
print("\n[Running Inference]...")

# 直接传入过滤后的有效点
predictions = bae_net.test_segmentation(
    test_points,
    test_voxels,
    junction_points=fixed_junc,
    endpoint_points=fixed_end
)
print(f"Predictions Shape: {predictions.shape}")
np.save('segmentation.npy', predictions)

# ==========================================
# 5. 生成 Mesh (Generate Mesh)
# ==========================================
print("\n[Generating Meshes]...")
gt_vertices, gt_triangles = bae_net.generate_gt_mesh_from_voxels(test_voxels, threshold=0.5)

vertices_list, triangles_list, total_v, total_f = bae_net.generate_mesh(
    test_voxels,
    junction_points=fixed_junc,
    endpoint_points=fixed_end
)

# ==========================================
# 6. 可视化结果
# ==========================================
if vertices_list:
    plotter = pv.Plotter(title="Segmentation Results", shape=(1, len(vertices_list) + 2))

    # --- 视口 0: 完整重建 ---
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
    # --- 视口 1~N: 部件 ---
    colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
    for i, (vertices, triangles) in enumerate(zip(vertices_list, triangles_list)):
        plotter.subplot(0, i + 1 + 1)
        plotter.add_text(f"Branch {i}", font_size=10)

        if vertices is not None and len(vertices) > 0:
            mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
            mesh.export(f'mesh_branch_{i}.ply')
            print(f"  - Branch {i}: {len(vertices)} verts")
            plotter.add_mesh(mesh, color=colors[i % len(colors)], show_edges=False)
        else:
            plotter.add_text("Empty", color='grey')
    # print("\n[Evaluating IoU]...")
    print("\n[Evaluating IoU on voxel grid]...")
    _ = bae_net.evaluate_shape_iou(shape_idx=idx, threshold=0.5)

    print("\n[Evaluating IoU in SDF space]...")
    _ = bae_net.evaluate_shape_iou_sdf(shape_idx=idx, threshold=0.5)

    print("\n[Evaluating *GT* SDF vs *GT* Voxel IoU]...")
    evaluate_gt_sdf_vs_voxel_iou(bae_net, shape_idx=idx)
    plotter.show()
else:
    print("❌ No meshes generated.")