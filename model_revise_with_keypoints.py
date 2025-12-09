import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

USE_SKELETON = True  # 默认 True，做实验 A 的时候改成 False

# ==========================================
# Helper Function: Weight Initialization
# ==========================================
def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        # Linear layers keep consistent with TF: use Normal(0, 0.02)
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
        # FIX: The original TF implementation uses Xavier initialization for Conv3d
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('InstanceNorm') != -1:
        if m.weight is not None:
            nn.init.constant_(m.weight.data, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


# ==========================================
# Encoder (Standard 3D CNN)
# ==========================================
class Encoder3D(nn.Module):
    def __init__(self, z_dim=128, ef_dim=32):
        super(Encoder3D, self).__init__()
        self.z_dim = z_dim
        self.ef_dim = ef_dim

        self.conv1 = nn.Conv3d(1, ef_dim, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.InstanceNorm3d(ef_dim, affine=True)

        self.conv2 = nn.Conv3d(ef_dim, ef_dim * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.InstanceNorm3d(ef_dim * 2, affine=True)

        self.conv3 = nn.Conv3d(ef_dim * 2, ef_dim * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.InstanceNorm3d(ef_dim * 4, affine=True)

        self.conv4 = nn.Conv3d(ef_dim * 4, ef_dim * 8, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.InstanceNorm3d(ef_dim * 8, affine=True)

        self.conv5 = nn.Conv3d(ef_dim * 8, z_dim, kernel_size=4, stride=1, padding=0)

    def forward(self, x):
        # x: [batch, 1, 64, 64, 64]
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.02)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.02)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.02)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.02)
        x = torch.sigmoid(self.conv5(x))
        return x.view(-1, self.z_dim)


# ==========================================
# Generator (Corrected: Inject Skeleton Features Early)
# ==========================================
class Generator(nn.Module):
    def __init__(self, z_dim=128, gf_dim=256, gf_split=4, L1reg=False):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.gf_split = gf_split
        self.L1reg = L1reg

        # Layer 1: Input Point + Z Code -> Hidden 1
        # Dim: z_dim + 3 -> gf_dim * 4
        self.fc1 = nn.Linear(z_dim + 3, gf_dim * 4)

        # Layer 2: Hidden 1 + Skeleton Distance -> Hidden 2
        # Dim: (gf_dim * 4) + 2 (Distance Features) -> gf_dim
        # We inject the skeleton distance here to let the network process it
        self.fc2 = nn.Linear(gf_dim * 4 + 2, gf_dim)

        # Layer 3: Hidden 2 -> Output Logits
        # Dim: gf_dim -> gf_split
        self.fc3 = nn.Linear(gf_dim, gf_split)

    def compute_min_distance(self, query_points, keypoints):
        """
        Compute Euclidean distance from query_points to the nearest keypoints.
        query_points: [B, N, 3]
        keypoints:    [B, K, 3]
        Return:       [B, N, 1]
        """
        if keypoints.size(1) == 0:
            return torch.ones(query_points.shape[0], query_points.shape[1], 1).to(query_points.device)

        dists = torch.cdist(query_points, keypoints, p=2)
        min_dists, _ = torch.min(dists, dim=2, keepdim=True)  # [B, N, 1]
        return min_dists

    def forward(self, points, z, junction_points=None, endpoint_points=None):
        B, N, _ = points.shape

        # 1. expand z
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if z.shape[0] == 1 and B > 1:
            z = z.expand(B, -1)
        z_expanded = z.unsqueeze(1).expand(-1, N, -1)  # [B,N,z_dim]

        # 2. 第一层
        pointz = torch.cat([points, z_expanded], dim=2)  # [B,N,z_dim+3]
        h1 = F.leaky_relu(self.fc1(pointz), 0.02)  # [B,N,4*gf_dim]

        # 3. 骨骼距离
        if USE_SKELETON and (junction_points is not None) and (endpoint_points is not None):
            # 正常用骨骼：真实距离特征
            d_junc = self.compute_min_distance(points, junction_points)  # [B,N,1]
            d_end = self.compute_min_distance(points, endpoint_points)  # [B,N,1]
            h1_aug = torch.cat([h1, d_junc, d_end], dim=2)  # [B,N,4*gf_dim+2]
        else:
            # 不用骨骼：用常数填充，所有点/shape 完全一样 → 不提供任何有用信息
            dummy = torch.ones(B, N, 1, device=points.device) * 10.0
            h1_aug = torch.cat([h1, dummy, dummy], dim=2)

        # 4. 第二层
        h2 = F.leaky_relu(self.fc2(h1_aug), 0.02)  # [B,N,gf_dim]

        # 5. 输出 logits（每个分支一个 logit）
        logits = self.fc3(h2)  # [B,N,K]
        return logits
    # def forward(self, points, z, junction_points=None, endpoint_points=None):
    #     # points: [B, N, 3]
    #     # z: [B, z_dim]
    #
    #     batch_size = points.shape[0]
    #     num_points = points.shape[1]
    #
    #     # 1. Expand Z
    #     if z.dim() == 1: z = z.unsqueeze(0)
    #     if z.shape[0] == 1 and batch_size > 1: z = z.expand(batch_size, -1)
    #     z_expanded = z.unsqueeze(1).expand(-1, num_points, -1)  # [B, N, z_dim]
    #
    #     # 2. Layer 1 Forward
    #     pointz = torch.cat([points, z_expanded], dim=2)  # [B, N, z_dim+3]
    #     h1 = F.leaky_relu(self.fc1(pointz), 0.02)  # [B, N, gf_dim*4]
    #
    #     # 3. Skeleton Distance Injection
    #     if junction_points is not None and endpoint_points is not None:
    #         d_junc = self.compute_min_distance(points, junction_points)  # [B, N, 1]
    #         d_end = self.compute_min_distance(points, endpoint_points)  # [B, N, 1]
    #
    #         # Concatenate to h1
    #         h1_aug = torch.cat([h1, d_junc, d_end], dim=2)  # [B, N, gf_dim*4 + 2]
    #     else:
    #         # Fallback: Create dummy distance (large value) to keep dimensions correct
    #         dummy_dist = torch.ones(batch_size, num_points, 1).to(points.device) * 10.0
    #         h1_aug = torch.cat([h1, dummy_dist, dummy_dist], dim=2)
    #
    #     # 4. Layer 2 Forward
    #     h2 = F.leaky_relu(self.fc2(h1_aug), 0.02)  # [B, N, gf_dim]
    #
    #     # 5. Layer 3 Forward (Output)
    #     logits = self.fc3(h2)
    #     logits_max, _ = torch.max(logits, dim=2, keepdim=True)
    #     h3 = torch.sigmoid(logits)
    #
    #     # h3 = torch.sigmoid(self.fc3(h2))  # [B, N, gf_split]
    #     #
    #     h3_max = torch.max(h3, dim=2, keepdim=True)[0]
    #
    #     return logits_max, h3_max


# ==========================================
# BAE_Net Model Wrapper
# ==========================================
class BAE_Net(nn.Module):
    def __init__(self, z_dim=128, ef_dim=32, gf_dim=256, gf_split=4, L1reg=False):
        super(BAE_Net, self).__init__()
        self.z_dim = z_dim
        self.ef_dim = ef_dim
        self.gf_dim = gf_dim
        self.gf_split = gf_split
        self.L1reg = L1reg

        self.encoder = Encoder3D(z_dim=z_dim, ef_dim=ef_dim)
        self.generator = Generator(z_dim=z_dim, gf_dim=gf_dim,
                                   gf_split=self.gf_split, L1reg=L1reg)

        self.apply(weights_init)

    def forward(self, voxels, points=None, junction_points=None,
                endpoint_points=None, mode='train'):
        z = self.encoder(voxels)  # [B,z_dim]

        if points is None:
            raise ValueError("points must be provided")

        logits = self.generator(points, z, junction_points, endpoint_points)  # [B,N,K]
        logits_max, _ = torch.max(logits, dim=2, keepdim=True)  # [B,N,1]

        if mode == 'train':
            # 训练：用 max-logit 做 BCEWithLogits，保留 branch_logits 以后可 debug
            return logits, logits_max
        else:
            # 推理：返回概率
            branch_probs = torch.sigmoid(logits)  # [B,N,K]
            occ_probs = torch.sigmoid(logits_max)  # [B,N,1]
            return branch_probs, occ_probs
    # def forward(self, voxels, points=None, junction_points=None, endpoint_points=None, mode='train'):
    #     z = self.encoder(voxels)
    #
    #     if mode == 'train':
    #         if points is not None:
    #             branch_pred, G = self.generator(points, z, junction_points, endpoint_points)
    #             return branch_pred, G
    #         else:
    #             raise ValueError("In training mode, points must be provided")
    #     else:  # inference
    #         if points is not None:
    #             branch_pred, occupancy = self.generator(points, z, junction_points, endpoint_points)
    #             return branch_pred, occupancy
    #         else:
    #             raise ValueError("In inference mode, points must be provided")


# ==========================================
# Training Wrapper & Data Loader
# ==========================================
class BAE_NET_Wrapper:
    def __init__(self,
                 L1reg=True,
                 checkpoint_dir='checkpoint/model_skeleton',
                 sample_dir='samples_skeleton',
                 data_dir='./data'):

        self.L1reg = L1reg
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.data_dir = data_dir

        self._load_data()

        gf_split = 6
        self.model = BAE_Net(
            z_dim=128, ef_dim=32, gf_dim=256,
            gf_split=gf_split,
            L1reg=L1reg
        )

        self.optimizer = None
        self.scheduler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def _load_data(self):
        print(f"\n{'=' * 20} 开始加载数据 (DEBUG模式) {'=' * 20}")
        print(f"数据目录: {self.data_dir}")

        if not os.path.exists(self.data_dir):
            print("❌ 目录不存在")
            return

        files = [f for f in os.listdir(self.data_dir) if f.endswith('.npz')]
        if not files:
            print("❌ 目录下没有 .npz 文件")
            self.data_voxels = np.zeros((0, 64, 64, 64))
            self.data_points = np.zeros((0, 100, 3))
            return

        voxels_list = []
        points_list = []
        occupancy_list = []
        occupancy_vox_list = []
        junction_list = []
        endpoint_list = []

        for file_idx, npz_name in enumerate(files):  # ★ 用 file_idx 表示第几个文件
            full_path = os.path.join(self.data_dir, npz_name)
            try:
                data = np.load(full_path)

                # --- 1. 读取基础数据 ---
                vox = data['voxels']  # [64,64,64]
                pts = data['sdf_points']  # [N,3]

                # SDF-based occupancy
                if 'occu_values' in data:
                    occ = data['occu_values']
                elif 'sdf_values' in data:
                    occ = (data['sdf_values'] <= 0).astype(np.float32)
                else:
                    print(f"⚠️ {npz_name}: 缺少 occu_values/sdf_values，跳过")
                    continue

                # --- 1b. voxel-based occupancy (在同一批 pts 上采样) ---
                dim = vox.shape[0]
                coords = (pts + 1.0) / 2.0
                idx_pts = np.clip((coords * (dim - 1)).astype(np.int32), 0, dim - 1)  # ★ 改名
                xi, yi, zi = idx_pts[:, 0], idx_pts[:, 1], idx_pts[:, 2]
                occ_vox = vox[xi, yi, zi].astype(np.float32)

                # --- 2. 骨骼 ---
                keys = list(data.files)
                j_pts = np.asarray(data["junction_points"], dtype=np.float32) \
                    if "junction_points" in keys else np.zeros((0, 3), dtype=np.float32)
                e_pts = np.asarray(data["endpoint_points"], dtype=np.float32) \
                    if "endpoint_points" in keys else np.zeros((0, 3), dtype=np.float32)

                if file_idx < 3:  # ★ 这里用 file_idx，而不是 idx_pts
                    print(f"\n📄 文件: {npz_name}")

                # 3. 尺度/对齐 debug 同你原来的逻辑，这里略写：
                inside_pts = pts[occ > 0.5]
                if len(inside_pts) == 0:
                    print("  ⚠️ 警告: 该物体没有内部点")
                    c_scale = 1.0
                    c_center = np.zeros(3)
                else:
                    c_min, c_max = np.min(inside_pts, axis=0), np.max(inside_pts, axis=0)
                    c_scale = np.max(c_max - c_min)
                    c_center = (c_min + c_max) / 2.0
                    print(f"  🔹 物体(内部点): 尺度={c_scale:.4f}, 中心={c_center}")

                j_pts = j_pts[:, [1, 0, 2]]
                e_pts = e_pts[:, [1, 0, 2]]
                if len(j_pts) > 0:
                    j_pts = (j_pts / (dim - 1)) * 2.0 - 1.0
                if len(e_pts) > 0:
                    e_pts = (e_pts / (dim - 1)) * 2.0 - 1.0
                # ...（后面你的骨骼尺度匹配逻辑保持不变，只把所有 if idx < 3 改成 if file_idx < 3）...

                # 5. 存入列表
                voxels_list.append(vox)
                points_list.append(pts)
                occupancy_list.append(occ)
                occupancy_vox_list.append(occ_vox)
                junction_list.append(j_pts)
                endpoint_list.append(e_pts)

            except Exception as e:
                print(f"Skipping {npz_name}: {e}")
                continue

        # ==== 收尾 ====
        if len(voxels_list) == 0:
            print("❌ 没有成功加载任何 shape，检查上面的 Skipping 信息。")
            # 防止后面再 ZeroDivisionError
            self.data_voxels = np.zeros((0, 64, 64, 64))
            self.data_points = np.zeros((0, 100, 3))
            self.data_occupancy = np.zeros((0, 100), dtype=np.float32)
            self.data_occupancy_vox = np.zeros((0, 100), dtype=np.float32)
            return

        self.data_voxels = np.array(voxels_list)
        self.data_points = np.array(points_list)
        self.data_occupancy = np.array(occupancy_list)
        self.data_occupancy_vox = np.array(occupancy_vox_list)

        max_j = max(max([len(j) for j in junction_list], default=0), 1)
        max_e = max(max([len(e) for e in endpoint_list], default=0), 1)

        self.data_junctions = np.full((len(junction_list), max_j, 3), 10.0, dtype=np.float32)
        self.data_endpoints = np.full((len(endpoint_list), max_e, 3), 10.0, dtype=np.float32)

        for i, pts in enumerate(junction_list):
            if len(pts) > 0:
                self.data_junctions[i, :len(pts), :] = pts
        for i, pts in enumerate(endpoint_list):
            if len(pts) > 0:
                self.data_endpoints[i, :len(pts), :] = pts

        print(f"\n{'=' * 20} 加载完成 {'=' * 20}")
        print(f"Loaded {len(voxels_list)} shapes.")
        print(f"Max Junctions: {max_j}, Max Endpoints: {max_e}")

    def _train_unsupervised(self, config):
        self.model.train()

        # 1. Define Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, 0.999)
        )

        # 2. Define Learning Rate Scheduler (CosineAnnealing)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epoch,
            eta_min=1e-6
        )

        # Resume logic
        start_epoch = 0
        if hasattr(config, 'resume_epoch') and config.resume_epoch > 0:
            self.load_checkpoint(config.resume_epoch)
            start_epoch = config.resume_epoch

        num_shapes = len(self.data_points)
        indices = np.arange(num_shapes)
        epoch_bar = tqdm(range(start_epoch, config.epoch), desc="Training", unit="epoch")

        for epoch in epoch_bar:
            np.random.shuffle(indices)
            total_loss = 0

            for idx in range(num_shapes):
                shape_idx = indices[idx]

                # Sampling Logic
                num_sample = 20000
                occupancy = self.data_occupancy_vox[shape_idx]

                pos_idx = np.where(occupancy == 1)[0]
                neg_idx = np.where(occupancy == 0)[0]
                # occupancy = self.data_occupancy[shape_idx]
                # pos_idx = np.where(occupancy == 1)[0]
                # neg_idx = np.where(occupancy == 0)[0]

                if len(pos_idx) == 0 or len(neg_idx) == 0:
                    continue

                current_sample_num = min(len(pos_idx), len(neg_idx), num_sample // 2)
                sample_pos = np.random.choice(pos_idx, size=current_sample_num, replace=False)
                sample_neg = np.random.choice(neg_idx, size=current_sample_num, replace=False)
                sample_idx = np.concatenate([sample_pos, sample_neg])
                np.random.shuffle(sample_idx)

                # Prepare Tensors
                batch_voxels = torch.from_numpy(self.data_voxels[shape_idx]).float().unsqueeze(0).unsqueeze(0).to(
                    self.device)
                batch_points = torch.from_numpy(self.data_points[shape_idx][sample_idx]).float().unsqueeze(0).to(
                    self.device)
                # batch_values = torch.from_numpy(self.data_occupancy[shape_idx][sample_idx]).float().unsqueeze(
                #     0).unsqueeze(2).to(self.device)
                batch_values = torch.from_numpy(occupancy[sample_idx]).float().unsqueeze(0).unsqueeze(2).to(self.device)

                batch_junc = torch.from_numpy(self.data_junctions[shape_idx]).float().unsqueeze(0).to(self.device)
                batch_end = torch.from_numpy(self.data_endpoints[shape_idx]).float().unsqueeze(0).to(self.device)

                # Forward pass
                branch_pred, pred_occupancy = self.model(
                    batch_voxels,
                    batch_points,
                    junction_points=batch_junc,
                    endpoint_points=batch_end,
                    mode='train'
                )
                loss = F.binary_cross_entropy_with_logits(
                    pred_occupancy,
                    batch_values  # 0/1 occupancy，从 SDF 来的
                )
                # loss = F.mse_loss(pred_occupancy.squeeze(1), batch_values)
                # Forward pass
                # 注意：我们需要 Generator 返回 branches (G_raw) 和 final (G)
                # 请修改 BAE_Net 的 forward 让它返回两个值: return h3, G
                # G_branches, pred_occupancy = self.model(
                #     batch_voxels,
                #     batch_points,
                #     junction_points=batch_junc,
                #     endpoint_points=batch_end,
                #     mode='train_debug'  # <--- 需要改一下 model forward
                # )
                #
                # loss = F.mse_loss(pred_occupancy.squeeze(1), batch_values)
                #
                # # ================= DEBUG PROBES (不要只看 Loss!) =================
                # with torch.no_grad():
                #     # 1. 看看 GT 是不是全是 0 (防止数据加载错)
                #     gt_mean = batch_values.mean().item()
                #
                #     # 2. 看看预测是不是全是 0 (模型摆烂)
                #     pred_mean = pred_occupancy.mean().item()
                #     pred_max = pred_occupancy.max().item()
                #
                #     # 3. 看看每个 Branch 的激活情况 (Dead Neuron 检测)
                #     # G_branches shape: [B, N, K]
                #     branch_activations = G_branches.mean(dim=(0, 1)).cpu().numpy()
                #
                #     if idx % 100 == 0:  # 每100个batch打印一次
                #         print(f"\n[Batch {idx}] Loss: {loss.item():.6f}")
                #         print(f"  > GT Mean   : {gt_mean:.4f} (Should be ~0.5)")
                #         print(f"  > Pred Mean : {pred_mean:.4f} (If ~0.0, model is outputting all empty)")
                #         print(f"  > Pred Max  : {pred_max:.4f} (If <0.5, model never predicts 'inside')")
                #         print(f"  > Branch Act: {branch_activations} (Look for dead branches)")
                # # =================================================================
                # # ... (L1 reg code)
                if self.L1reg:
                    l1_reg = torch.norm(self.model.generator.fc3.weight, 1)
                    loss += 1e-6 * l1_reg

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            # 3. Update Scheduler
            self.scheduler.step()

            current_lr = self.scheduler.get_last_lr()[0]
            avg_loss = total_loss / num_shapes
            epoch_bar.set_postfix(avg_loss=f"{avg_loss:.6f}", lr=f"{current_lr:.6f}")

            if epoch % 10000 == 0 and epoch > 0:
                self.save_checkpoint(epoch, avg_loss)

    def save_checkpoint(self, epoch, loss):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        scheduler_state = self.scheduler.state_dict() if self.scheduler else None
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': scheduler_state,
            'loss': loss
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, f'ckpt_epoch_{epoch}.pth'))
        print(f"Saved checkpoint to {self.checkpoint_dir}/ckpt_epoch_{epoch}.pth")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']

    # ==========================================
    # Fixed: Segmentation Test Function
    # ==========================================
    def test_segmentation(self, test_points, test_voxels, junction_points=None, endpoint_points=None,
                          use_postprocessing=False):
        """
        Robust testing function with DEBUG prints to verify skeleton input.
        """
        self.model.eval()

        print("\n" + "=" * 40)
        print("[DEBUG] 进入 test_segmentation 函数")

        # --- Debug 1: 检查原始输入 ---
        if junction_points is None:
            print("  ❌ [Input] junction_points is NONE")
        else:
            print(f"  ✅ [Input] junction_points shape: {junction_points.shape}")
            if len(junction_points) > 0:
                print(f"     -> Range: Min {junction_points.min():.3f}, Max {junction_points.max():.3f}")
            else:
                print("     -> Warning: Array is empty!")

        if endpoint_points is None:
            print("  ❌ [Input] endpoint_points is NONE")
        else:
            print(f"  ✅ [Input] endpoint_points shape: {endpoint_points.shape}")
            if len(endpoint_points) > 0:
                print(f"     -> Range: Min {endpoint_points.min():.3f}, Max {endpoint_points.max():.3f}")
            else:
                print("     -> Warning: Array is empty!")

        # --- 1. Data Consistency Check (Auto-Rescale) ---
        points_in = test_points.copy()
        p_min, p_max = points_in.min(), points_in.max()

        j_in = junction_points.copy() if junction_points is not None else None
        e_in = endpoint_points.copy() if endpoint_points is not None else None

        # if p_min < -0.6 or p_max > 0.6:
        #     print(f"  ⚠️ [Rescale] 检测到点云范围 [{p_min:.2f}, {p_max:.2f}]，执行 /= 2.0 缩放")
        #     # points_in /= 2.0
        #     if j_in is not None: j_in /= 2.0
        #     if e_in is not None: e_in /= 2.0
        #
        #     # Debug Rescale result
        #     if e_in is not None and len(e_in) > 0:
        #         print(f"     -> Skeleton Rescaled Range: {e_in.min():.3f} ~ {e_in.max():.3f}")

        with torch.no_grad():
            # --- 2. Handle Voxel Dimensions ---
            t_voxels = torch.FloatTensor(test_voxels).to(self.device)
            if t_voxels.ndim == 3:
                batch_voxels = t_voxels.unsqueeze(0).unsqueeze(0)
            elif t_voxels.ndim == 4:
                batch_voxels = t_voxels.unsqueeze(1)
            else:
                batch_voxels = t_voxels

            # --- 3. Handle Point Dimensions ---
            if points_in.ndim == 2:
                batch_points = torch.FloatTensor(points_in).unsqueeze(0).to(self.device)
            else:
                batch_points = torch.FloatTensor(points_in).to(self.device)

            # --- 4. Handle Skeleton Points ---
            def get_dummy_tensor():
                print("  ⚠️ [Dummy] 生成了 Dummy Tensor (10.0) 填充缺失的骨骼分支")
                return torch.ones(1, 1, 3).to(self.device) * 10.0

            j_tensor, e_tensor = None, None

            if j_in is not None and len(j_in) > 0:
                j_tensor = torch.FloatTensor(j_in).unsqueeze(0).to(self.device)

            if e_in is not None and len(e_in) > 0:
                e_tensor = torch.FloatTensor(e_in).unsqueeze(0).to(self.device)

            # CRITICAL: Ensure both are present
            if j_tensor is not None and e_tensor is None:
                e_tensor = get_dummy_tensor()
            if e_tensor is not None and j_tensor is None:
                j_tensor = get_dummy_tensor()

            # --- Debug 2: 检查最终传给模型的 Tensor ---
            print("  [Model Input Check]")
            if j_tensor is not None:
                print(f"     -> J_Tensor sent to model: {j_tensor.shape}")
            else:
                print("     -> J_Tensor is None (Model will use internal Fallback)")

            if e_tensor is not None:
                print(f"     -> E_Tensor sent to model: {e_tensor.shape}")
            else:
                print("     -> E_Tensor is None (Model will use internal Fallback)")

            print("=" * 40 + "\n")

            # --- 5. Inference ---
            branch_pred, _ = self.model(
                batch_voxels,
                batch_points,
                junction_points=j_tensor,
                endpoint_points=e_tensor,
                mode='inference'
            )

            branch_pred = branch_pred.squeeze(0).cpu().numpy()

            if use_postprocessing:
                branch_pred = self._postprocess_segmentation(branch_pred, points_in)

            return branch_pred

    def _postprocess_segmentation(self, predictions, points):
        from sklearn.neighbors import KDTree

        valid_mask = np.max(predictions, axis=1) > 1e-2
        valid_points = points[valid_mask]
        valid_pred = predictions[valid_mask]
        valid_labels = np.argmax(valid_pred, axis=1)

        if len(valid_points) > 0:
            kdtree = KDTree(valid_points, leaf_size=8)
            _, nearest_idx = kdtree.query(points, k=1)
            final_labels = valid_labels[nearest_idx.flatten()]

            one_hot = np.zeros_like(predictions)
            one_hot[np.arange(len(final_labels)), final_labels] = 1
            return one_hot

        return predictions


    def generate_gt_mesh_from_voxels(self, voxels, threshold=0.5):
        """
        使用原始 voxels 生成 ground truth mesh。
        支持输入形状:
        - [D, H, W]
        - [1, D, H, W]
        - [1, 1, D, H, W]
        """
        try:
            import mcubes
        except ImportError:
            print("Please install PyMCubes: pip install PyMCubes")
            return None, None

        vol = np.asarray(voxels)

        # 自动去掉前面的 batch/channel 维度
        while vol.ndim > 3 and vol.shape[0] == 1:
            vol = vol[0]

        if vol.ndim != 3:
            raise ValueError(f"[GT Mesh] Expect 3D volume, got shape {vol.shape}")

        vol = vol.astype(np.float32)

        dim = vol.shape[0]
        vertices, triangles = mcubes.marching_cubes(vol, threshold)

        if len(vertices) == 0:
            print("[GT Mesh] marching_cubes 找不到等值面 (可能 voxels 全 0 或全 1)")
            return None, None

        # 和预测 mesh 一样做归一化
        vertices = vertices / dim - 0.5
        return vertices, triangles


    # ==========================================
    # Added: Mesh Generation Function
    # ==========================================
    def generate_mesh(self, voxels, junction_points=None, endpoint_points=None, threshold=0.5):
        try:
            import mcubes
        except ImportError:
            print("Please install PyMCubes: pip install PyMCubes")
            return None, None, None, None

        self.model.eval()

        with torch.no_grad():
            # --- 1. Handle Voxel Dimensions ---
            t_voxels = torch.FloatTensor(voxels).to(self.device)
            if t_voxels.ndim == 3:
                batch_voxels = t_voxels.unsqueeze(0).unsqueeze(0)
            elif t_voxels.ndim == 4:
                batch_voxels = t_voxels.unsqueeze(1)
            else:
                batch_voxels = t_voxels

            # Get Shape Code (Z)
            z = self.model.encoder(batch_voxels)

            # --- 2. Handle Skeleton Points ---
            def get_dummy_tensor():
                return torch.ones(1, 1, 3).to(self.device) * 10.0

            j_tensor, e_tensor = None, None

            if junction_points is not None and len(junction_points) > 0:
                j_tensor = torch.FloatTensor(junction_points).unsqueeze(0).to(self.device)

            if endpoint_points is not None and len(endpoint_points) > 0:
                e_tensor = torch.FloatTensor(endpoint_points).unsqueeze(0).to(self.device)

            if j_tensor is not None and e_tensor is None:
                e_tensor = get_dummy_tensor()
            if e_tensor is not None and j_tensor is None:
                j_tensor = get_dummy_tensor()

            # --- 3. Create Grid ---
            dim = 64
            coords = np.linspace(-1, 1, dim)
            grid_x, grid_y, grid_z = np.meshgrid(coords, coords, coords, indexing='ij')
            grid_points = np.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], axis=1)

            # --- 4. Batch Inference ---
            batch_size = 8192
            predictions = []
            total_mesh = []

            for i in range(0, len(grid_points), batch_size):
                batch = grid_points[i:i + batch_size]
                batch_tensor = torch.FloatTensor(batch).unsqueeze(0).to(self.device)

                branch_prob, total_prob = self._generator_infer(
                    batch_tensor,
                    z,
                    j_tensor,
                    e_tensor
                )

                # branch_prob: [1, B, K]
                # total_prob : [1, B, 1]
                predictions.append(branch_prob.squeeze(0).cpu().numpy())
                total_mesh.append(total_prob.squeeze(0).cpu().numpy())

            predictions = np.concatenate(predictions, axis=0)
            predictions = predictions.reshape(dim, dim, dim, -1)

            total_mesh = np.concatenate(total_mesh, axis=0)
            total_mesh = total_mesh.reshape(dim, dim, dim)

            predictions = []
            total_mesh = []

            for i in range(0, len(grid_points), batch_size):
                batch = grid_points[i:i + batch_size]
                batch_tensor = torch.FloatTensor(batch).unsqueeze(0).to(self.device)  # [1,B,3]

                branch_logits = self.model.generator(
                    batch_tensor, z,
                    junction_points=j_tensor,
                    endpoint_points=e_tensor
                )  # [1,B,K]
                branch_probs = torch.sigmoid(branch_logits)  # [1,B,K]
                occ_probs, _ = torch.max(branch_probs, dim=2, keepdim=True)  # [1,B,1]

                predictions.append(branch_probs.squeeze(0).cpu().numpy())  # [B,K]
                total_mesh.append(occ_probs.squeeze(0).cpu().numpy())  # [B,1]

            predictions = np.concatenate(predictions, axis=0)  # [N,K]
            total_mesh = np.concatenate(total_mesh, axis=0)  # [N,1]

            predictions = predictions.reshape(dim, dim, dim, -1)
            total_mesh = total_mesh.reshape(dim, dim, dim)

            all_vertices, all_triangles = [], []
            K = predictions.shape[-1]

            for k in range(K):
                field = predictions[:, :, :, k]
                verts, tris = mcubes.marching_cubes(field, threshold)
                if len(verts) > 0:
                    verts = verts / dim - 0.5
                    all_vertices.append(verts)
                    all_triangles.append(tris)
                else:
                    all_vertices.append(None)
                    all_triangles.append(None)

            total_vertices, total_triangles = mcubes.marching_cubes(total_mesh, threshold)
            if len(total_vertices) > 0:
                total_vertices = total_vertices / dim - 0.5

            print("[Mesh Debug] total_mesh min/max:", total_mesh.min(), total_mesh.max())
            print("[Mesh Debug] mean:", total_mesh.mean())

            return all_vertices, all_triangles, total_vertices, total_triangles

    def evaluate_shape_iou(self, shape_idx=0, threshold=0.5, batch_size=8192):
        """
        在 64^3 规则网格上，用当前模型 + 骨骼预测 occupancy，
        和 voxel GT 做 IoU。
        """
        self.model.eval()
        device = self.device

        with torch.no_grad():
            # ---------- 1. GT 体素 ----------
            voxels = self.data_voxels[shape_idx]  # [64,64,64]
            gt = (voxels > 0.5).astype(np.int32)

            # ---------- 2. 编码 z ----------
            t_vox = torch.from_numpy(voxels).float().unsqueeze(0).unsqueeze(0).to(device)
            z = self.model.encoder(t_vox)  # [1,z_dim]

            # ---------- 3. 取骨骼（过滤掉 padding=10.0） ----------
            raw_j = self.data_junctions[shape_idx]  # [J_pad,3]
            raw_e = self.data_endpoints[shape_idx]  # [E_pad,3]

            mask_j = np.abs(raw_j[:, 0] - 10.0) > 1e-4
            mask_e = np.abs(raw_e[:, 0] - 10.0) > 1e-4
            j_valid = raw_j[mask_j]
            e_valid = raw_e[mask_e]

            def to_tensor(arr):
                if arr is None or len(arr) == 0:
                    return None
                return torch.from_numpy(arr).float().unsqueeze(0).to(device)

            j_tensor = to_tensor(j_valid)
            e_tensor = to_tensor(e_valid)

            def get_dummy():
                # 和 test_segmentation / generate_mesh 的逻辑保持一致
                return torch.ones(1, 1, 3, device=device) * 10.0

            if j_tensor is not None and e_tensor is None:
                e_tensor = get_dummy()
            if e_tensor is not None and j_tensor is None:
                j_tensor = get_dummy()

            # ---------- 4. 在 [-1,1]^3 网格上预测 ----------
            dim = voxels.shape[0]
            coords = np.linspace(-1.0, 1.0, dim)
            gx, gy, gz = np.meshgrid(coords, coords, coords, indexing='ij')
            grid_points = np.stack([gx.flatten(), gy.flatten(), gz.flatten()], axis=1)

            preds = []
            for i in range(0, grid_points.shape[0], batch_size):
                batch_np = grid_points[i:i + batch_size]  # [B,3]
                batch_tensor = torch.from_numpy(batch_np).float().unsqueeze(0).to(device)  # [1,B,3]

                # 统一用 _generator_infer，拿到整体 occupancy 概率
                _, total_prob = self._generator_infer(
                    batch_tensor,
                    z,
                    j_tensor,
                    e_tensor
                )  # [1,B,1]

                preds.append(total_prob.squeeze(0).cpu().numpy())  # [B,1]

            pred_grid = np.concatenate(preds, axis=0).reshape(dim, dim, dim)

            # ---------- 5. 二值化 & IoU ----------
            pred_occ = (pred_grid >= threshold).astype(np.int32)

            inter = np.logical_and(gt == 1, pred_occ == 1).sum()
            union = np.logical_or(gt == 1, pred_occ == 1).sum()
            iou = inter / (union + 1e-8)

            print(f"[IoU] shape {shape_idx}: IoU={iou:.4f}, "
                  f"pred_vol={pred_occ.sum()}, gt_vol={gt.sum()}")
            return iou

    def _generator_infer(self, points, z, j_tensor=None, e_tensor=None):
        """
        统一封装 generator 的输出：
        - branch_prob: [B, N, K]，每个 branch 的 occupancy 概率
        - total_prob:  [B, N, 1]，对所有 branch 做 max 后的整体 occupancy 概率
        无论你的 Generator.forward 返回 1 个还是 2 个，这里都转成这两个东西。
        """
        out = self.model.generator(
            points,
            z,
            junction_points=j_tensor,
            endpoint_points=e_tensor
        )

        # 1. 解包：1 个返回值 or (branch, total)
        if isinstance(out, tuple):
            branch_raw, total_raw = out
        else:
            branch_raw = out
            total_raw = None

        # 2. 统一成“概率”
        #   - 如果数值明显不在 [0,1]，当成 logits，用 sigmoid
        #   - 如果已经在 [0,1]，直接用
        with torch.no_grad():
            bmin = float(branch_raw.min())
            bmax = float(branch_raw.max())

        if bmin < 0.0 or bmax > 1.0:
            branch_prob = torch.sigmoid(branch_raw)
        else:
            branch_prob = branch_raw

        # 3. total_prob：如果 Generator 没给，就自己 max 一下
        if total_raw is not None:
            # 也做一下 sigmoid 防守
            with torch.no_grad():
                tmin = float(total_raw.min())
                tmax = float(total_raw.max())
            if tmin < 0.0 or tmax > 1.0:
                total_prob = torch.sigmoid(total_raw)
            else:
                total_prob = total_raw
        else:
            # 自己根据 branch_prob 取 max
            total_prob, _ = torch.max(branch_prob, dim=2, keepdim=True)

        return branch_prob, total_prob
    def evaluate_shape_iou_sdf(self, shape_idx=0, threshold=0.5, batch_size=8192):
        """
        在 SDF 采样点上算 IoU 和 MSE：
          - GT: self.data_points[shape_idx], self.data_occupancy[shape_idx]
          - Pred: 用当前模型 + 骨骼预测这些点的 occupancy 概率
        """
        self.model.eval()
        device = self.device

        with torch.no_grad():
            # ---------- 1. 取 SDF 点 & GT occupancy ----------
            pts = self.data_points[shape_idx]             # [N,3] in [-1,1]^3
            occ = self.data_occupancy[shape_idx]          # [N,] 0/1 (from SDF)
            gt_bin = (occ > 0.5).astype(np.int32)
            N = pts.shape[0]

            # ---------- 2. 编码 z（和训练完全一致） ----------
            voxels = self.data_voxels[shape_idx]          # [64,64,64]
            t_vox = torch.from_numpy(voxels).float().unsqueeze(0).unsqueeze(0).to(device)
            z = self.model.encoder(t_vox)                 # [1,z_dim]

            # ---------- 3. 骨骼（同上：过滤 padding=10） ----------
            raw_j = self.data_junctions[shape_idx]
            raw_e = self.data_endpoints[shape_idx]

            mask_j = np.abs(raw_j[:, 0] - 10.0) > 1e-4
            mask_e = np.abs(raw_e[:, 0] - 10.0) > 1e-4
            j_valid = raw_j[mask_j]
            e_valid = raw_e[mask_e]

            def to_tensor(arr):
                if arr is None or len(arr) == 0:
                    return None
                return torch.from_numpy(arr).float().unsqueeze(0).to(device)

            j_tensor = to_tensor(j_valid)
            e_tensor = to_tensor(e_valid)

            def get_dummy():
                return torch.ones(1, 1, 3, device=device) * 10.0

            if j_tensor is not None and e_tensor is None:
                e_tensor = get_dummy()
            if e_tensor is not None and j_tensor is None:
                j_tensor = get_dummy()

            # ---------- 4. 在同一批 SDF 点上预测 ----------
            preds = []
            for i in range(0, N, batch_size):
                batch_np = pts[i:i + batch_size]                   # [B,3]
                batch_tensor = torch.from_numpy(batch_np).float().unsqueeze(0).to(device)  # [1,B,3]

                _, total_prob = self._generator_infer(
                    batch_tensor,
                    z,
                    j_tensor,
                    e_tensor
                )  # [1,B,1]

                preds.append(total_prob.squeeze(0).cpu().numpy())  # [B,1]

            pred_full = np.concatenate(preds, axis=0).reshape(-1)  # [N,]

            # ---------- 5. IoU + MSE ----------
            pred_bin = (pred_full >= threshold).astype(np.int32)

            inter = np.logical_and(gt_bin == 1, pred_bin == 1).sum()
            union = np.logical_or(gt_bin == 1, pred_bin == 1).sum()
            iou = inter / (union + 1e-8)

            mse = ((pred_full - gt_bin.astype(np.float32)) ** 2).mean()

            print(f"[SDF IoU] shape {shape_idx}: IoU={iou:.4f}, "
                  f"MSE={mse:.6f}, pred_pos={pred_bin.sum()}, "
                  f"gt_pos={gt_bin.sum()}, N={N}")
            return iou, mse
