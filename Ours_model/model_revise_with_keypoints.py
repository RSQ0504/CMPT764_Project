import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

USE_SKELETON = True  

def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('InstanceNorm') != -1:
        if m.weight is not None:
            nn.init.constant_(m.weight.data, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


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

        pointz = torch.cat([points, z_expanded], dim=2)  # [B,N,z_dim+3]
        h1 = F.leaky_relu(self.fc1(pointz), 0.02)  # [B,N,4*gf_dim]

        if USE_SKELETON and (junction_points is not None) and (endpoint_points is not None):
            d_junc = self.compute_min_distance(points, junction_points)  # [B,N,1]
            d_end = self.compute_min_distance(points, endpoint_points)  # [B,N,1]
            h1_aug = torch.cat([h1, d_junc, d_end], dim=2)  # [B,N,4*gf_dim+2]
        else:
            dummy = torch.ones(B, N, 1, device=points.device) * 10.0
            h1_aug = torch.cat([h1, dummy, dummy], dim=2)

        h2 = F.leaky_relu(self.fc2(h1_aug), 0.02)  # [B,N,gf_dim]

        logits = self.fc3(h2)  # [B,N,K]
        return logits

class Net(nn.Module):
    def __init__(self, z_dim=128, ef_dim=32, gf_dim=256, gf_split=4, L1reg=False):
        super(Net, self).__init__()
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
            return logits, logits_max
        else:
            branch_probs = torch.sigmoid(logits)  # [B,N,K]
            occ_probs = torch.sigmoid(logits_max)  # [B,N,1]
            return branch_probs, occ_probs
    
class NET_Wrapper:
    def __init__(self,
                 L1reg=True,
                 checkpoint_dir='checkpoint/model_skeleton',
                 sample_dir='samples_skeleton',
                 data_dir='./data', gf_split=4):

        self.L1reg = L1reg
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.data_dir = data_dir

        self._load_data()

        self.model = Net(
            z_dim=128, ef_dim=32, gf_dim=256,
            gf_split=gf_split,
            L1reg=L1reg
        )

        self.optimizer = None
        self.scheduler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def _load_data(self):
        if not os.path.exists(self.data_dir):
            return

        files = [f for f in os.listdir(self.data_dir) if f.endswith('.npz')]
        if not files:
            self.data_voxels = np.zeros((0, 64, 64, 64))
            self.data_points = np.zeros((0, 100, 3))
            return

        voxels_list = []
        points_list = []
        occupancy_list = []
        occupancy_vox_list = []
        junction_list = []
        endpoint_list = []

        for file_idx, npz_name in enumerate(files): 
            full_path = os.path.join(self.data_dir, npz_name)
            try:
                data = np.load(full_path)


                vox = data['voxels']  # [64,64,64]
                pts = data['sdf_points']  # [N,3]

                # SDF-based occupancy
                if 'occu_values' in data:
                    occ = data['occu_values']
                elif 'sdf_values' in data:
                    occ = (data['sdf_values'] <= 0).astype(np.float32)
                else:
                    continue

                dim = vox.shape[0]
                coords = (pts + 1.0) / 2.0
                idx_pts = np.clip((coords * (dim - 1)).astype(np.int32), 0, dim - 1)
                xi, yi, zi = idx_pts[:, 0], idx_pts[:, 1], idx_pts[:, 2]
                occ_vox = vox[xi, yi, zi].astype(np.float32)

                keys = list(data.files)
                j_pts = np.asarray(data["junction_points"], dtype=np.float32) \
                    if "junction_points" in keys else np.zeros((0, 3), dtype=np.float32)
                e_pts = np.asarray(data["endpoint_points"], dtype=np.float32) \
                    if "endpoint_points" in keys else np.zeros((0, 3), dtype=np.float32)


                inside_pts = pts[occ > 0.5]
                if len(inside_pts) == 0:
                    c_scale = 1.0
                    c_center = np.zeros(3)
                else:
                    c_min, c_max = np.min(inside_pts, axis=0), np.max(inside_pts, axis=0)
                    c_scale = np.max(c_max - c_min)
                    c_center = (c_min + c_max) / 2.0

                j_pts = j_pts[:, [1, 0, 2]]
                e_pts = e_pts[:, [1, 0, 2]]
                if len(j_pts) > 0:
                    j_pts = (j_pts / (dim - 1)) * 2.0 - 1.0
                if len(e_pts) > 0:
                    e_pts = (e_pts / (dim - 1)) * 2.0 - 1.0

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

        print(f"Loaded {len(voxels_list)} shapes.")
        print(f"Max Junctions: {max_j}, Max Endpoints: {max_e}")

    def _train_unsupervised(self, config):
        self.model.train()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, 0.999)
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epoch,
            eta_min=1e-6
        )

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

                num_sample = 20000
                occupancy = self.data_occupancy_vox[shape_idx]

                pos_idx = np.where(occupancy == 1)[0]
                neg_idx = np.where(occupancy == 0)[0]

                if len(pos_idx) == 0 or len(neg_idx) == 0:
                    continue

                current_sample_num = min(len(pos_idx), len(neg_idx), num_sample // 2)
                sample_pos = np.random.choice(pos_idx, size=current_sample_num, replace=False)
                sample_neg = np.random.choice(neg_idx, size=current_sample_num, replace=False)
                sample_idx = np.concatenate([sample_pos, sample_neg])
                np.random.shuffle(sample_idx)

                batch_voxels = torch.from_numpy(self.data_voxels[shape_idx]).float().unsqueeze(0).unsqueeze(0).to(
                    self.device)
                batch_points = torch.from_numpy(self.data_points[shape_idx][sample_idx]).float().unsqueeze(0).to(
                    self.device)

                batch_values = torch.from_numpy(occupancy[sample_idx]).float().unsqueeze(0).unsqueeze(2).to(self.device)

                batch_junc = torch.from_numpy(self.data_junctions[shape_idx]).float().unsqueeze(0).to(self.device)
                batch_end = torch.from_numpy(self.data_endpoints[shape_idx]).float().unsqueeze(0).to(self.device)

                branch_pred, pred_occupancy = self.model(
                    batch_voxels,
                    batch_points,
                    junction_points=batch_junc,
                    endpoint_points=batch_end,
                    mode='train'
                )
                loss = F.binary_cross_entropy_with_logits(
                    pred_occupancy,
                    batch_values  
                )
                
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


    def test_segmentation(self, test_points, test_voxels, junction_points=None, endpoint_points=None,
                          use_postprocessing=False):

        self.model.eval()

        print("\n" + "=" * 40)

        points_in = test_points.copy()
        p_min, p_max = points_in.min(), points_in.max()

        j_in = junction_points.copy() if junction_points is not None else None
        e_in = endpoint_points.copy() if endpoint_points is not None else None


        with torch.no_grad():
            t_voxels = torch.FloatTensor(test_voxels).to(self.device)
            if t_voxels.ndim == 3:
                batch_voxels = t_voxels.unsqueeze(0).unsqueeze(0)
            elif t_voxels.ndim == 4:
                batch_voxels = t_voxels.unsqueeze(1)
            else:
                batch_voxels = t_voxels

            if points_in.ndim == 2:
                batch_points = torch.FloatTensor(points_in).unsqueeze(0).to(self.device)
            else:
                batch_points = torch.FloatTensor(points_in).to(self.device)

            def get_dummy_tensor():
                return torch.ones(1, 1, 3).to(self.device) * 10.0

            j_tensor, e_tensor = None, None

            if j_in is not None and len(j_in) > 0:
                j_tensor = torch.FloatTensor(j_in).unsqueeze(0).to(self.device)

            if e_in is not None and len(e_in) > 0:
                e_tensor = torch.FloatTensor(e_in).unsqueeze(0).to(self.device)

            if j_tensor is not None and e_tensor is None:
                e_tensor = get_dummy_tensor()
            if e_tensor is not None and j_tensor is None:
                j_tensor = get_dummy_tensor()

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
            return None, None

        vertices = vertices / dim - 0.5
        return vertices, triangles



    def generate_mesh(self, voxels, junction_points=None, endpoint_points=None, threshold=0.5):
        try:
            import mcubes
        except ImportError:
            print("Please install PyMCubes: pip install PyMCubes")
            return None, None, None, None

        self.model.eval()

        with torch.no_grad():
            t_voxels = torch.FloatTensor(voxels).to(self.device)
            if t_voxels.ndim == 3:
                batch_voxels = t_voxels.unsqueeze(0).unsqueeze(0)
            elif t_voxels.ndim == 4:
                batch_voxels = t_voxels.unsqueeze(1)
            else:
                batch_voxels = t_voxels

            z = self.model.encoder(batch_voxels)

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

            dim = 64
            coords = np.linspace(-1, 1, dim)
            grid_x, grid_y, grid_z = np.meshgrid(coords, coords, coords, indexing='ij')
            grid_points = np.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], axis=1)

            batch_size = 8192


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
            
            max_idx = np.argmax(predictions, axis=-1)
            max_predictions = np.zeros_like(predictions)
            dim_range = np.arange(predictions.shape[0])
            x, y, z = np.ogrid[:predictions.shape[0], :predictions.shape[1], :predictions.shape[2]]
            max_predictions[x, y, z, max_idx] = predictions[x, y, z, max_idx]
            predictions = max_predictions

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

        self.model.eval()
        device = self.device

        with torch.no_grad():
            voxels = self.data_voxels[shape_idx]  # [64,64,64]
            gt = (voxels > 0.5).astype(np.int32)

            t_vox = torch.from_numpy(voxels).float().unsqueeze(0).unsqueeze(0).to(device)
            z = self.model.encoder(t_vox)  # [1,z_dim]

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
                return torch.ones(1, 1, 3, device=device) * 10.0

            if j_tensor is not None and e_tensor is None:
                e_tensor = get_dummy()
            if e_tensor is not None and j_tensor is None:
                j_tensor = get_dummy()

            dim = voxels.shape[0]
            coords = np.linspace(-1.0, 1.0, dim)
            gx, gy, gz = np.meshgrid(coords, coords, coords, indexing='ij')
            grid_points = np.stack([gx.flatten(), gy.flatten(), gz.flatten()], axis=1)

            preds = []
            for i in range(0, grid_points.shape[0], batch_size):
                batch_np = grid_points[i:i + batch_size]  # [B,3]
                batch_tensor = torch.from_numpy(batch_np).float().unsqueeze(0).to(device)  # [1,B,3]

                _, total_prob = self._generator_infer(
                    batch_tensor,
                    z,
                    j_tensor,
                    e_tensor
                )  # [1,B,1]

                preds.append(total_prob.squeeze(0).cpu().numpy())  # [B,1]

            pred_grid = np.concatenate(preds, axis=0).reshape(dim, dim, dim)

            pred_occ = (pred_grid >= threshold).astype(np.int32)

            inter = np.logical_and(gt == 1, pred_occ == 1).sum()
            union = np.logical_or(gt == 1, pred_occ == 1).sum()
            iou = inter / (union + 1e-8)

            print(f"[IoU] shape {shape_idx}: IoU={iou:.4f}, "
                  f"pred_vol={pred_occ.sum()}, gt_vol={gt.sum()}")
            return iou

    def _generator_infer(self, points, z, j_tensor=None, e_tensor=None):

        out = self.model.generator(
            points,
            z,
            junction_points=j_tensor,
            endpoint_points=e_tensor
        )

        if isinstance(out, tuple):
            branch_raw, total_raw = out
        else:
            branch_raw = out
            total_raw = None

        with torch.no_grad():
            bmin = float(branch_raw.min())
            bmax = float(branch_raw.max())

        if bmin < 0.0 or bmax > 1.0:
            branch_prob = torch.sigmoid(branch_raw)
        else:
            branch_prob = branch_raw

        if total_raw is not None:
            with torch.no_grad():
                tmin = float(total_raw.min())
                tmax = float(total_raw.max())
            if tmin < 0.0 or tmax > 1.0:
                total_prob = torch.sigmoid(total_raw)
            else:
                total_prob = total_raw
        else:
            total_prob, _ = torch.max(branch_prob, dim=2, keepdim=True)

        return branch_prob, total_prob
    def evaluate_shape_iou_sdf(self, shape_idx=0, threshold=0.5, batch_size=8192):

        self.model.eval()
        device = self.device

        with torch.no_grad():
            pts = self.data_points[shape_idx]             # [N,3] in [-1,1]^3
            occ = self.data_occupancy[shape_idx]          # [N,] 0/1 (from SDF)
            gt_bin = (occ > 0.5).astype(np.int32)
            N = pts.shape[0]

            voxels = self.data_voxels[shape_idx]          # [64,64,64]
            t_vox = torch.from_numpy(voxels).float().unsqueeze(0).unsqueeze(0).to(device)
            z = self.model.encoder(t_vox)                 # [1,z_dim]

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

            pred_bin = (pred_full >= threshold).astype(np.int32)

            inter = np.logical_and(gt_bin == 1, pred_bin == 1).sum()
            union = np.logical_or(gt_bin == 1, pred_bin == 1).sum()
            iou = inter / (union + 1e-8)

            mse = ((pred_full - gt_bin.astype(np.float32)) ** 2).mean()

            print(f"[SDF IoU] shape {shape_idx}: IoU={iou:.4f}, "
                  f"MSE={mse:.6f}, pred_pos={pred_bin.sum()}, "
                  f"gt_pos={gt_bin.sum()}, N={N}")
            return iou, mse
