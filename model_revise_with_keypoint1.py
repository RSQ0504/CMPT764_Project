# file: model_revise.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


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
    def __init__(self, z_dim=128, gf_dim=256, gf_split=4, L1reg=False, l2_extra_size=0, l3_extra_size=0):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.gf_split = gf_split
        self.L1reg = L1reg


        # Layer 1: z_dim+3 -> gf_dim*4 (e.g., 1024)
        self.fc1 = nn.Linear(z_dim + 3, gf_dim * 4)

        # Layer 2: gf_dim*4 -> gf_dim (e.g., 256)
        self.fc2 = nn.Linear(gf_dim * 4 + l2_extra_size, gf_dim)

        # Layer 3: gf_dim -> gf_split (e.g., 4)
        self.fc3 = nn.Linear(gf_dim + l3_extra_size, gf_split)

    def forward(self, points, z, j_points=None, e_points=None):
        # points: [batch, 3]
        # z: [batch, z_dim] or [1, z_dim]
        batch_size = points.shape[0]

        if z.dim() == 1:
            z = z.unsqueeze(0)
        if z.shape[0] == 1 and batch_size > 1:
            z = z.expand(batch_size, -1)

        pointz = torch.cat([points, z], dim=1)

        h1 = F.leaky_relu(self.fc1(pointz), 0.02)
        j_points = j_points.unsqueeze(0).expand(batch_size, -1) 
        h2 = F.leaky_relu(self.fc2(torch.cat([h1, j_points], dim=1)), 0.02)
        e_points = e_points.unsqueeze(0).expand(batch_size, -1)
        h3 = torch.sigmoid(self.fc3(torch.cat([h2, e_points], dim=1)))

        # Max pooling to get occupancy
        h3_max = torch.max(h3, dim=1, keepdim=True)[0]

        return h3, h3_max


class BAE_Net(nn.Module):
    def __init__(self, z_dim=128, ef_dim=32, gf_dim=256, gf_split=4, L1reg=False, l2_extra_size=0, l3_extra_size=0):
        super(BAE_Net, self).__init__()
        self.z_dim = z_dim
        self.ef_dim = ef_dim
        self.gf_dim = gf_dim
        self.gf_split = gf_split
        self.L1reg = L1reg

        self.encoder = Encoder3D(z_dim=z_dim, ef_dim=ef_dim)
        self.generator = Generator(z_dim=z_dim, gf_dim=gf_dim,
                                   gf_split=self.gf_split, L1reg=L1reg, l2_extra_size=l2_extra_size, l3_extra_size=l3_extra_size)

        self.apply(weights_init)

    def forward(self, voxels, points=None, j_points=None, e_points=None, mode='train'):
        z = self.encoder(voxels)

        if mode == 'train':
            if points is not None:
                _, G = self.generator(points, z, j_points, e_points)
                return G
            else:
                raise ValueError("In training mode, points must be provided")
        else:  # inference
            if points is not None:
                branch_pred, occupancy = self.generator(points, z)
                return branch_pred, occupancy
            else:
                raise ValueError("In inference mode, points must be provided")


class BAE_NET_Wrapper:
    def __init__(self,
                 L1reg=True,
                 checkpoint_dir='checkpoint/model_revised_keypoint',
                 sample_dir='samples',
                 data_dir='./data'):

        self.L1reg = L1reg
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.data_dir = data_dir

        # Load data
        self._load_data()

        # Default split set to 4 as requested
        gf_split = 4
        # print(self.l2_extra_size, self.l3_extra_size)
        self.model = BAE_Net(
            z_dim=128, ef_dim=32, gf_dim=256,
            gf_split=gf_split,
            L1reg=L1reg, l2_extra_size=self.l2_extra_size, l3_extra_size=self.l3_extra_size
        )

        self.optimizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def _load_data(self):
        voxels_list = []
        points_list = []
        values_list = []
        occupancy_list = []
        j_list = []
        e_list = []

        print(f"Loading data from {self.data_dir}...")
        for npz in os.listdir(self.data_dir):
            if not npz.endswith('.npz'):
                continue
            data_npz_name = f'{self.data_dir}/{npz}'
            if not os.path.exists(data_npz_name):
                raise FileNotFoundError(f"Cannot load {data_npz_name}")
            data = np.load(data_npz_name)
            
            # for k in data.keys():
            #     print(k, data[k].shape)
            # quit()    
            voxels_list.append(data['voxels'])  # shape (D,H,W)
            points_list.append(data['sdf_points'])  # shape (num_points,3)
            if 'sdf_values' in data:
                values_list.append(data['sdf_values'])  # shape (num_points,)
            if "occu_values" in data:
                occupancy_list.append(data['occu_values'])  # shape (num_points,)
            if "junction_points" in data:
                j_list.append(data['junction_points'])
            if "endpoint_points" in data:
                e_list.append(data['endpoint_points'])

        self.data_voxels = np.array(voxels_list)  # shape (num_shapes,D,H,W)
        self.data_points = np.array(points_list)  # shape (num_shapes,num_points,3)


        if len(self.data_points) > 0:
            p_min = self.data_points.min()
            p_max = self.data_points.max()
            print(f"\n[Data Check] Original Range: Min={p_min:.4f}, Max={p_max:.4f}")

            if p_min < -0.6 or p_max > 0.6:
                print("ACTION: Detected range ~[-1, 1]. Rescaling by 0.5 to match target range [-0.5, 0.5]...")
                self.data_points /= 2.0

                new_min, new_max = self.data_points.min(), self.data_points.max()
                print(f"        New Range: Min={new_min:.4f}, Max={new_max:.4f}")
            else:
                print("PASSED: Data range is already compatible with [-0.5, 0.5].")

        if len(values_list) > 0:
            self.data_values = np.array(values_list)
            self.data_occupancy = (self.data_values <= 0).astype(np.float32)
        if len(occupancy_list) > 0:
            self.data_values = np.array(occupancy_list)
            self.data_occupancy = np.array(occupancy_list)
        if len(j_list) > 0:
            j_list = np.array(j_list)
            j_list = j_list / 64.0 - 0.5
            self.data_junctions = j_list.reshape(-1, 1).squeeze(-1)
        if len(e_list) > 0:
            e_list = np.array(e_list)
            e_list = e_list / 64.0 - 0.5
            self.data_endpoints = e_list.reshape(-1, 1).squeeze(-1)
        self.l2_extra_size = self.data_junctions.shape[0] if len(j_list) > 0 else 0
        self.l3_extra_size = self.data_endpoints.shape[0] if len(e_list) > 0 else 0
        print(f"Loaded {len(voxels_list)} shapes.\n")

    def _train_unsupervised(self, config):
        self.model.train()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, 0.999)
        )

        num_shapes = len(self.data_points)
        indices = np.arange(num_shapes)
        epoch_bar = tqdm(range(config.epoch), desc="Training", unit="epoch")

        for epoch in epoch_bar:
            np.random.shuffle(indices)
            total_loss = 0

            for idx in range(num_shapes):
                shape_idx = indices[idx]

                num_sample = 20000
                occupancy = self.data_occupancy[shape_idx]
                pos_idx = np.where(occupancy == 1)[0]
                neg_idx = np.where(occupancy == 0)[0]

                num_pos_sample = min(len(pos_idx), num_sample // 2)
                num_neg_sample = min(len(neg_idx), num_sample // 2)

                current_sample_num = min(num_pos_sample, num_neg_sample)

                sample_pos = np.random.choice(pos_idx, size=current_sample_num, replace=False)
                sample_neg = np.random.choice(neg_idx, size=current_sample_num, replace=False)

                sample_idx = np.concatenate([sample_pos, sample_neg])
                np.random.shuffle(sample_idx)

                # data_voxels[shape_idx] -> (D,H,W) -> (1,1,D,H,W)
                batch_voxels = torch.from_numpy(self.data_voxels[shape_idx]).float().unsqueeze(0).unsqueeze(0).to(
                    self.device)

                batch_points = torch.from_numpy(self.data_points[shape_idx][sample_idx]).float().to(self.device)
                batch_values = torch.from_numpy(self.data_occupancy[shape_idx][sample_idx]).float().unsqueeze(1).to(
                    self.device)  # [N, 1]
                
                j_points = torch.from_numpy(self.data_junctions).float().to(self.device)
                e_points = torch.from_numpy(self.data_endpoints).float().to(self.device)
                # Forward pass
                # pred_occupancy: [N, 1]
                pred_occupancy = self.model(batch_voxels, batch_points, j_points=j_points, e_points=e_points, mode='train')

                # Calculate loss
                loss = F.mse_loss(pred_occupancy, batch_values)

                if self.L1reg:
                    l1_reg = torch.norm(self.model.generator.fc3.weight, 1)
                    loss += 1e-6 * l1_reg

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                loss_val = loss.item()
                last_loss = loss_val

            avg_loss = total_loss / num_shapes
            epoch_bar.set_postfix(
                loss=f"{last_loss:.6f}",
                avg_loss=f"{avg_loss:.6f}"
            )

            if epoch % 10000 == 0 and epoch > 0:
                self.save_checkpoint(epoch, avg_loss)

    def save_checkpoint(self, epoch, loss):
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': {
                'gf_split': self.model.gf_split,
            }
        }

        torch.save(
            checkpoint,
            os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        )

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']

    def test_segmentation(self, test_points, test_voxels, test_junctions, test_endpoints, use_postprocessing=True):
        self.model.eval()

        with torch.no_grad():
            # test_voxels: expecting (1, D, H, W) or (D,H,W) -> convert to (1, 1, D, H, W)
            if test_voxels.ndim == 3:
                test_voxels = test_voxels[np.newaxis, np.newaxis, ...]
            elif test_voxels.ndim == 4:
                test_voxels = test_voxels[np.newaxis, ...]

            batch_voxels = torch.FloatTensor(test_voxels).to(self.device)
            batch_points = torch.FloatTensor(test_points).to(self.device)
            batch_junctions = torch.FloatTensor(test_junctions).to(self.device)
            batch_endpoints = torch.FloatTensor(test_endpoints).to(self.device)

            z = self.model.encoder(batch_voxels)
            branch_pred, pred = self.model.generator(batch_points, z, j_points=batch_junctions, e_points=batch_endpoints)

            if use_postprocessing:
                branch_pred = self._postprocess_segmentation(
                    branch_pred.cpu().numpy(),
                    test_points
                )
            else:
                branch_pred = branch_pred.cpu().numpy()

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

    def generate_mesh(self, voxels, junctions, endpoints, threshold=0.5):
        try:
            import mcubes
        except ImportError:
            print("Please install PyMCubes: pip install PyMCubes")
            return None, None

        self.model.eval()

        with torch.no_grad():
            # Handle voxel shape input: (D,H,W) -> (1,1,D,H,W)
            if voxels.ndim == 3:
                input_voxels = voxels[np.newaxis, np.newaxis, ...]
            else:
                input_voxels = voxels

            batch_voxels = torch.FloatTensor(input_voxels).to(self.device)
            batch_junctions = torch.FloatTensor(junctions).to(self.device)
            batch_endpoints = torch.FloatTensor(endpoints).to(self.device)
            z = self.model.encoder(batch_voxels)

            dim = 64
            coords = np.linspace(-0.5, 0.5, dim)
            grid_x, grid_y, grid_z = np.meshgrid(coords, coords, coords, indexing='ij')
            grid_points = np.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], axis=1)

            batch_size = 8192
            predictions = []
            total_mesh = []

            for i in range(0, len(grid_points), batch_size):
                batch = grid_points[i:i + batch_size]
                batch_tensor = torch.FloatTensor(batch).to(self.device)
                branch_pred, pred = self.model.generator(batch_tensor, z, j_points=batch_junctions, e_points=batch_endpoints)
                predictions.append(branch_pred.cpu().numpy())
                total_mesh.append(pred.cpu().numpy())

            predictions = np.concatenate(predictions, axis=0)
            predictions = predictions.reshape(dim, dim, dim, -1)

            total_mesh = np.concatenate(total_mesh, axis=0)
            total_mesh = total_mesh.reshape(dim, dim, dim)

            all_vertices = []
            all_triangles = []

            for branch in range(predictions.shape[-1]):
                vertices, triangles = mcubes.marching_cubes(
                    predictions[:, :, :, branch],
                    threshold
                )
                if len(vertices) > 0:
                    vertices = vertices / dim - 0.5
                    all_vertices.append(vertices)
                    all_triangles.append(triangles)

            total_vertices, total_triangles = mcubes.marching_cubes(
                total_mesh,
                threshold
            )
            if len(total_vertices) > 0:
                total_vertices = total_vertices / dim - 0.5

            return all_vertices, all_triangles, total_vertices, total_triangles