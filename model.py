# file: model_torch.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import h5py


class Encoder3D(nn.Module):
    def __init__(self, z_dim=128, ef_dim=32):
        super(Encoder3D, self).__init__()
        self.z_dim = z_dim
        self.ef_dim = ef_dim
        
        self.conv1 = nn.Conv3d(1, ef_dim, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.InstanceNorm3d(ef_dim)
        
        self.conv2 = nn.Conv3d(ef_dim, ef_dim*2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.InstanceNorm3d(ef_dim*2)
        
        self.conv3 = nn.Conv3d(ef_dim*2, ef_dim*4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.InstanceNorm3d(ef_dim*4)
        
        self.conv4 = nn.Conv3d(ef_dim*4, ef_dim*8, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.InstanceNorm3d(ef_dim*8)
        
        self.conv5 = nn.Conv3d(ef_dim*8, z_dim, kernel_size=4, stride=1, padding=0)
        
    def forward(self, x):
        # x: [batch, 1, 64, 64, 64]
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.02)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.02)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.02)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.02)
        x = torch.sigmoid(self.conv5(x))
        return x.view(-1, self.z_dim)

class Generator(nn.Module):
    def __init__(self, z_dim=128, gf_dim=256, gf_split=8, L1reg=False):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.gf_split = gf_split
        self.L1reg = L1reg
        
        self.fc1 = nn.Linear(z_dim + 3, 256)
        self.fc2 = nn.Linear(256, 256)
        
        if L1reg:
            self.fc3 = nn.Linear(256, 4)
        else:
            self.fc3 = nn.Linear(256, 4)
        
    def forward(self, points, z):
        # points: [batch, 3]
        # z: [batch, z_dim] or [1, z_dim] for inference
        batch_size = points.shape[0]
        
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if z.shape[0] == 1 and batch_size > 1:
            z = z.expand(batch_size, -1)
        
        # print(points.shape, z.shape)
        
        pointz = torch.cat([points, z], dim=1)
        
        h1 = F.leaky_relu(self.fc1(pointz), 0.02)
        h2 = F.leaky_relu(self.fc2(h1), 0.02)
        h3 = torch.sigmoid(self.fc3(h2))
        
        if self.training:
            h3_max = torch.max(h3, dim=1, keepdim=True)[0]
        else:
            h3_max = torch.max(h3, dim=1, keepdim=True)[0]
        
        return h3, h3_max

class BAE_Net(nn.Module):
    def __init__(self, z_dim=128, ef_dim=32, gf_dim=256, gf_split=4, 
                 L1reg=False):
        super(BAE_Net, self).__init__()
        self.z_dim = z_dim
        self.ef_dim = ef_dim
        self.gf_dim = gf_dim
        self.gf_split = gf_split
        self.L1reg = L1reg
        
        
        self.encoder = Encoder3D(z_dim=z_dim, ef_dim=ef_dim)
        self.generator = Generator(z_dim=z_dim, gf_dim=gf_dim, 
                                  gf_split=self.gf_split, L1reg=L1reg)
        
    def forward(self, voxels, points=None, mode='train'):
        z = self.encoder(voxels)
        
        if mode == 'train':
            if points is not None:
                _, G = self.generator(points, z)
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
                 checkpoint_dir='checkpoint', 
                 sample_dir='samples',
                 data_dir='./data'):
        

        self.L1reg = L1reg
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.data_dir = data_dir
        
        # Load data
        self._load_data()
        gf_split = 4
        
        self.model = BAE_Net(
            z_dim=128, ef_dim=32, gf_dim=256, 
            gf_split=gf_split,
            L1reg=L1reg
        )
        
        self.optimizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def _load_data(self):
        voxels_list = []
        points_list = []
        values_list = []
        
        shape = "dog"

        data_npz_name = f'{self.data_dir}/{shape}/voxel_and_sdf.npz'
        if not os.path.exists(data_npz_name):
            raise FileNotFoundError(f"Cannot load {data_npz_name}")
            
        data = np.load(data_npz_name)
        voxels_list.append(data['voxels'])        # shape (D,H,W)
        points_list.append(data['sdf_points'])    # shape (num_points,3)
        values_list.append(data['sdf_values'])    # shape (num_points,)
        
        # 转成 numpy array
        self.data_voxels = np.array(voxels_list)      # shape (num_shapes,D,H,W)
        self.data_points = np.array(points_list)      # shape (num_shapes,num_points,3)
        self.data_values = np.array(values_list)      # shape (num_shapes,num_points)
        self.data_occupancy = (self.data_values <= 0).astype(np.float32)  # shape (num_shapes,num_points)
        # print(self.data_voxels.shape, self.data_points.shape, self.data_values.shape)

            


    def _train_unsupervised(self, config):
        self.model.train()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.learning_rate, 
            betas=(config.beta1, 0.999)
        )
        
        num_shapes = len(self.data_points)
        indices = np.arange(num_shapes)
        print(indices)
        
        for epoch in range(config.epoch):
            np.random.shuffle(indices)
            total_loss = 0
            
            for idx in range(num_shapes):
                shape_idx = indices[idx]
                
                num_sample = 20000
                
                occupancy = self.data_occupancy[shape_idx]
                pos_idx = np.where(occupancy == 1)[0]
                neg_idx = np.where(occupancy == 0)[0]
                num_pos_sample = min(len(pos_idx), num_sample // 2)
                num_neg_sample = num_pos_sample
                sample_pos = np.random.choice(pos_idx, size=num_pos_sample, replace=False)
                sample_neg = np.random.choice(neg_idx, size=num_neg_sample, replace=False)
                sample_idx = np.concatenate([sample_pos, sample_neg])
                # print(sample_idx.shape)
                # quit()
                np.random.shuffle(sample_idx)

                
                
                batch_voxels = torch.FloatTensor(
                    np.array(self.data_voxels[shape_idx:shape_idx+1])
                ).to(self.device).unsqueeze(-1)
                print(batch_voxels.shape)
                batch_points = torch.FloatTensor(
                    np.array([self.data_points[shape_idx][sample_idx]]).squeeze(0)
                ).to(self.device)
                # batch_values = torch.FloatTensor(
                #     np.array([self.data_values[shape_idx][sample_idx]]).squeeze(0)
                # ).to(self.device)
                batch_values = torch.FloatTensor(
                    np.array([self.data_occupancy[shape_idx][sample_idx]]).squeeze(0)
                ).to(self.device)
                print(batch_voxels.shape,batch_points.shape, batch_values.shape)
                # Forward pass
                pred_occupancy = self.model(batch_voxels, batch_points, mode='train')
                
                # Calculate loss
                loss = F.mse_loss(pred_occupancy, batch_values)
                
                if self.L1reg:
                    # Add L1 regularization
                    l1_reg = 0
                    for param in self.model.generator.fc3.parameters():
                        l1_reg += torch.norm(param, 1)
                    loss += 1e-6 * l1_reg
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                print(f"Epoch [{epoch}/{config.epoch}] Batch [{idx}/{num_shapes}] Loss: {loss.item():.6f}")
            
            avg_loss = total_loss / num_shapes
            print(f"Epoch [{epoch}/{config.epoch}] Average Loss: {avg_loss:.6f}")
            
            # Save checkpoint
            if epoch % 10000 == 0:
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
        print(f"Checkpoint saved at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    
    def test_segmentation(self, test_points, test_voxels, use_postprocessing=True):
        self.model.eval()
        
        with torch.no_grad():
            batch_voxels = torch.FloatTensor(test_voxels).to(self.device)
            batch_points = torch.FloatTensor(test_points).to(self.device)
            
            # Get latent code
            z = self.model.encoder(batch_voxels)
            
            # Get segmentation
            branch_pred, pred = self.model.generator(batch_points, z)

            
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
    
    def generate_mesh(self, voxels, threshold=0.4):
        try:
            import mcubes
        except ImportError:
            print("Please install PyMCubes: pip install PyMCubes")
            return None, None
        
        self.model.eval()
        
        with torch.no_grad():
            batch_voxels = torch.FloatTensor(voxels).to(self.device)
            z = self.model.encoder(batch_voxels)
            
            dim = 64
            coords = np.linspace(-0.5, 0.5, dim)
            grid_x, grid_y, grid_z = np.meshgrid(coords, coords, coords, indexing='ij')
            grid_points = np.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], axis=1)
            
            batch_size = 8192
            predictions = []
            
            for i in range(0, len(grid_points), batch_size):
                batch = grid_points[i:i+batch_size]
                batch_tensor = torch.FloatTensor(batch).to(self.device)
                branch_pred, _ = self.model.generator(batch_tensor, z)
                predictions.append(branch_pred.cpu().numpy())
            
            predictions = np.concatenate(predictions, axis=0)
            
            predictions = predictions.reshape(dim, dim, dim, -1)
            
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
            
            return all_vertices, all_triangles