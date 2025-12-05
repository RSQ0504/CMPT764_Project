import numpy as np
import pyvista as pv
from skimage import measure
import torch

from model import Encoder3D, BAE_NET_Wrapper

class Config:
    def __init__(self):
        self.learning_rate = 1e-3   # Adam 学习率
        self.beta1 = 0.9            # Adam β1
        self.epoch = 500  

npz_path = "./reference_models_processed/dog/voxel_and_sdf.npz"
data = np.load(npz_path)
voxels = data['voxels']

voxels = voxels.astype(np.float32)
x = torch.from_numpy(voxels).unsqueeze(0).unsqueeze(0)

encoder = Encoder3D(z_dim=128)
z = encoder(x)
print(z.shape)

bae_net = BAE_NET_Wrapper(data_dir='./reference_models_processed')
config = Config()
bae_net._train_unsupervised(config)
    