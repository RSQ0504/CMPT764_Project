import os
import h5py
import numpy as np

def process_h5_file(h5_path, output_dir, res_list=['64'], max_num=100):
    print(f"Processing {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        if 'voxels' not in f:
            print(f"[WARN] 'voxels' not in {h5_path}, skip.")
            return
        
        vox_all = f['voxels']  # shape maybe (batch, D, H, W, 1) or (D,H,W,1) or similar
        vox_data = vox_all[()]

        if vox_data.ndim == 5 and vox_data.shape[-1] == 1:
            vox_data = vox_data[..., 0]
        if vox_data.ndim == 4:
            num_objs = vox_data.shape[0]
        elif vox_data.ndim == 3:
            num_objs = 1
            vox_data = vox_data[np.newaxis, ...]
        else:
            raise ValueError(f"Unexpected voxels shape {vox_data.shape} in {h5_path}")

        for res in res_list:
            pkey = f'points_{res}'
            vkey = f'values_{res}'
            if pkey not in f or vkey not in f:
                continue
            pts_all = f[pkey][()]
            vals_all = f[vkey][()]

            # 确认 pts_all / vals_all 的 batch size
            if pts_all.ndim != 3 or vals_all.ndim != 3:
                print(f"[WARN] Unexpected dims for {pkey} or {vkey} in {h5_path}")
                continue
            if pts_all.shape[0] != num_objs or vals_all.shape[0] != num_objs:
                print(f"[WARN] batch size mismatch voxels vs points/values in {h5_path}")
                continue
            num_objs = min(max_num, num_objs)
            # print(max_num)
            base = os.path.splitext(os.path.basename(h5_path))[0]
            for i in range(num_objs):
                vox_i = vox_data[i].astype(np.uint8)

                pts_i = pts_all[i].astype(np.float32)
                pts_i = (pts_i + 0.5) / float(res) * 2.0 - 1.0

                vals_i = vals_all[i, :, 0].astype(np.uint8)  
                vals_i = vals_i.astype(np.float32)       
                out_fname = f"{base}_res{res}_{i}.npz"
                out_path = os.path.join(output_dir, out_fname)
                np.savez_compressed(out_path,
                                    voxels=vox_i,
                                    sdf_points=pts_i,
                                    occu_values=vals_i)
                print("Saved:", out_path)


if __name__ == '__main__':
    for item in os.listdir("/home/david/Documents/1TB/Courses/CMPT764/BAE-NET/data"):
        i,name = item.split('_')
        input_dir = f'/home/david/Documents/1TB/Courses/CMPT764/BAE-NET/data/{item}/{i}_vox.hdf5' 
        output_dir = f'./train/{name}'
        os.makedirs(output_dir, exist_ok=True)
        process_h5_file(input_dir, output_dir, res_list=['64'])