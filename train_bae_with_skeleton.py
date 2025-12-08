
import os
import argparse
import numpy as np
import torch

# load wrapper
from model_revise_with_keypoints import BAE_NET_Wrapper  # make sure this file is in the same folder

"""
Train BAE-Net (skeleton-aware version) on npz files that contain voxels,
point samples, occupancy, and skeleton keypoints (junctions + endpoints).
The model and wrapper are defined in model_revise_with_keypoints.py.
"""

"""
Example usage:
python train_bae_with_skeleton_keypoints.py \
    --data_dir ./train_with_skeleton/couch \
    --checkpoint_dir ./checkpoint/bae_skeleton_couch \
    --epochs 200000 \
    --lr 1e-3
"""

class Config:
    """
    Simple config object matching what BAE_NET_Wrapper._train_unsupervised expects.
    """
    def __init__(self, learning_rate=1e-3, beta1=0.9, epoch=200000, resume_epoch=0):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.epoch = epoch
        self.resume_epoch = resume_epoch


def inspect_one_npz(data_dir):
    """
    look at one npz file and print which skeleton-related keys exist.
    Confirm valid skeleton-augmented data.
    """
    npz_files = [f for f in os.listdir(data_dir) if f.endswith(".npz")]
    if not npz_files:
        print(f"[WARN] No npz files found in {data_dir}")
        return

    sample_path = os.path.join(data_dir, npz_files[0])
    with np.load(sample_path) as data:
        keys = list(data.files)
    print(f"[Info] Example npz file: {sample_path}")
    print(f"[Info] Available keys: {keys}")

    for k in ["voxels", "sdf_points", "occu_values", "sdf_values",
              "skeleton_segments", "skeleton_points",
              "junction_points", "endpoint_points"]:
        print(f"    {'✓' if k in keys else '✗'} {k}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train skeleton-aware BAE-Net on skeleton-augmented npz dataset."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./train_with_skeleton/couch",
        help="Directory that contains npz files for one category "
             "(e.g., ./train_with_skeleton/couch).",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoint/bae_with_skeleton",
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200000,
        help="Number of training epochs/iterations.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for Adam optimizer.",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="Beta1 parameter for Adam optimizer.",
    )
    parser.add_argument(
        "--no_l1reg",
        action="store_true",
        help="Disable L1 regularization on the generator's last layer.",
    )
    parser.add_argument(
        "--resume_epoch",
        type=int,
        default=0,
        help="If >0, try resuming training from this epoch checkpoint (if implemented).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # error check on data directory
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"data_dir does not exist: {args.data_dir}")

    print("========== Skeleton-aware BAE-Net Training ==========")
    print(f"[Config] data_dir       = {args.data_dir}")
    print(f"[Config] checkpoint_dir = {args.checkpoint_dir}")
    print(f"[Config] epochs         = {args.epochs}")
    print(f"[Config] lr             = {args.lr}")
    print(f"[Config] beta1          = {args.beta1}")
    print(f"[Config] L1 reg         = {not args.no_l1reg}")
    print(f"[Config] resume_epoch   = {args.resume_epoch}")
    print("=====================================================")

    # confirm skeleton keys exist
    inspect_one_npz(args.data_dir)

    # device information
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    # build config for wrapper
    config = Config(
        learning_rate=args.lr,
        beta1=args.beta1,
        epoch=args.epochs,
        resume_epoch=args.resume_epoch,
    )

    # load voxels, points, occupancy and skeleton keypoints from args.data_dir
    model_wrapper = BAE_NET_Wrapper(
        L1reg=not args.no_l1reg,
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir,
    )

    print("[Info] Data loaded. Starting training...")
    model_wrapper._train_unsupervised(config)
    print("[Info] Training finished.")


if __name__ == "__main__":
    main()
