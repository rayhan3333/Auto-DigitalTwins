import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
import argparse
import numpy as np
from data_tools.process_utils import *
from glob import glob
import torch
from utils.other_utils import vis_depth
from sklearn.decomposition import PCA
from utils.mesh_utils import MeshExtractor


def pca_align(poses, xyz):
    center = xyz.mean(0)
    xyz = xyz - center
    poses[:, :3, 3] -= center
    pca = PCA(n_components=3)
    pca.fit(xyz)
    axes = pca.components_
    if np.linalg.det(axes) < 0:
        # if the determinant is negative, flip the third axis
        axes[2] = -axes[2]
    T1 = np.eye(4)
    T1[:3, :3] = axes
    poses = T1 @ poses
    xyz = xyz @ axes.T
    range_xyz = np.max(xyz, axis=0) - np.min(xyz, axis=0)
    range_xyz = np.max(range_xyz)
    return poses, xyz


def process_mask(mask_tensor, mode="crop", target_size=518):
    """
    Preprocess image tensor(s) to target size with crop or pad mode.
    Args:
        mask_tensor (torch.Tensor): tensor of shape (C, H, W) or (T, C, H, W)
        mode (str): 'crop' or 'pad'
        target_size (int): Target size for width/height
    Returns:
        torch.Tensor: Preprocessed image tensor(s), same batch dim as input
    """
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")
    if mask_tensor.dim() == 3:
        tensors = [mask_tensor]
        squeeze = True
    elif mask_tensor.dim() == 4:
        tensors = list(mask_tensor)
        squeeze = False
    else:
        raise ValueError("Input tensor must be (C, H, W) or (T, C, H, W)")
    processed = []
    for mask in tensors:
        C, H, W = mask.shape
        if mode == "pad":
            if W >= H:
                new_W = target_size
                new_H = round(H * (new_W / W) / 14) * 14
            else:
                new_H = target_size
                new_W = round(W * (new_H / H) / 14) * 14
            out = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(new_H, new_W), mode="nearest").squeeze(0)
            h_padding = target_size - new_H
            w_padding = target_size - new_W
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            if h_padding > 0 or w_padding > 0:
                out = torch.nn.functional.pad(
                    out, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
        else:  # crop
            new_W = target_size
            new_H = round(H * (new_W / W) / 14) * 14
            out = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(new_H, new_W), mode="nearest").squeeze(0)
            if new_H > target_size:
                start_y = (new_H - target_size) // 2
                out = out[:, start_y : start_y + target_size, :]
        processed.append(out)
    result = torch.stack(processed)
    if squeeze:
        return result[0]
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/videoartgs/realscan")
    parser.add_argument("--video_name", type=str, default="light")
    parser.add_argument("--model", type=str, default="da3")
    parser.add_argument("--reprocess", action="store_true", default=True)
    parser.add_argument("--visualize", action="store_true", default=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    data_path = args.data_dir
    scenes = sorted(os.listdir(data_path))
    scene_names = [os.path.basename(s) for s in scenes if os.path.isdir(os.path.join(data_path, s))]
    IMAGE_DIR = 'images'
    DEPTH_DIR = 'depth'
    num_cano = 100
    visualize_depth = True
    for scene_name in scene_names:
        out_path = f'{data_path}/{scene_name}/data.npz'
        if args.video_name != "" and args.video_name != scene_name:
            continue
        if os.path.exists(out_path) and not args.reprocess:
            print(f"Results already exist for scene: {scene_name}")
            continue
        print(f"Processing {scene_name}...")
        scene_path = f'{data_path}/{scene_name}'
        assert os.path.exists(f'{scene_path}/masks'), f"masks not found for {scene_name}"

        results = np.load(f'{scene_path}/{args.model}_result.npz', allow_pickle=True)
        video = results['video'] # [T, 3, H, W]
        depth = results['depths']
        intrinsics = results['intrinsics']
        extrinsics = results['extrinsics']
        poses = results['poses']
        T, H, W = depth.shape
        mask_paths = sorted(glob(f"{scene_path}/masks/*.npy"))
        masks = [np.load(mask_path) for mask_path in mask_paths]
        masks = np.stack(masks, 0).squeeze() # [T, K, H, W]
        masks = torch.from_numpy(masks).float()
        masks = process_mask(masks)

        depth_conf = results['conf']
        threshold = np.quantile(depth_conf, 0.1)
        valid = depth_conf >= threshold
        depth[~valid] = 0

        data = {
            "video": video,
            "intrinsics": intrinsics,
            "masks": masks,
            "depths": depth,
        }

        if visualize_depth and not os.path.exists(f'{scene_path}/vis_depth'):
            os.makedirs(f'{scene_path}/vis_depth', exist_ok=True)
            for i in range(len(depth)):
                vis_depth(depth[i], f'{scene_path}/vis_depth/{i:06d}.png')

        video = video.transpose(0, 2, 3, 1) # [T, H, W, 3]
        depth[masks[:, 0] < 0.5] = 0
        mesh_extractor = MeshExtractor(poses[:num_cano], intrinsics[:num_cano], W, H, video[:num_cano], depth[:num_cano])
        mesh = mesh_extractor.extract_mesh()
        o3d.io.write_triangle_mesh(f'{scene_path}/mesh.ply', mesh)
        xyz = np.asarray(mesh.vertices)
        poses, xyz = pca_align(poses, xyz)
        data["poses"] = poses
        data["extrinsics"] = np.linalg.inv(poses)
        np.savez(f'{scene_path}/data.npz', **data)
        mesh.vertices = o3d.utility.Vector3dVector(xyz)
        o3d.io.write_triangle_mesh(f'{scene_path}/mesh.ply', mesh)
        gen_pcd_cano(f'{scene_path}/point_cloud.ply', intrinsics[:num_cano], poses[:num_cano], video[:num_cano], depth[:num_cano], reprocess=True, cluster=True, visualize=args.visualize, eps=0.04, coord_trans=False)