import os
import sys
import subprocess
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

import cv2
import json 
import numpy as np
from glob import glob
from PIL import Image
from pathlib import Path
from motion_analysis import analyze_trajectory, read_joint_infos_vlm
from argparse import ArgumentParser

# IMAGE_FOLDER = "rgba"
IMAGE_FOLDER = "images"


def prepare_data(data_dir, scene_name, n_canonical, input_path, with_canonical=False):
    data_path = f"{data_dir}/{scene_name}"
    img_paths = sorted(glob(f"{data_path}/{IMAGE_FOLDER}/*.png"))
    if not with_canonical:
        img_paths = img_paths[n_canonical:]
    imgs = [np.array(Image.open(img)) for img in img_paths]
    video = np.stack(imgs, 0)
    depth_paths = [img_path.replace(IMAGE_FOLDER, "depth") for img_path in img_paths]
    depths = [cv2.imread(depth_path, -1) / 1e3 for depth_path in depth_paths]
    depths = np.stack(depths, 0)

    if video.shape[3] == 4:
        alpha = video[:, :, :, 3] / 255.0
        video = (video[:, :, :, :3] * alpha[..., None]) + (1 - alpha[..., None]) * 255
        depths[alpha < 0.5] = 0
    video = video.astype(np.uint8)
    

    file = json.load(open(f"{data_path}/transforms.json", "r"))
    fx, fy = file['focal_x'], file['focal_y']
    cx, cy = file['cx'], file['cy']
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    intrinsics = np.stack([K] * len(imgs), 0)
    poses = np.array([f['transform_matrix'] for f in file['frames']])
    if not with_canonical:
        poses = poses[n_canonical:]
    # blender camera to opencv camera
    poses[:, :3, :3] = poses[:, :3, :3] @ np.diag([1, -1, -1])
    extrinsics = np.linalg.inv(poses)

    print(f"Data shape: video: {video.shape}, depths: {depths.shape}, intrinsics: {intrinsics.shape}, extrinsics: {extrinsics.shape}")
    data = {
        "video": video,
        "depths": depths,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
    }
    if with_canonical:
        np.savez(f"{data_path}/{scene_name}_full.npz", **data)
    else:
        np.savez(input_path, **data)


def prepare_data_realscan(data_dir, scene_name, n_canonical, input_path, with_canonical=False):
    data_path = f"{data_dir}/{scene_name}"
    vggt_results = np.load(f"{data_path}/data.npz", allow_pickle=True)
    depth = vggt_results['depths'] # [T, H, W]
    video = vggt_results['video']
    intrinsics = vggt_results['intrinsics']
    extrinsics = vggt_results['extrinsics']
    masks = vggt_results['masks'] # [T, K, H, W]
    depth[masks[:, 0] == 0] = 0
    if not with_canonical:
        video = video[n_canonical:]
        depth = depth[n_canonical:]
        intrinsics = intrinsics[n_canonical:]
        extrinsics = extrinsics[n_canonical:]
    data = {
        "video": video,
        "depths": depth,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
    }
    print(f"Data shape: video: {data['video'].shape}, depths: {data['depths'].shape}, intrinsics: {data['intrinsics'].shape}, extrinsics: {data['extrinsics'].shape}")
    np.savez(input_path, **data)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/videoartgs/realscan")
    parser.add_argument("--tapip3d_dir", type=str, default="./third_party/TAPIP3D")
    parser.add_argument("--video_name", type=str, default="t_1p")
    parser.add_argument("--reprocess", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    tapip3d_dir = Path(args.tapip3d_dir).resolve()
    if 'v2a' in args.data_dir:
        n_canonical = 24
    elif 'videoartgs' in args.data_dir:
        if 'realscan' in args.data_dir:
            n_canonical = 100
        else:   
            n_canonical = 150
    else:
        raise ValueError(f"Unknown data directory: {data_dir}")

    scenes = sorted(os.listdir(data_dir))
    scene_names = [os.path.basename(s) for s in scenes if os.path.isdir(os.path.join(data_dir, s))]
    n_query_frames = 4
    n_query_points = 8192
    results = []
    is_realscan = 'realscan' in args.data_dir
    for scene_name in scene_names:
        if args.video_name != "" and args.video_name != scene_name:
            continue
        print(f"Processing {scene_name}...")
        try:
            joint_infos = read_joint_infos_vlm(f'{data_dir}/{scene_name}/joint_infos_vlm.json')
            n_dyn_joints = len(joint_infos)
            nq = n_query_frames + n_dyn_joints // 2
        except:
            print(f"Skipping {scene_name} because it doesn't have joint infos.")
            continue
        input_path = f"{data_dir}/{scene_name}/{scene_name}.npz"
        output_dir = f"{data_dir}/{scene_name}"
        output_path = f"{output_dir}/{scene_name}.n{nq}.npz"
        if not os.path.exists(output_path) or args.reprocess:
            if is_realscan:
                prepare_data_realscan(data_dir, scene_name, n_canonical, input_path)
            else:
                prepare_data(data_dir, scene_name, n_canonical, input_path)
            cmd = f"cd {tapip3d_dir} && python inference.py --input_path {input_path} --n_query_frames {nq} --n_query_points {n_query_points} --output_dir {output_dir}"
            subprocess.run(cmd, shell=True)
        if os.path.exists(input_path):
            os.system(f"rm {input_path}")
        output = analyze_trajectory(
            scene_name, data_dir, nq, 
            use_vis_mask=False if is_realscan else True, # realscan has more noise in the vis mask
            visualize=False, 
            print_info=False,
            realscan=is_realscan
        )
        results.append([
            scene_name,
            output.max(0)[0],
            output.max(0)[1],
            output.mean(0)[0],
            output.mean(0)[1],
        ])
    print(results)

