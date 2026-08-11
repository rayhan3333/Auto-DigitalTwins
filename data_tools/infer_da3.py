import os
import glob
import torch
import argparse
import numpy as np
from depth_anything_3.api import DepthAnything3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/videoartgs/realscan")
    parser.add_argument("--video_name", type=str, default="light")
    parser.add_argument("--reprocess", action="store_true", default=True)
    return parser.parse_args()


if __name__ == "__main__":
    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
    model = model.to("cuda")
    model.eval()
    args = parse_args()
    scenes = os.listdir(args.data_dir)
    print("Found scenes: ", scenes)
    for scene in scenes:
        out_path = f"{args.data_dir}/{scene}/da3_result.npz"
        if args.video_name != "" and args.video_name != scene:
            continue
        if os.path.exists(out_path) and not args.reprocess:
            print(f"Results already exist for scene: {scene}")
            continue
        else:
            img_path = f"{args.data_dir}/{scene}/images"
            images = sorted(glob.glob(os.path.join(img_path, "*.png")))
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    prediction = model.inference(images, process_res=518)
            extrinsics = np.zeros((len(images), 4, 4))
            extrinsics[:, :3] = prediction.extrinsics
            extrinsics[:, 3, 3] = 1
            depth_conf = prediction.conf
            threshold = np.quantile(depth_conf, 0.1)
            valid = depth_conf >= threshold
            depths = prediction.depth
            depths[~valid] = 0

            da3_results = {
                'video': prediction.processed_images.transpose(0, 3, 1, 2) / 255, # [T, 3, H, W]
                'depths': depths, # [T, H, W]
                'conf': prediction.conf, # [T, H, W]
                'extrinsics': extrinsics, # [T, 4, 4]
                'intrinsics': prediction.intrinsics, # [T, 3, 3]
                'poses': np.linalg.inv(extrinsics), # [T, 4, 4]
            }
            np.savez(out_path, **da3_results)
            print(f"Results saved to {out_path}")
            print(f"video: {da3_results['video'].shape}, depths: {da3_results['depths'].shape}, conf: {da3_results['conf'].shape}, extrinsics: {da3_results['extrinsics'].shape}, intrinsics: {da3_results['intrinsics'].shape}, poses: {da3_results['poses'].shape}")