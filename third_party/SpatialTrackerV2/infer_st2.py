import os
import glob
import torch
import argparse
import numpy as np
from PIL import Image
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/videoartgs/realscan")
    parser.add_argument("--video_name", type=str, default="light")
    parser.add_argument("--reprocess", action="store_true", default=True)
    return parser.parse_args()


if __name__ == "__main__":
    model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
    model.eval()
    model = model.to("cuda")
    args = parse_args()
    scenes = os.listdir(args.data_dir)
    print("Found scenes: ", scenes)
    for scene in scenes:
        out_path = f"{args.data_dir}/{scene}/st2_result.npz"
        if args.video_name != "" and args.video_name != scene:
            continue
        if os.path.exists(out_path) and not args.reprocess:
            print(f"Results already exist for scene: {scene}")
            continue
        else:
            print(f"Processing scene: {scene}")
            img_path = f"{args.data_dir}/{scene}/images"
            img_paths = sorted(glob.glob(os.path.join(img_path, "*.png")))
            img_tensors = [torch.from_numpy(np.array(Image.open(img_path).convert("RGB"))).permute(2, 0, 1) for img_path in img_paths]
            video_tensor = torch.stack(img_tensors) # (N, C, H, W)
            video_tensor = preprocess_image(video_tensor) / 255

            
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    predictions = model(video_tensor[None].cuda())
            poses, intrinsic = predictions["poses_pred"], predictions["intrs"]
            depth_map, depth_conf = predictions["points_map"][..., 2], predictions["unc_metric"]
            
            # get the results of the current batch
            depth = depth_map.squeeze().cpu().numpy()
            extrs = torch.inverse(poses).squeeze().cpu().numpy()
            intrs = intrinsic.squeeze().cpu().numpy()
            depth_conf = depth_conf.squeeze().cpu().numpy()
            poses = poses.squeeze().cpu().numpy()
                    
            threshold = np.quantile(depth_conf, 0.1)
            valid = depth_conf >= threshold
            depth[~valid] = 0
            
            results = {
                    "video": video_tensor.squeeze().numpy(), # [T, 3, H, W]
                    "depths": depth, # [T, H, W]
                    "conf": depth_conf, # [T, H, W]
                    "extrinsics": extrs, # [T, 4, 4]
                    "intrinsics": intrs, # [T, 3, 3]
                    "poses": poses, # [T, 4, 4]
                }
            np.savez(out_path, **results)
            print(f"Results saved to {out_path}")
            print(f"video: {results['video'].shape}, depths: {results['depths'].shape}, extrinsics: {results['extrinsics'].shape}, intrinsics: {results['intrinsics'].shape}, poses: {results['poses'].shape}, conf: {results['conf'].shape}")