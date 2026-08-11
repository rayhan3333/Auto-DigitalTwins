#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import torch
from scene import Scene, DeformModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_mask
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussian_renderer import GaussianModel
import numpy as np
from pytorch_lightning import seed_everything
import torch.nn.functional as F



def render_set(args, iteration, views, gaussians, deform, visualize=False):
    model_path = args.model_path
    save_dir = os.path.join(model_path, "ours_{}".format(iteration))
    gts_path = os.path.join(model_path, "ours_{}".format(iteration), "gt")
    makedirs(save_dir, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    gs_mask = deform.get_mask(gaussians.get_xyz)
    gt_rgbs = []
    gt_alphas = []
    masks = []
    mask_alphas = []
    for view in tqdm(views[0], desc=f"Rendering mask for static frames"):
        gt_image = torch.clamp(view.original_image.to("cuda"), 0.0, 1.0)
        gt_alpha = view.gt_alpha_mask
        gt_rgbs.append(gt_image)
        gt_alphas.append(gt_alpha)
        mask, mask_alpha = render_mask(view, gaussians, part_prob=gs_mask)
        masks.append(mask)
        mask_alphas.append(mask_alpha)
    for view in tqdm(views[1], desc=f"Rendering mask for dynamic frames"):
        gt_image = torch.clamp(view.original_image.to("cuda"), 0.0, 1.0)
        gt_alpha = view.gt_alpha_mask
        gt_rgbs.append(gt_image)
        gt_alphas.append(gt_alpha)
        d_values = deform.deform.one_transform(gaussians, view.fid, None, is_training=False)
        gs_mask = d_values['prob']
        d_xyz, d_rot = d_values['d_xyz'], d_values['d_rotation']
        mask, mask_alpha = render_mask(view, gaussians, d_xyz, d_rot, part_prob=gs_mask)
        masks.append(mask)
        mask_alphas.append(mask_alpha)
    mask_alphas = torch.stack(mask_alphas)
    masks = torch.stack(masks) # [N, K, H, W]
    mask_prob = F.softmax(masks, dim=1)
    masks = torch.argmax(mask_prob, dim=1) # [N, H, W]
    gt_alphas = torch.stack(gt_alphas).squeeze() # [N, H, W]
    # masks[gt_alphas > 0] += 1
    masks[mask_alphas > 0] += 1
    np.save(os.path.join(save_dir, "mask.npy"), masks.cpu().numpy())
    np.save(os.path.join(save_dir, "mask_prob.npy"), mask_prob.cpu().numpy())

    if visualize:
        vis_dir = os.path.join(save_dir, "vis_mask")
        makedirs(vis_dir, exist_ok=True)
        palette = torch.cat([torch.ones(1, 3), torch.rand(10, 3)], dim=0)
        for i in range(len(gt_rgbs)):
            gt = gt_rgbs[i]
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:06d}'.format(i) + ".png"))
            mask = masks[i].cpu()
            mask_vis = palette[mask].permute(2, 0, 1)
            torchvision.utils.save_image(mask_vis, os.path.join(vis_dir, '{0:06d}'.format(i) + ".png"))


def render_sets(args, dataset: ModelParams, iteration):
    with torch.no_grad():
        deform = DeformModel(dataset)
        loaded = deform.load_weights(dataset.model_path, iteration=iteration)
        if not loaded:
            raise ValueError(f"Failed to load weights from {dataset.model_path}")
        deform.update(int(iteration))

        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration)
        # gaussians.load_ply(f'./outputs/{args.dataset}/{args.subset}/{args.scene_name}/cano/point_cloud/iteration_{iteration}/point_cloud.ply')

        cam_traj = scene.getTrainCameras()
        render_set(args, iteration, cam_traj, gaussians, deform, visualize=args.visualize)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    
    parser.add_argument("--iteration", default=-1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument('--seed', type=int, default=0)

    args = get_combined_args(parser)
    # import sys
    # import json
    # args = parser.parse_args(sys.argv[1:])
    # args.source_path = f"{args.source_path}/{args.dataset}/{args.subset}/{args.scene_name}"
    # joint_infos = json.load(open(f'{args.source_path}/joint_infos.json', 'r'))
    # joint_types = [joint_info['joint_type'] for joint_info in joint_infos]
    # args.joint_types = joint_types
    # args.num_slots = len(joint_types)
    print("Rendering " + args.source_path + ' with '+ args.model_path)
    safe_state(args.quiet)
    seed_everything(args.seed)
    render_sets(args, model.extract(args), args.iteration)
