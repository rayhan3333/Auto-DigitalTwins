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

import os
import cv2
import glob
import torch
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from utils.metrics import *


def load_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1)
    return img # [3, H, W]


def evaluate(args, name, iteration, eval_app=False):
    model_path = args.model_path
    
    save_dir = os.path.join(model_path, name, f"ours_{iteration}")
    mesh_path = os.path.join(save_dir, "meshes")
    gt_path = os.path.join(args.source_path, "gt")

    pred_joint_list = load_joint_infos(os.path.join(save_dir, 'joint_info.json'))
    gt_joint_list = read_gt(f'{args.source_path}/gt/mobility_v2.json')
    num_d_joints = len(gt_joint_list)

    s, d_list, w, perm = eval_CD(gt_path, mesh_path, num_d_joints, n_trials=3)

    results_list, _ = eval_axis_and_state_all(pred_joint_list, gt_joint_list, perm)
    output = pd.DataFrame()

    output['CD_static'] = [s]
    output['CD_whole'] = [w]
    output['CD_dynamic'] = [np.mean(np.array(d_list)).item()]
    output['angle'] = [np.mean(np.array(results_list)[:, 0]).item()]
    output['distance'] = [np.mean(np.array(results_list)[:, 1]).item()]

    if os.path.exists(f'{args.source_path}/gt/gt_joint_value.npy') and 'v2a' in args.source_path: # v2a has gt joint value
        gt_joint_value = np.load(f'{args.source_path}/gt/gt_joint_value.npy')
        pred_joint_value = np.load(os.path.join(save_dir, "joint_value.npy"))[-1].squeeze()
        gt_joint_value = gt_joint_value - gt_joint_value[0:1]
        pred_joint_value = pred_joint_value - pred_joint_value[0:1]
        if gt_joint_list[0]['joint_type'] == 'r':
            # normalize to (-pi, pi)
            gt_joint_value = np.mod(gt_joint_value + np.pi, 2 * np.pi) - np.pi
            pred_joint_value = np.mod(pred_joint_value + np.pi, 2 * np.pi) - np.pi
        sign = np.sign(gt_joint_value * pred_joint_value)
        pred_joint_value = pred_joint_value * sign
        if gt_joint_list[0]['joint_type'] == 'p':
            joint_state_error = 100 * np.mean(np.abs(pred_joint_value - gt_joint_value))
        else:
            joint_state_error = np.mean(np.abs(np.rad2deg(pred_joint_value) - np.rad2deg(gt_joint_value)))

        output['joint_state_error'] = [joint_state_error]

    for i, d in enumerate(d_list):
        output[f'CD_dynamic_{i}'] = [d]
        output[f'angle_{i}'] = [results_list[i][0]]
        output[f'distance_{i}'] = [results_list[i][1]]
    
    PSNR, SSIM, LPIPS = -1, -1, -1
    if eval_app:
        render_path = os.path.join(save_dir, "renders", "-1")
        gts_path = os.path.join(save_dir, "gt", "-1")
        img_paths = sorted(glob.glob(os.path.join(render_path, "*.png")))
        gt_img_paths = sorted(glob.glob(os.path.join(gts_path, "*.png")))
        PSNR, SSIM, LPIPS = 0, 0, 0
        for img_path, gt_img_path in zip(img_paths, gt_img_paths):
            rgb = load_img(img_path)
            gt_rgb = load_img(gt_img_path)
            PSNR += psnr(rgb[None], gt_rgb[None])
            SSIM += ssim_func(rgb[None], gt_rgb[None])
            LPIPS += lpips(rgb[None], gt_rgb[None])
        n = len(img_paths)
        PSNR, SSIM, LPIPS = (PSNR / n).item(), (SSIM / n).item(), (LPIPS / n).item()

    output['PSNR'] = [PSNR]
    output['SSIM'] = [SSIM]
    output['LPIPS'] = [LPIPS]
    
    output = output.transpose()
    output.index.name = 'Metric'
    print(output)
    output.to_csv(os.path.join(save_dir, "result.csv"), index=True, header=['Value'])


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1)
    parser.add_argument("--eval_app", action="store_true")

    args = get_combined_args(parser)
    print("Evaluating " + args.source_path + ' with '+ args.model_path)
    seed_everything(0)
    evaluate(args, 'train', args.iteration, args.eval_app)
