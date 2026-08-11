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
import open3d as o3d
from os import makedirs
from gaussian_renderer import render_gsplat
import torchvision
from utils.general_utils import safe_state, vis_depth
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussian_renderer import GaussianModel
from utils.mesh_utils import GaussianExtractor
from utils.metrics import *
from utils.geo_utils import find_biggest_cluster


def get_rotation_axis_angle(k, theta):
    '''
    Rodrigues' rotation formula
    args:
    * k: direction unit vector of the axis to rotate about
    * theta: the (radian) angle to rotate with
    return:
    * 3x3 rotation matrix
    '''
    if np.linalg.norm(k) == 0.:
        return np.eye(3)
    k = k / np.linalg.norm(k)
    kx, ky, kz = k[0], k[1], k[2]
    cos, sin = np.cos(theta), np.sin(theta)
    R = np.zeros((3, 3))
    R[0, 0] = cos + (kx**2) * (1 - cos)
    R[0, 1] = kx * ky * (1 - cos) - kz * sin
    R[0, 2] = kx * kz * (1 - cos) + ky * sin
    R[1, 0] = kx * ky * (1 - cos) + kz * sin
    R[1, 1] = cos + (ky**2) * (1 - cos)
    R[1, 2] = ky * kz * (1 - cos) - kx * sin
    R[2, 0] = kx * kz * (1 - cos) - ky * sin
    R[2, 1] = ky * kz * (1 - cos) + kx * sin
    R[2, 2] = cos + (kz**2) * (1 - cos)
    return R


def save_axis_mesh(k, center, filepath):
    '''support rotate only for now'''
    axis = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.01, cone_radius=0.02, cylinder_height=0.7, cone_height=0.04)
    arrow = np.array([0., 0., 1.], dtype=np.float32)
    n = np.cross(arrow, k)
    rad = np.arccos(np.dot(arrow, k))
    R_arrow = get_rotation_axis_angle(n, rad)
    axis.rotate(R_arrow, center=(0, 0, 0))
    axis.translate(center[:3])
    o3d.io.write_triangle_mesh(filepath, axis)


def vis_joint(mesh_path, pred_joint_list, gt_joint_list=None):
    for i in range(len(pred_joint_list)):
        joint_info = pred_joint_list[i]
        pos = joint_info['origin']
        center = joint_info['center']
        if joint_info["joint_type"] == 'p':
            pos = center
        else:
            pos += joint_info['direction'] * np.dot(joint_info['direction'], center - pos)
        save_axis_mesh(joint_info['direction'], pos, 
                       f'{mesh_path}/axis_{i}_{joint_info["joint_type"]}.ply')
        
        if gt_joint_list is not None:
            gt_joint_info = gt_joint_list[i]
            gt_pos = gt_joint_info['origin']
            if gt_joint_info["joint_type"] == 'p':
                gt_pos = center
            else:
                gt_pos += gt_joint_info['direction'] * np.dot(gt_joint_info['direction'], center - gt_pos)
            save_axis_mesh(gt_joint_info['direction'], gt_pos, 
                        f'{mesh_path}/axis_{i}_{gt_joint_info["joint_type"]}_gt.ply')


def render_set(args, name, iteration, views, gaussians: GaussianModel, pipe, background, deform: DeformModel, mode):
    model_path = args.model_path
    save_dir = os.path.join(model_path, name, f"ours_{iteration}")
    mesh_path = os.path.join(save_dir, "meshes")
    makedirs(save_dir, exist_ok=True)
    makedirs(mesh_path, exist_ok=True)

    xc = gaussians.get_xyz
    gs_mask = deform.deform.get_mask(xc)
    gs_mask_id = torch.argmax(gs_mask, dim=-1)
    num_d_joints = len(deform.deform.joint_types) - 1
    whole_mesh = o3d.geometry.TriangleMesh()
    n_dyn_frames = len(views[1])
    _, _, theta = deform.deform.art_model(torch.arange(n_dyn_frames)[:, None].cuda() / n_dyn_frames) 
    theta = theta.squeeze(-1).cpu().numpy() # (1 + num_d_joints, n_dyn_frames)  
    np.save(os.path.join(save_dir, "joint_value.npy"), theta)
    if mode == 'recon':
        mask_ids = range(-1, num_d_joints + 1)
    else:
        mask_ids = [-1]
    for mask_id in mask_ids:
        if mask_id == -1:
            render_path = os.path.join(save_dir, "renders", "{}".format(mask_id))
            gts_path = os.path.join(save_dir, "gt", "{}".format(mask_id))
            depth_path = os.path.join(save_dir, "depth", "{}".format(mask_id))
            mask_path = os.path.join(save_dir, "masks", "{}".format(mask_id))

            makedirs(render_path, exist_ok=True)
            makedirs(gts_path, exist_ok=True)
            makedirs(depth_path, exist_ok=True)
            makedirs(mask_path, exist_ok=True)
        if mask_id > 0 and 'real_' in args.source_path: # filter noise gaussians for real-world objects
            mask_part = gs_mask_id == mask_id
            _, mask_cluster = find_biggest_cluster(xc[mask_part].cpu().numpy(), eps=0.05, min_samples=2)
            keep_mask = torch.ones(len(xc), dtype=torch.bool).cuda()
            keep_mask[mask_part] = torch.tensor(mask_cluster, dtype=torch.bool).cuda()
        else:
            keep_mask = None
        rgbs = []
        gt_rgbs = []
        depths = []
        vis_mask = gs_mask_id == mask_id if mask_id != -1 else None
        if keep_mask != None and vis_mask != None:
            vis_mask = vis_mask & keep_mask

        if vis_mask is not None and vis_mask.sum() <= 10:
            print(f"Too few visible points for part {mask_id}, skip rendering")
            continue
        if mode == 'mask':
            mask = gs_mask_id
        else:
            mask = None
        for view in tqdm(views[0], desc=f"Rendering part {mask_id} at state 0"):
            gt_image = torch.clamp(view.original_image.to("cuda"), 0.0, 1.0)
            gt_rgbs.append(gt_image.cpu().numpy())
            results = render_gsplat(view, gaussians, pipe, background, vis_mask=vis_mask, mask=mask)
            rgb = torch.clamp(results["render"], 0.0, 1.0)
            rgbs.append(rgb.cpu().numpy())
            depth = results['depth']
            depth = depth * (results["alpha"].squeeze(-1) > 0.9)
            depths.append(depth.cpu().numpy())
        
        for view in tqdm(views[1], desc=f"Rendering part {mask_id} at state 1"):
            gt_image = torch.clamp(view.original_image.to("cuda"), 0.0, 1.0)
            gt_rgbs.append(gt_image.cpu().numpy())
            results = render_gsplat(view, gaussians, pipe, background, vis_mask=vis_mask, mask=mask)
            # view.gt_alpha_mask = torch.clamp(results["alpha"].squeeze(-1), 0.0, 1.0)
            rgb = torch.clamp(results["render"], 0.0, 1.0)
            rgbs.append(rgb.cpu().numpy())
            depth = results['depth']
            depth = depth * (results["alpha"].squeeze(-1) > 0.9)
            depths.append(depth.cpu().numpy())

        if mode == 'recon' and mask_id != -1:
            cameras = views[0] + views[1]
            gsExtractor = GaussianExtractor(cameras, gt_rgbs, depths, depth_trunc=10)
            try:
                mesh = gsExtractor.extract_mesh()
            except:
                print(f"Error extracting mesh for {name} mask_id {mask_id}")
                continue
            save_path = os.path.join(mesh_path, f'part_{mask_id}.ply')
            o3d.io.write_triangle_mesh(save_path, mesh)
            whole_mesh += mesh

        if mask_id == -1:
            for i, view in tqdm(enumerate(views[1]), desc=f"Rendering part {mask_id} at state 1"):
                d_value = deform.deform.one_transform(gaussians, view.fid, None, is_training=False)
                d_xyz, d_rot = d_value['d_xyz'], d_value['d_rotation']
                results = render_gsplat(view, gaussians, pipe, background, d_xyz=d_xyz, d_rot=d_rot, vis_mask=vis_mask, mask=mask)
                rgb = torch.clamp(results["render"], 0.0, 1.0)
                rgbs[len(views[0]) + i] = rgb.cpu().numpy()
                depths[len(views[0]) + i] = results['depth'].cpu().numpy()
            for i in range(len(rgbs)):
                rgb = torch.tensor(rgbs[i])
                gt = torch.tensor(gt_rgbs[i])
                if mode != 'mask':
                    vis_depth(depths[i], os.path.join(depth_path, '{0:06d}'.format(i) + ".png"))
                    torchvision.utils.save_image(rgb, os.path.join(render_path, '{0:06d}'.format(i) + ".png"))
                    torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:06d}'.format(i) + ".png"))
                else:
                    torchvision.utils.save_image(rgb, os.path.join(mask_path, '{0:06d}'.format(i) + ".png"))
    if mode == 'recon':
        o3d.io.write_triangle_mesh(os.path.join(mesh_path, 'whole_mesh.ply'), whole_mesh)
    # export joint info
    pred_joint_list = [{}] + deform.deform.get_joint_param()
    mesh_files = [f'meshes/part_{i}.ply' for i in range(len(pred_joint_list))]

    joint_limit = theta.min(1), theta.max(1)
    export_joint_info_json(pred_joint_list, mesh_files, joint_limit, save_dir)
    pred_joint_list = load_joint_infos(os.path.join(save_dir, 'joint_info.json'))
    try:
        gt_joint_list = read_gt(f'{args.source_path}/gt/mobility_v2.json')
    except:
        gt_joint_list = None
    vis_joint(mesh_path, pred_joint_list, gt_joint_list)


def render_sets(args, dataset: ModelParams, iteration, pipe, mode):
    with torch.no_grad():
        deform = DeformModel(dataset)
        loaded = deform.load_weights(dataset.model_path, iteration=iteration)
        if not loaded:
            raise ValueError(f"Failed to load weights from {dataset.model_path}")
        deform.update(20000)
        try:
            gaussians = GaussianModel(dataset.sh_degree, use_2dgs=args.use_2dgs, use_marble=args.use_marble)
        except:
            gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        cam_traj = scene.getTrainCameras()

        render_set(args, "train", iteration, cam_traj, gaussians, pipe, background, deform, mode)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mode", default='recon', choices=['render', 'recon', 'mask'])

    args = get_combined_args(parser)

    print("Rendering " + args.source_path + ' with '+ args.model_path)
    safe_state(args.quiet)
    seed_everything(args.seed)
    render_sets(args, model.extract(args), args.iteration, pipeline.extract(args), args.mode)
