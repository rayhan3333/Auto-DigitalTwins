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
import math
from scene.gaussian_model import GaussianModel
from utils.dual_quaternion import quaternion_mul
import seaborn as sns
import numpy as np
from gsplat import rasterization, rasterization_2dgs
from torch.nn import functional as F
from utils.sh_utils import eval_sh


def render_gsplat(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, d_xyz= None, d_rot=None, scaling_modifier=1.0, scale_const=None, random_bg_color=False, mask=None, vis_mask=None, is_training=False, render_features=False, use_2dgs=False, freeze_cano=False, part_prob=None):
    bg = bg_color if not random_bg_color else torch.rand_like(bg_color)

    xyz = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    shs = pc.get_sh
    sh_degree = pc.active_sh_degree
    if pipe.convert_SHs_python:
        shs_view = shs.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_sh.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors = torch.clamp_min(sh2rgb + 0.5, 0.0)
        sh_degree = None
    else:
        colors = shs
    if render_features:
        if part_prob is not None:
            feature = part_prob
        else:
            feature = F.softmax(pc.get_feature, dim=1)
    else:
        feature = None
    
    if freeze_cano:
        xyz = xyz.detach()
        opacity = opacity.detach()
        scales = scales.detach()
        rotations = rotations.detach()
        colors = colors.detach()
        if feature is not None and part_prob is None:
            feature = feature.detach()

    means3D = xyz + d_xyz if d_xyz is not None else xyz
    if scale_const is not None:
        opacity = torch.ones_like(pc.get_opacity)
    rotations = quaternion_mul(d_rot, rotations) if d_rot is not None else rotations

    if mask != None:
        pallete = torch.from_numpy(np.array(sns.color_palette("hls", mask.max() + 2)[1:])).float().to(pc.get_xyz.device)
        colors = pallete[mask]
        sh_degree = None

    if scale_const is not None:
        scales = scale_const * torch.ones_like(scales)

    # Rasterize visible Gaussians to image.
    if vis_mask is not None:
        means3D = means3D[vis_mask]
        colors = colors[vis_mask] if colors is not None else None
        opacity = opacity[vis_mask]
        scales = scales[vis_mask]
        rotations = rotations[vis_mask]
        if feature is not None:
            feature = feature[vis_mask]

    K = viewpoint_camera.get_intrinsics_matrices().cuda()
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1)[None]
    H, W = int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)
    if use_2dgs:
        bg_color = torch.cat([bg_color, torch.zeros(1, device=bg_color.device)], dim=-1)
        render, alpha, render_normals, surf_normals, _, _, info = rasterization_2dgs(
            means=means3D,
            quats=rotations,
            scales=scales,
            opacities=opacity.squeeze(-1),
            colors=colors,
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            packed=False,
            render_mode="RGB+ED",
            sh_degree=sh_degree,
            backgrounds=bg_color[None],
            # rasterize_mode='antialiased',
            # absgrad=True,
            # tile_size=16
        )
    else:
        render, alpha, info = rasterization(
            means=means3D,
            quats=rotations,
            scales=scales,
            opacities=opacity.squeeze(-1),
            colors=colors,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            packed=False,
            render_mode="RGB+ED",
            sh_degree=sh_degree,
            backgrounds=bg_color[None],
            # rasterize_mode='antialiased',
            # absgrad=True,
            # tile_size=16
        )
    if is_training:
        info["means2d"].retain_grad()
    
    rgb = render[:, ..., :3]
    depth = render[..., 3:4]

    if render_features:
        # feat_rescale_factor = 1
        # resolution = 160 if feat_model == 'featup' else 80
        # downscale = 1.0 if not is_training else (feat_rescale_factor*resolution/max(H,W))
        feat_K = K.clone()
        # feat_K[:, :2, :] *= downscale
        # def get_img_resolution(H, W):
        #     if H<W:
        #         new_W = resolution
        #         new_H = int((H/W)*resolution)
        #     else:
        #         new_H = resolution
        #         new_W = int((W/H)*resolution)
        #     return new_H, new_W
        # h, w = get_img_resolution(H, W)
        # if is_training:
        #     feat_h, feat_w = feat_rescale_factor*h, feat_rescale_factor*w
        # else:
        #     feat_h, feat_w = H, W
        feat_h, feat_w = H, W
        feats, feat_alpha, _ = rasterization(
            means=means3D,
            quats=rotations,
            scales=scales,
            opacities=opacity.squeeze(-1),
            colors=feature,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=feat_K,  # [1, 3, 3]
            width=feat_w,
            height=feat_h,
            packed=False,
            # tile_size = 10
        )

        N, D = feature.shape
        # feat_shape = feats.shape
        # feats = torch.where(feat_alpha > 0, feats / feat_alpha.detach(), torch.zeros(D, device=feature.device))
        feats = feats.squeeze().permute(2, 0, 1)
    else:
        feats = None
    
    radii = info["radii"].squeeze()
    if radii.shape[-1] == 2: # gsplat 1.5.0 return [N, 2], which is the size of a aabb rectangle
        radii = (radii[..., 0] + radii[..., 1]) / 2
    
    return {"render": rgb.permute(0, 3, 1, 2).squeeze(0),
            "viewspace_points": info["means2d"],
            "visibility_filter": radii > 0,
            "radii": radii,
            "depth": depth.permute(0, 3, 1, 2).squeeze(0),
            "alpha": alpha,
            "bg_color": bg,
            "feat": feats,
            "colors": colors,
            "means3D": means3D
            }


def render_mask(viewpoint_camera, pc: GaussianModel, d_xyz= None, d_rot=None, use_2dgs=False, part_prob=None):
    xyz = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    if part_prob is not None:
        feature = part_prob
    else:
        feature = F.softmax(pc.get_feature, dim=1)
    means3D = xyz + d_xyz if d_xyz is not None else xyz
    rotations = quaternion_mul(d_rot, rotations) if d_rot is not None else rotations

    K = viewpoint_camera.get_intrinsics_matrices().cuda()
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1)[None]
    H, W = int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)
    if use_2dgs:
        feats, feat_alpha, *_ = rasterization_2dgs(
            means=means3D,
            quats=rotations,
            scales=scales,
            opacities=opacity.squeeze(-1),
            colors=feature,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            packed=False,
        )
    else:
        feats, feat_alpha, _ = rasterization(
            means=means3D,
            quats=rotations,
            scales=scales,
            opacities=opacity.squeeze(-1),
            colors=feature,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            packed=False,
        )
    
    feats = feats.squeeze().permute(2, 0, 1)
    return feats, feat_alpha.squeeze()