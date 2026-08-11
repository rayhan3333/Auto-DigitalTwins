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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.graphics_utils import fov2focal


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask, image_name, uid, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda", fid=None, state_id=None, depth=None, mono_depth=None, flow_dirs=[], feat=None, part_mask=None, conf=None, human_mask=None, **kwargs):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.flow_dirs = flow_dirs

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.fid = torch.Tensor([fid]).to(self.data_device)
        self.state_id = torch.Tensor(state_id).to(self.data_device) if state_id is not None else None
        
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        self.fx = fov2focal(FoVx, self.image_width)
        self.fy = fov2focal(FoVy, self.image_height)
        self.cx = self.image_width / 2.0
        self.cy = self.image_height / 2.0
        
        self.depth = torch.Tensor(depth).to(self.data_device) if depth is not None else None
        self.mono_depth = torch.Tensor(mono_depth).to(self.data_device) if mono_depth is not None else None
        self.gt_alpha_mask = gt_alpha_mask
        self.feats = torch.Tensor(feat) if feat is not None else None
        self.part_mask = torch.Tensor(part_mask).to(self.data_device) if part_mask is not None else None
        self.conf = torch.Tensor(conf).to(self.data_device) if conf is not None else None
        self.kwargs = kwargs

        if gt_alpha_mask is not None:
            self.gt_alpha_mask = self.gt_alpha_mask.to(self.data_device)
            # self.original_image *= gt_alpha_mask.to(self.data_device)
        self.human_mask = human_mask
        if human_mask is not None:
            self.human_mask = torch.from_numpy(human_mask).to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).to(self.data_device)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).to(self.data_device)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.corr = {}
        self.view_world_transform = self.world_view_transform.inverse() # .cuda()

    def reset_extrinsic(self, R, T):
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, self.trans, self.scale)).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    
    def get_intrinsics_matrices(self):
        """Returns the intrinsic matrices for each camera.

        Returns:
            Pinhole camera intrinsics matrices
        """
        K = torch.zeros((1, 3, 3), dtype=torch.float32)
        K[..., 0, 0] = self.fx
        K[..., 1, 1] = self.fy
        K[..., 0, 2] = self.cx
        K[..., 1, 2] = self.cy
        K[..., 2, 2] = 1.0
        return K


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

    def reset_extrinsic(self, R, T):
        self.world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
