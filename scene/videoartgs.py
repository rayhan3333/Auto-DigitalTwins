import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scene.gaussian_model import GaussianModel
from utils.dual_quaternion import *
from scene.module import gumbel_softmax, ProgressiveBandHashGrid, TimePrediction, TimeInterpolation, PolynomialEmbedding
from pytorch3d.loss import chamfer_distance
from utils.geo_utils import find_nearest_points_knn


class HybridSeg(nn.Module):
    def __init__(self, num_slots, slot_size, scale_factor=1.0, shift_weight=0.5):
        super().__init__()
        num_slots = num_slots - 1 # remove the static slot
        self.num_slots = num_slots

        self.grid = ProgressiveBandHashGrid(3, start_level=6, n_levels=12, start_step=0, update_steps=500)
        dim = num_slots * 4 + self.grid.n_output_dims + 3
        self.mlp = nn.Sequential(
                nn.Linear(dim, slot_size),
                nn.ReLU(),
                nn.Linear(slot_size, num_slots * 2),
            )
        self.center = nn.Parameter(torch.randn(num_slots, 3) * 0.01)
        self.logscale = nn.Parameter((torch.rand(num_slots, 3) * 0.1).log())
        self.rot = nn.Parameter(torch.Tensor([[1, 0, 0, 0]]).repeat(self.num_slots, 1))

        self.scale_factor = scale_factor
        self.shift_weight = shift_weight

        self.motion_grid = ProgressiveBandHashGrid(3, start_level=6, n_levels=12, start_step=0, update_steps=500)
        self.motion_mlp = nn.Sequential(
            nn.Linear(self.grid.n_output_dims + 3, slot_size),
            nn.ReLU(),
            nn.Linear(slot_size, slot_size),
            nn.ReLU(),
            nn.Linear(slot_size, 1),
        )


    def cal_mask(self, x, rel_pos, tau, is_training=False):
        static_logits = self.motion_mlp(torch.cat([x, self.motion_grid(x)], -1)) # [N, 1]

        dist = (rel_pos ** 2).sum(-1) # [N, K]
        x_rel = torch.cat([rel_pos, torch.norm(rel_pos, p=2, dim=-1, keepdim=True)], dim=-1) # [N, K, 4]
        info = torch.cat([x_rel.reshape(x.shape[0], -1), self.grid(x), x], -1)
        delta = self.mlp(info) # [N, K * 2]
        logscale, shift = torch.split(delta, delta.shape[-1] // 2, dim=-1) # [N, K]

        dist = dist * (self.shift_weight * logscale).exp()
        logits = -dist + shift * self.shift_weight

        slots = None
        hard = not is_training
        logits = torch.cat([static_logits, logits], dim=-1)
        mask, _ = gumbel_softmax(logits, tau=tau, hard=hard, dim=1, is_training=is_training)
        return slots, mask
        
    def forward(self, x, tau, is_training=False):
        '''
            x: position of canonical gaussians [N, 3]
        '''
        rel_pos = self.cal_relative_pos(x[:, None], self.center[None], self.rot[None], self.scale_factor * self.get_scale[None]) # [N, K, 3]
        slots, mask = self.cal_mask(x, rel_pos, tau, is_training)
        return slots, mask
    
    def forward_obs_space(self, x, tau, slot_qr, slot_qd, is_training=False):
        slot_qr, slot_qd = slot_qr[1:], slot_qd[1:] # remove the static slot
        # x: [N, 3], slot_qr: [K, 4], slot_qd: [K, 4]
        N, K = x.shape[0], slot_qr.shape[0]
        center = dual_quaternion_apply((slot_qr, slot_qd), self.center)[None] # [1, K, 3]
        rot = quaternion_mul(slot_qr, self.rot)[None] # [1, K, 4]
        scale = self.get_scale[None] * self.scale_factor # [1, K, 3]
        rel_pos = self.cal_relative_pos(x[:, None], center, rot, scale) # [N, K, 3]
        slots, mask = self.cal_mask(x, rel_pos, tau, is_training)
        return mask
    
    def forward_obs_space_batch(self, x, tau, slot_qr, slot_qd, is_training=False):
        slot_qr, slot_qd = slot_qr[:, 1:], slot_qd[:, 1:] # remove the static slot
        B, N, K = x.shape[0], x.shape[1], slot_qr.shape[1]
        slot_qr, slot_qd = slot_qr.reshape(B*K, 4), slot_qd.reshape(B*K, 4)
        center = self.center[None].repeat(x.shape[0], 1, 1).reshape(B*K, 3) # [B*K, 3]
        center = dual_quaternion_apply((slot_qr, slot_qd), center).reshape(B, 1, K, 3) # [B, 1, K, 3]
        rot = quaternion_mul(slot_qr, self.rot[None].repeat(x.shape[0], 1, 1).reshape(B*K, 4)).reshape(B, 1, K, 4) # [B, 1, K, 4]
        scale = self.get_scale[None].repeat(x.shape[0], 1, 1).reshape(B, 1, K, 3) * self.scale_factor # [B, 1, K, 3]
        rel_pos = self.cal_relative_pos(x.reshape(B, N, 1, 3), center, rot, scale).reshape(B*N, K, 3) # [B*N, K, 3]
        slots, mask = self.cal_mask(x.reshape(B*N, 3), rel_pos, tau, is_training)
        return mask.reshape(B, N, K+1)
    
    def forward_obs_space_xbatch(self, x, tau, slot_qr, slot_qd, is_training=False):
        slot_qr, slot_qd = slot_qr[:, 1:], slot_qd[:, 1:] # remove the static slot
        # x: [N, 3], slot_qr: [N, K, 4], slot_qd: [N, K, 4]
        N, K = x.shape[0], slot_qr.shape[1]
        slot_qr, slot_qd = slot_qr.reshape(N*K, 4), slot_qd.reshape(N*K, 4)
        center = self.center[None].repeat(N, 1, 1).reshape(N*K, 3) # [N*K, 3]
        center = dual_quaternion_apply((slot_qr, slot_qd), center).reshape(N, K, 3)
        rot = quaternion_mul(slot_qr, self.rot[None].repeat(N, 1, 1).reshape(N*K, 4)).reshape(N, K, 4) 
        scale = self.get_scale[None].repeat(N, 1, 1).reshape(N, K, 3) * self.scale_factor 
        rel_pos = self.cal_relative_pos(x.reshape(N, 1, 3), center, rot, scale).reshape(N, K, 3) 
        slots, mask = self.cal_mask(x.reshape(N, 3), rel_pos, tau, is_training)
        return mask.reshape(N, K+1)
    
    def init(self, center, scale):
        self.center = nn.Parameter(center[1:])
        self.logscale = nn.Parameter(torch.log(scale[1:].repeat(1, 3)))
    
    def cal_relative_pos(self, x, center, rot, scale):
        return quaternion_apply(rot, (x - center)) / scale # [N, K, 3]
    
    @property
    def get_scale(self):
        return torch.exp(self.logscale)
    
    @property
    def get_rot(self):
        return F.normalize(self.rot, p=2, dim=-1)
    
    def reg_loss(self, xc, mask, opacity=None):
        mask = mask[:, 1:] # remove the static slot
        xc = xc.detach()
        # regularize centers
        if opacity is not None:
            opacity = opacity.detach()
            m = mask * opacity
        else:
            m = mask
        m = m / (m.sum(0, keepdim=True) + 1e-5)
        c = torch.einsum('nk,nj->kj', m, xc)
        reg_loss = F.mse_loss(self.center, c) * 0.1
        return reg_loss


class CenterBasedSeg(nn.Module):
    def __init__(self, num_slots, slot_size, scale_factor=1.0, shift_weight=0.5):
        super().__init__()
        self.num_slots = num_slots

        self.grid = ProgressiveBandHashGrid(3, start_level=6, n_levels=12, start_step=0, update_steps=500)
        dim = num_slots * 4 + self.grid.n_output_dims + 3
        self.mlp = nn.Sequential(
                nn.Linear(dim, slot_size),
                nn.ReLU(),
                nn.Linear(slot_size, num_slots * 2),
            )
        self.center = nn.Parameter(torch.randn(num_slots, 3) * 0.01)
        self.logscale = nn.Parameter((torch.rand(num_slots, 3) * 0.1).log())
        self.rot = nn.Parameter(torch.Tensor([[1, 0, 0, 0]]).repeat(self.num_slots, 1))

        self.scale_factor = scale_factor
        self.shift_weight = shift_weight

    def cal_mask(self, x, rel_pos, tau, is_training=False):
        dist = (rel_pos ** 2).sum(-1) # [N, K]

        x_rel = torch.cat([rel_pos, torch.norm(rel_pos, p=2, dim=-1, keepdim=True)], dim=-1) # [N, K, 4]
        info = torch.cat([x_rel.reshape(x.shape[0], -1), self.grid(x), x], -1)
        delta = self.mlp(info) # [N, K * 2]
        logscale, shift = torch.split(delta, delta.shape[-1] // 2, dim=-1) # [N, K]

        dist = dist * (self.shift_weight * logscale).exp()
        logits = -dist + shift * self.shift_weight

        slots = None
        hard = (tau - 0.1) < 1e-3
        mask, _ = gumbel_softmax(logits, tau=tau / (self.num_slots - 1), hard=hard, dim=1, is_training=is_training)
        return slots, mask
        
    def forward(self, x, tau, is_training=False):
        '''
            x: position of canonical gaussians [N, 3]
        '''
        rel_pos = self.cal_relative_pos(x[:, None], self.center[None], self.rot[None], self.scale_factor * self.get_scale[None]) # [N, K, 3]
        slots, mask = self.cal_mask(x, rel_pos, tau, is_training)
        return slots, mask
    
    def forward_obs_space(self, x, tau, slot_qr, slot_qd, is_training=False):
        # x: [N, 3], slot_qr: [K, 4], slot_qd: [K, 4]
        N, K = x.shape[0], slot_qr.shape[0]
        center = dual_quaternion_apply((slot_qr, slot_qd), self.center)[None] # [1, K, 3]
        rot = quaternion_mul(slot_qr, self.rot)[None] # [1, K, 4]
        scale = self.get_scale[None] * self.scale_factor # [1, K, 3]
        rel_pos = self.cal_relative_pos(x[:, None], center, rot, scale) # [N, K, 3]
        slots, mask = self.cal_mask(x, rel_pos, tau, is_training)
        return mask
    
    def forward_obs_space_batch(self, x, tau, slot_qr, slot_qd, is_training=False):
        B, N, K = x.shape[0], x.shape[1], slot_qr.shape[1]
        slot_qr, slot_qd = slot_qr.reshape(B*K, 4), slot_qd.reshape(B*K, 4)
        center = self.center[None].repeat(x.shape[0], 1, 1).reshape(B*K, 3) # [B*K, 3]
        center = dual_quaternion_apply((slot_qr, slot_qd), center).reshape(B, 1, K, 3) # [B, 1, K, 3]
        rot = quaternion_mul(slot_qr, self.rot[None].repeat(x.shape[0], 1, 1).reshape(B*K, 4)).reshape(B, 1, K, 4) # [B, 1, K, 4]
        scale = self.get_scale[None].repeat(x.shape[0], 1, 1).reshape(B, 1, K, 3) * self.scale_factor # [B, 1, K, 3]
        rel_pos = self.cal_relative_pos(x.reshape(B, N, 1, 3), center, rot, scale).reshape(B*N, K, 3) # [B*N, K, 3]
        slots, mask = self.cal_mask(x.reshape(B*N, 3), rel_pos, tau, is_training)
        return mask.reshape(B, N, K)
    
    def forward_obs_space_xbatch(self, x, tau, slot_qr, slot_qd, is_training=False):
        # x: [N, 3], slot_qr: [N, K, 4], slot_qd: [N, K, 4]
        N, K = x.shape[0], slot_qr.shape[1]
        slot_qr, slot_qd = slot_qr.reshape(N*K, 4), slot_qd.reshape(N*K, 4)
        center = self.center[None].repeat(N, 1, 1).reshape(N*K, 3) # [N*K, 3]
        center = dual_quaternion_apply((slot_qr, slot_qd), center).reshape(N, K, 3)
        rot = quaternion_mul(slot_qr, self.rot[None].repeat(N, 1, 1).reshape(N*K, 4)).reshape(N, K, 4) 
        scale = self.get_scale[None].repeat(N, 1, 1).reshape(N, K, 3) * self.scale_factor 
        rel_pos = self.cal_relative_pos(x.reshape(N, 1, 3), center, rot, scale).reshape(N, K, 3) 
        slots, mask = self.cal_mask(x.reshape(N, 3), rel_pos, tau, is_training)
        return mask.reshape(N, K)

    
    def init(self, center, scale):
        self.center = nn.Parameter(center)
        self.logscale = nn.Parameter(torch.log(scale.repeat(1, 3) / 2))
    
    def cal_relative_pos(self, x, center, rot, scale):
        return quaternion_apply(rot, (x - center)) / scale # [N, K, 3]
    
    @property
    def get_scale(self):
        return torch.exp(self.logscale)
    
    @property
    def get_rot(self):
        return F.normalize(self.rot, p=2, dim=-1)
    
    def reg_loss(self, xc, mask, opacity=None):
        xc = xc.detach()
        # regularize centers
        if opacity is not None:
            opacity = opacity.detach()
            m = mask * opacity
        else:
            m = mask
        m = m / (m.sum(0, keepdim=True) + 1e-5)
        c = torch.einsum('nk,nj->kj', m, xc)
        reg_loss = F.mse_loss(self.center, c) * 0.1
        return reg_loss


class ArticulationModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.joint_types = args.joint_types
        self.num_joints = len(self.joint_types)
        self.time_model_type = args.time_model_type
        self.points_direction = args.points_direction
        
        if self.points_direction:
            self.pts0 = nn.Parameter(torch.randn(self.num_joints, 3) * 1e-5)
            self.pts1 = nn.Parameter(torch.randn(self.num_joints, 3) * 1e-5)
        else:
            self.origins = nn.Parameter(torch.randn(self.num_joints, 3) * 1e-5)
            self.directions = nn.Parameter(torch.ones(self.num_joints, 3) * 1e-5)

        if self.time_model_type == 'interpolate':
            self.time_model = TimeInterpolation(1, self.num_joints)
        elif self.time_model_type == 'predict':
            self.time_model = TimePrediction(1, 64, self.num_joints)
        elif self.time_model_type == 'polynomial':
            self.time_model = PolynomialEmbedding(1, self.num_joints, degree=10)
        
        self.register_buffer('qr_s', torch.Tensor([1, 0, 0, 0]))
        self.register_buffer('qd_s', torch.Tensor([0, 0, 0, 0]))
    
    def init(self, origin, direction):
        self.origins = nn.Parameter(origin)
        self.directions = nn.Parameter(direction)

    def huristic_update(self, gs, group_id):
        new_origins = []
        with torch.no_grad():
            xc, opacity = gs.get_xyz, gs.get_opacity.squeeze()
            valid_mask = (opacity > 0.9)
            xc, group_id = xc[valid_mask], group_id[valid_mask]
            x_list = [xc[group_id == i] for i in range(self.num_joints)]
            for i, origin in enumerate(self.origins):
                if i == 0 or x_list[i].shape[0] < 100:
                    new_origins.append(origin)
                    continue
                new_origin = find_nearest_points_knn(origin, x_list[i])
                for _ in range(self.n_huristic_iter):
                    new_origin = find_nearest_points_knn(new_origin, x_list[0])
                    new_origin = find_nearest_points_knn(new_origin, x_list[i])
                new_origins.append(new_origin)
            self.origins.copy_(torch.stack(new_origins))

    def reg_loss(self):
        return self.time_model.reg_loss()

    def origin_regularization(self, gs, group_id):
        with torch.no_grad():
            xc, opacity = gs.get_xyz, gs.get_opacity.squeeze()
            valid_mask = (opacity > 0.9)
            xc, group_id = xc[valid_mask], group_id[valid_mask]
            x_list = [xc[group_id == i] for i in range(self.num_joints)]
        reg_loss = torch.tensor([0.], device=xc.device)
        if x_list[0].shape[0] < 100:
            return reg_loss
        n_revolute = 0
        for i, origin in enumerate(self.origins):
            if self.joint_types[i] == 'r' and x_list[i].shape[0] > 100:
                dist2root = torch.norm(origin - x_list[0], dim=-1).min()
                dist2part = torch.norm(origin - x_list[i], dim=-1).min()
                new_origin = find_nearest_points_knn(origin.detach(), x_list[i])
                for _ in range(self.n_huristic_iter):
                    new_origin = find_nearest_points_knn(new_origin, x_list[0])
                    new_origin = find_nearest_points_knn(new_origin, x_list[i])
                dist2new_origin = torch.norm(origin - new_origin, dim=-1)
                dist = dist2root + 2 * dist2part + 10 * dist2new_origin
                reg_loss += dist
                n_revolute += 1
        return reg_loss / n_revolute
    
    def axis2qr(self, axis, theta):
        half_angle = theta.squeeze(-1) / 2
        init_dir = F.normalize(axis, p=2, dim=-1)
        sin_ = torch.sin(half_angle)
        cos_ = torch.cos(half_angle)
        qr = torch.zeros(theta.shape[0], 4).to(axis.device)
        qr[:, 0] = cos_
        qr[:, 1] = init_dir[:, 0] * sin_
        qr[:, 2] = init_dir[:, 1] * sin_
        qr[:, 3] = init_dir[:, 2] * sin_
        return qr

    def forward(self, t):
        qrs = []
        qds = []
        thetas = self.time_model(t) # [num_joints, T, 1] or [num_joints, T, 8]
        if self.points_direction:
            directions = self.pts1 - self.pts0
            origins = self.pts0
        else:
            directions = self.directions
            origins = self.origins

        for i, joint_type in enumerate(self.joint_types):
            if i == 0:
                assert joint_type == 's'
                qr, qd = self.qr_s[None].repeat(t.shape[0], 1), self.qd_s[None].repeat(t.shape[0], 1)
            else:
                direction = F.normalize(directions[i], p=2, dim=-1)[None] # [1, 3]
                theta = thetas[i] # [T, 1]
                if joint_type == 'p':
                    qr = self.qr_s[None].repeat(t.shape[0], 1)
                    t0 = torch.cat([torch.zeros(t.shape[0], 1).to(qr.device), direction * theta], dim=-1) # [T, 4]
                    qd = 0.5 * quaternion_mul(t0, qr)
                else:
                    qr = self.axis2qr(direction, theta)
                    t0 = torch.cat([torch.zeros(t.shape[0], 1).to(qr.device), origins[i][None].repeat(t.shape[0], 1)], dim=-1)
                    qd = 0.5 * (quaternion_mul(t0, qr) - quaternion_mul(qr, t0)) # better for multi-part real world objects
            qrs.append(qr)
            qds.append(qd)
        qrs, qds = torch.stack(qrs, 1), torch.stack(qds, 1)
        qrs, qds = qrs.squeeze(0), qds.squeeze(0)
        # if self.iter % 1000 == 0:
        #     print(self.directions.tolist(), self.origins.tolist(), thetas.tolist(), qrs.tolist(), qds.tolist())
        return qrs, qds, thetas 

    def get_joint_param(self):
        joint_info_list = []
        if self.points_direction:
            directions = F.normalize(self.pts1 - self.pts0, p=2, dim=-1)
            origins = self.pts0
        else:
            directions = F.normalize(self.directions, p=2, dim=-1)
            origins = self.origins
        # directions = self.direction_mlp(directions)
        # origins = self.origin_mlp(origins)
        origins += directions * torch.einsum('ij,ij->i', directions, -origins)[:, None]
        for i in range(1, self.num_joints):
            direction = directions[i]
            if self.joint_types[i] == 'r':
                joint_info = {
                    "joint_type": 'r',
                    'origin': origins[i].cpu().numpy(),
                    'direction': direction.cpu().numpy(),
                }
            elif self.joint_types[i] == 'p':
                joint_info = {
                    "joint_type": 'p',
                    'origin': np.zeros(3),
                    'direction': direction.cpu().numpy(),
                }
            joint_info_list.append(joint_info)
        return joint_info_list


class VideoArtGS(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.slot_size = args.slot_size
        self.joint_types = args.joint_types
        self.num_slots = len(self.joint_types)
        seg_type = args.seg_type if hasattr(args, 'seg_type') else 'hybrid'
        if seg_type == 'hybrid':
            self.seg_model = HybridSeg(self.num_slots, self.slot_size, scale_factor=args.scale_factor, shift_weight=args.shift_weight)
        elif seg_type == 'center':
            self.seg_model = CenterBasedSeg(self.num_slots, self.slot_size, scale_factor=args.scale_factor, shift_weight=args.shift_weight)
        else:
            raise ValueError(f"Invalid seg_type: {seg_type}")
        self.art_model = ArticulationModel(args)
        self.tau = 1.0
        self.tau_decay_steps = args.tau_decay_steps
        self.iter = 0
        self.window_size = 10
        self.max_window_size = 1000
        self.mask_inv = args.mask_inv

    def init_from_joint_info(self, joint_infos, init_joint_info=True, init_center=True):
        print('Init VideoArtGS from joint info.')
        center, scale, origin, direction = [], [], [], []
        K = len(joint_infos)
        for joint_info in joint_infos:
            center.append(joint_info['center'])
            scale.append(joint_info['dist_max'])
            origin.append(joint_info['origin'])
            direction.append(joint_info['direction'])
        center = torch.tensor(center, dtype=torch.float32).cuda().reshape(K, 3)
        scale = torch.tensor(scale, dtype=torch.float32).cuda().reshape(K, 1)
        origin = torch.tensor(origin, dtype=torch.float32).cuda().reshape(K, 3)
        direction = torch.tensor(direction, dtype=torch.float32).cuda().reshape(K, 3)
        print(f"init_center: {init_center}, init_joint_info: {init_joint_info}")
        if init_center:
            self.seg_model.init(center, scale)
        if init_joint_info:
            self.art_model.init(origin, direction)

    def slotdq_to_gsdq(self, slot_qr, slot_qd, mask):
        # slot_qr: [K, 4], slot_qd: [K, 4], mask: [N, K]
        qr = torch.einsum('nk, kl->nl', mask, slot_qr)   # [N, 4]
        qd = torch.einsum('nk, kl->nl', mask, slot_qd)   # [N, 4]
        return normalize_dualquaternion(qr, qd)
    
    def slotdq_to_gsdq_batch(self, slot_qr, slot_qd, mask):
        # slot_qr: [B, K, 4], slot_qd: [B, K, 4], mask: [B, N, K]
        qr = torch.einsum('bnk, bkl->bnl', mask, slot_qr)   # [B, N, 4]
        qd = torch.einsum('bnk, bkl->bnl', mask, slot_qd)   # [B, N, 4]
        return normalize_dualquaternion(qr, qd)
    
    def get_slot_deform(self, fid, state=None):
        if state is None:
            qrs, qds, thetas = self.art_model(fid)
        elif state == 0:
            qrs, qds = self.art_model.qr_s[None].repeat(self.num_slots, 1), self.art_model.qd_s[None].repeat(self.num_slots, 1)
        else:
            qrs, qds = self.art_model.qr_s[None].repeat(self.num_slots, 1), self.art_model.qd_s[None].repeat(self.num_slots, 1)
            qrs1, qds1 = self.art_model(fid)
            qrs[state] = qrs1[state]
            qds[state] = qds1[state]
        return qrs, qds, thetas

    def deform_pts(self, xc, mask, slot_qr, slot_qd):
        # xc: [N, 3], mask: [N, K], slot_qr: [K, 4], slot_qd: [K, 4]
        gs_qr, gs_qd = self.slotdq_to_gsdq(slot_qr, slot_qd, mask)
        xt = dual_quaternion_apply((gs_qr, gs_qd), xc)
        return xt, gs_qr
    
    def deform_pts_batch(self, xc, mask, slot_qr, slot_qd):
        # xc: [B, N, 3], mask: [B, N, K], slot_qr: [B, K, 4], slot_qd: [B, K, 4]
        gs_qr, gs_qd = self.slotdq_to_gsdq_batch(slot_qr, slot_qd, mask) # [B, N, 4]
        xt = dual_quaternion_apply((gs_qr, gs_qd), xc) # [B, N, 3]
        return xt, gs_qr
    
    def trainable_parameters(self):
        params = [
            {'params': self.art_model.origins, 'name': 'origins'},
            {'params': self.art_model.directions, 'name': 'directions'},
            {'params': list(self.art_model.time_model.parameters()), 'name': 'time_model'},
            {'params': list(self.seg_model.parameters()), 'name': 'seg_model'},
            ]
        return params
    
    def get_mask(self, xc, is_training=False):
        tau = self.tau if is_training else 0.1
        slots, mask = self.seg_model(xc, tau, is_training)
        self.slots = slots
        return mask
    
    @torch.no_grad()
    def get_joint_param(self):
        joint_infos = self.art_model.get_joint_param()
        for i, joint_info in enumerate(joint_infos):
            joint_info['center'] = self.seg_model.center[i].cpu().numpy()
        return joint_infos
    
    def reg_loss(self, xc, mask, opacity=None):
        return self.seg_model.reg_loss(xc, mask, opacity)

    def obs2cano(self, xt, t, is_training=False):
        qr, qd, _ = self.get_slot_deform(t)
        mask_inv = self.seg_model.forward_obs_space(xt, self.tau, qr, qd, is_training)
        qr_inv, qd_inv = dual_quaternion_inverse((qr, qd))
        xc, _ = self.deform_pts(xt, mask_inv, qr_inv, qd_inv)
        return xc, mask_inv
    
    def cano2obs(self, xc, t, mask=None, is_training=False):
        if mask is None:
            mask = self.get_mask(xc, is_training) # [N, K]
        qr, qd, _ = self.get_slot_deform(t)
        xt, rot = self.deform_pts(xc, mask, qr, qd)
        return xt, rot
    

    def sample_track_o2o(self, track3d, vis_mask3d, static_points=None):
        """
        Sample a short track with random start and end time, end time is within 30 frames of start time
        """
        T, N, _ = track3d.shape
        bs_time, bs_point, bs_static = 64, 512, 512
        src_id = torch.randint(0, T, (bs_time,), device=track3d.device)
        tgt_id = src_id + torch.randint(-30, 30, (bs_time,), device=track3d.device)
        tgt_id = torch.clamp(tgt_id, 0, T-1)

        src_mask, tgt_mask = vis_mask3d[src_id], vis_mask3d[tgt_id]
        valid_mask = src_mask * tgt_mask
        point_idx = torch.randperm(N)[:bs_point]
        xt = track3d[src_id][:, point_idx]
        xt_tgt = track3d[tgt_id][:, point_idx]
        valid_mask = valid_mask[:, point_idx]
        t = src_id / T
        t_tgt = tgt_id / T

        if static_points is not None:
            static_rand_idx = torch.randint(0, static_points.shape[0], (bs_time, bs_static), device=track3d.device)
            sampled_static_points = static_points[static_rand_idx].reshape(bs_time, bs_static, 3)
            xt = torch.cat([xt, sampled_static_points], dim=1)
            xt_tgt = torch.cat([xt_tgt, sampled_static_points], dim=1)
            valid_mask = torch.cat([valid_mask, torch.ones(bs_time, bs_static, dtype=valid_mask.dtype, device=valid_mask.device)], dim=1)
        else:
            sampled_static_points = None
        return xt, xt_tgt, valid_mask, t, t_tgt
    
    def sample_track_c2o(self, track3d, vis_mask3d):
        """
        Sample a track with random end time, end time is within window_size frames of start time
        """
        T, N, _ = track3d.shape
        bs_time = 64
        tgt_id = torch.randint(0, self.window_size, (bs_time,), device=track3d.device)
        tgt_id = torch.clamp(tgt_id, 0, T-1)

        src_mask, tgt_mask = vis_mask3d[0:1], vis_mask3d[tgt_id]
        valid_mask = src_mask * tgt_mask

        xc = track3d[0]
        xt_tgt = track3d[tgt_id]
        valid_mask = valid_mask
        t_tgt = tgt_id / T
        return xc, xt_tgt, valid_mask, t_tgt
    
    def track_loss_one_sample_o2o(self, xt, xt_tgt, valid_mask, t, t_tgt):
        """
        Track loss for one sample from random xt to random xt_tgt
        """
        qr, qd, _ = self.get_slot_deform(t[:, None])
        mask_inv = self.seg_model.forward_obs_space_batch(xt, self.tau, qr, qd, is_training=True)
        qr_inv, qd_inv = dual_quaternion_inverse((qr, qd))
        xc, _ = self.deform_pts_batch(xt, mask_inv, qr_inv, qd_inv)
        track_loss, mask = self.track_loss_one_sample_c2o(xc, xt_tgt, valid_mask, t_tgt, mask_inv=mask_inv if self.mask_inv else None)
        return track_loss
    
    def track_loss_one_sample_c2o(self, xc, xt_tgt, valid_mask, t_tgt, mask_inv=None, reg=False):
        """
        Track loss for one sample from canonical xc (t=0) to xt_tgt
        """
        B, N, _ = xt_tgt.shape
        if mask_inv is not None:
            mask = mask_inv
        else:
            mask = self.get_mask(xc.reshape(-1, 3), is_training=True)
        if xc.ndim == 2:
            xc = xc.reshape(1, N, -1).repeat(B, 1, 1)
            mask = mask.reshape(1, N, -1).repeat(B, 1, 1)
        else:
            xc = xc.reshape(B, N, -1)
            mask = mask.reshape(B, N, -1)
        qr_tgt, qd_tgt, _ = self.get_slot_deform(t_tgt[:, None])
        xt_pred, _ = self.deform_pts_batch(xc, mask, qr_tgt, qd_tgt)
        track_loss = (xt_pred - xt_tgt).norm(p=2, dim=-1)[valid_mask].mean()
        if reg:
            track_loss += self.reg_loss(xc[0], mask[0])
        return track_loss, mask
    
    def track_loss_c2o(self, track3d, vis_mask3d, reg=True):
        sampled_track_c2o = self.sample_track_c2o(track3d, vis_mask3d)
        track_loss, _ = self.track_loss_one_sample_c2o(*sampled_track_c2o, reg=reg)
        return track_loss
    
    def track_loss_o2o(self, track3d, vis_mask3d, static_points=None):
        sampled_short_track = self.sample_track_o2o(track3d, vis_mask3d, static_points)
        track_loss = self.track_loss_one_sample_o2o(*sampled_short_track)
        return track_loss
    
    
    def one_transform(self, gaussians:GaussianModel, fid, state, is_training):
        xc = gaussians.get_xyz.detach()
        N = xc.shape[0]
        mask = self.get_mask(xc, is_training) # [N, K]
        # group_id = gaussians.get_group_id

        if state is None or state > 0:
            qr, qd, _ = self.get_slot_deform(fid, state)
            xt, rot = self.deform_pts(xc, mask, qr, qd)
            # xt, rot = self.deform_grouped_pts(xc, group_id, qr, qd)
            d_xyz = xt - xc
            d_rotation = rot.detach()

            return {
                'd_xyz': d_xyz,
                'd_rotation': d_rotation,
                'xt': xt,
                'mask': mask.argmax(-1),
                'prob': mask,
                # 'mask': group_id,
            }
        else:
            dr = torch.zeros(N, 4).to(xc.device)
            dr[:, 0] = 1
            return {
                'd_xyz': torch.zeros_like(gaussians.get_xyz),
                'd_rotation': dr,
                'xt': gaussians.get_xyz.detach(),
                'mask': mask.argmax(-1),
                'prob': mask,
                # 'mask': group_id,
            }
        
    def forward(self, gaussians: GaussianModel, fids, is_training=False):
        xc = gaussians._xyz.detach()
        N = xc.shape[0]
        d_values_list = []
        mask = self.get_mask(xc, is_training) # [N, K]
        for fid in fids:
            if fid == 0:
                dr = torch.zeros(N, 4).to(xc.device)
                dr[:, 0] = 1
                d_values = {
                    'd_xyz': torch.zeros_like(gaussians.get_xyz),
                    'd_rotation': dr,
                    'xt': gaussians.get_xyz.detach(),
                    'mask': mask.argmax(-1),
                    'prob': mask,
                }
            else:
                qr, qd, _ = self.get_slot_deform(fid)
                xt, rot = self.deform_pts(xc, mask, qr, qd)
                d_xyz = xt - xc
                d_rotation = rot.detach()
                d_values = {
                    'd_xyz': d_xyz,
                    'd_rotation': d_rotation,
                    'xt': xt,
                    'mask': mask.argmax(-1),
                    'prob': mask,
                }
            d_values_list.append(d_values)

        return d_values_list
    
    
    def update(self, iteration, *args, **kwargs):
        self.tau = self.cosine_anneal(iteration, self.tau_decay_steps, 0, 1.0, 0.1)
        self.seg_model.grid.update_step(global_step=iteration)

    def cosine_anneal(self, step, final_step, start_step=0, start_value=1.0, final_value=0.1):
        if start_value <= final_value or start_step >= final_step:
            return final_value
        
        if step < start_step:
            value = start_value
        elif step >= final_step:
            value = final_value
        else:
            a = 0.5 * (start_value - final_value)
            b = 0.5 * (start_value + final_value)
            progress = (step - start_step) / (final_step - start_step)
            value = a * math.cos(math.pi * progress) + b
        return value