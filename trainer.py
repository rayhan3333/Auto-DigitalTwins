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
import tqdm
import torch
import torchvision
from random import randint
from utils.loss_utils import l1_loss, ssim, ContrastiveLoss
from gaussian_renderer import render_gsplat
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state, get_linear_noise_func, vis_depth
from utils.log_utils import prepare_output_and_logger, training_report
from utils.metrics import *
from utils.depth_loss import DepthLoss
import torch.nn as nn


class Trainer:
    def __init__(self, args, dataset, opt, pipe, saving_iterations):
        self.dataset = dataset
        self.args = args
        self.opt = opt
        self.pipe = pipe
        self.saving_iterations = saving_iterations

        self.gaussians = GaussianModel(dataset.sh_degree, args.feature_dim, use_2dgs=args.use_2dgs, use_marble=args.use_marble)
        self.scene = Scene(dataset, self.gaussians, load_iteration=-1)
        args.max_time = self.scene.num_frames
        self.tb_writer = prepare_output_and_logger(args)
        self.deform = DeformModel(self.dataset)
        print('Init GaussianModel and DeformModel.')

        self.cano_init_iter = args.cano_init_iter
        if self.scene.loaded_iter is None:
            p = args.source_path.replace('./data/', 'outputs/')
            coarse_name = self.args.coarse_name
            self.gaussians.load_ply(f'{p}/{coarse_name}/point_cloud/iteration_{self.cano_init_iter}/point_cloud.ply')
            print('Init canonical gaussians from coarse gaussian.')

        self.gaussians.training_setup(self.opt)

        self.static_points = None      
        self.track_data = np.load(f"{self.args.source_path}/filtered.npz")
        self.track3d = torch.from_numpy(self.track_data["coords"]).float().cuda()
        self.vis_mask3d = torch.from_numpy(self.track_data["visibs"]).bool().cuda()
        self.T_max = self.track3d.shape[0]

        self.deform_init_iter = args.deform_init_iter
        self.init_deform(self.deform_init_iter)
        self.freeze_cano_steps = args.freeze_cano_steps
        self.init_cano_steps = args.init_cano_steps

        self.track_loss_weight = args.track_loss_weight
        
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.iter_start = torch.cuda.Event(enable_timing=True)
        self.iter_end = torch.cuda.Event(enable_timing=True)
        self.iteration = 1 if self.scene.loaded_iter is None else self.scene.loaded_iter

        self.viewpoint_stacks = self.scene.getTrainCameras()
        
        self.ema_loss_for_log = 0.0
        self.best_iteration = 15000
        self.best_joint_error = 1e10
        self.joint_metrics = []

        self.progress_bar = tqdm.tqdm(range(self.iteration - 1, opt.iterations), desc="Training progress")
        self.smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)

        self.metric_depth_loss_weight = args.metric_depth_loss_weight
        self.mono_depth_loss_weight = args.mono_depth_loss_weight
        self.reg_weight = self.args.opacity_reg_weight

        self.depth_loss = DepthLoss()
        self.contrastive_loss = ContrastiveLoss(args)
        self.mask_loss = nn.CrossEntropyLoss()  
        self.mask_loss_weight = args.mask_loss_weight
        os.makedirs(f'{args.model_path}/zvis', exist_ok=True)
        self.palette = np.concatenate([np.zeros((1, 3)), np.random.randint(0, 256, (100, 3), dtype=np.uint8)], 0)
    
    def init_deform(self, deform_iter=20000):
        self.deform = DeformModel(self.dataset)
        self.deform.deform.max_window_size = len(self.track3d)
        if self.scene.loaded_iter is None:
            deform_dir = self.args.source_path.replace('./data/', 'outputs/')
            deform_path = f'{deform_dir}/{self.args.deform_name}/deform/iteration_{deform_iter}/deform.pth'
            self.deform.deform.load_state_dict(torch.load(deform_path, weights_only=True))
        else:
            self.deform.load_weights(self.args.model_path, iteration=self.scene.loaded_iter)
        self.deform.train_setting(self.opt)

    def train(self, iters=5000):
        for i in tqdm.trange(iters):
            self.train_step()

    def train_step(self):
        self.iter_start.record()
        if self.iteration % self.opt.oneupSHdegree_step == 0:
            self.gaussians.oneupSHdegree()

        state = randint(0, len(self.viewpoint_stacks)-1)
        id = randint(0, len(self.viewpoint_stacks[state]) - 1)
        viewpoint_cam = self.viewpoint_stacks[state][id]

        # Render
        render_features = self.iteration > self.init_cano_steps and self.args.feature_dim > 0
        # render_features = self.iteration > self.init_cano_steps
        random_bg = (not self.dataset.white_background and self.opt.random_bg_color) and viewpoint_cam.gt_alpha_mask is not None
        bg = self.background if not random_bg else torch.rand_like(self.background).cuda()
        d_values = self.deform.deform.one_transform(self.gaussians, viewpoint_cam.fid, None, is_training=True)  # []
        d_xyz, d_rot = d_values['d_xyz'], d_values['d_rotation']
        freeze_cano = self.iteration < self.freeze_cano_steps and self.iteration >= self.init_cano_steps
        render_pkg_re = render_gsplat(viewpoint_cam, self.gaussians, self.pipe, bg, d_xyz=d_xyz, d_rot=d_rot, is_training=True, freeze_cano=freeze_cano, render_features=render_features, part_prob=d_values['prob'], use_2dgs=self.args.use_2dgs)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re["viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        gt_alpha_mask = viewpoint_cam.gt_alpha_mask
        if random_bg:
            gt_image = gt_alpha_mask * gt_image + (1 - gt_alpha_mask) * bg[:, None, None]
        elif self.dataset.white_background and viewpoint_cam.gt_alpha_mask is not None:
            gt_image = gt_alpha_mask * gt_image + (1 - gt_alpha_mask) * self.background[:, None, None]

        if viewpoint_cam.human_mask != None:
            valid_mask = viewpoint_cam.human_mask < 0.5
            valid_mask = valid_mask.unsqueeze(0) # [1, H, W]
        else:
            valid_mask = torch.ones(1, gt_image.shape[1], gt_image.shape[2], dtype=torch.bool, device="cuda")
        Ll1 = l1_loss(image, gt_image, valid_mask)
        loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image, mask=valid_mask))

        if gt_alpha_mask is not None:
            valid_mask = valid_mask & (gt_alpha_mask > 0.5)

        # depth loss
        depth_loss = torch.tensor([0.])
        
        if self.metric_depth_loss_weight > 0:
            depth = render_pkg_re['depth']
            gt_depth = viewpoint_cam.depth.cuda()
            depth_valid_mask = (gt_depth > 0.1) & valid_mask
            n_valid_pixel = depth_valid_mask.sum()
            if n_valid_pixel > 100:
                depth_loss = (torch.log(1 + torch.abs(depth - gt_depth)) * depth_valid_mask).sum() / n_valid_pixel
                loss = loss + depth_loss * self.metric_depth_loss_weight

        mono_depth_loss = torch.tensor([0.])
        if self.mono_depth_loss_weight > 0:
            depth = render_pkg_re['depth']
            mono_depth = viewpoint_cam.mono_depth.cuda()
            # mono_depth_loss = depth_rank_loss(depth, mono_depth, gt_alpha_mask)
            mono_depth_loss = self.depth_loss(depth, mono_depth[None], valid_mask)
            loss = loss + mono_depth_loss * self.mono_depth_loss_weight

        if gt_alpha_mask is not None:
            valid_mask = valid_mask & (gt_alpha_mask > 0.5)

        if self.iteration > 3000:
            loss = loss + self.deform.deform.reg_loss(self.gaussians.get_xyz, d_values['prob'], self.gaussians.get_opacity)
    
        if self.track_loss_weight > 0:
            idx = torch.randperm(self.track3d.shape[1])[:512]
            track_loss = self.deform.deform.track_loss_c2o(self.track3d[:, idx], self.vis_mask3d[:, idx], reg=False)
            loss = loss + track_loss * self.track_loss_weight

        
        loss.backward()
        self.iter_end.record()

        with torch.no_grad():
            # Progress bar
            self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
            if self.iteration % 10 == 0:
                self.progress_bar.set_postfix({"Loss": f"{self.ema_loss_for_log:.{6}f}"})
                self.progress_bar.update(10)
            if self.iteration == self.opt.iterations:
                self.progress_bar.close()

            if self.iteration % 1000 == 0:
                try:
                    joint_types = self.deform.deform.joint_types[1:]
                    pred_joint_list = self.deform.deform.get_joint_param()
                    gt_info_list = read_gt(os.path.expanduser(f'{self.args.source_path}/gt/mobility_v2.json'))
                    self.joint_metrics, real_perm = eval_axis_and_state_all(pred_joint_list, gt_info_list)
                except:
                    print('No ground truth info for joint evaluation.')
            # # Log and save
            training_report(self.tb_writer, self.iteration, Ll1, depth_loss, mono_depth_loss, loss, 
                            self.iter_start.elapsed_time(self.iter_end), self.scene, self.joint_metrics)
            if self.iteration % 100 == 0 and self.iteration > 15000:
                cur_joint_error = sum([sum(m) for m in self.joint_metrics]) if len(self.joint_metrics) > 0 else 1e5
                if cur_joint_error < self.best_joint_error or (self.iteration == self.args.iterations and self.best_iteration <= 15000):
                    self.best_iteration = self.iteration
                    self.best_joint_error = cur_joint_error
                
            if self.iteration in self.saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(self.iteration))
                self.scene.save(self.iteration)
                self.deform.save_weights(self.args.model_path, self.iteration)
            if self.iteration == self.best_iteration:
                print("\n[ITER {}] Saving Gaussians".format(self.iteration))
                self.scene.save(self.iteration, is_best=True)
                self.deform.save_weights(self.args.model_path, self.iteration, is_best=True)
            
            if not freeze_cano:
                # Keep track of max radii in image-space for pruning
                if self.gaussians.max_radii2D.shape[0] == 0:
                    self.gaussians.max_radii2D = torch.zeros_like(radii)
                self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # Densification
                if self.iteration < self.opt.densify_until_iter:
                    self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, image.shape[2], image.shape[1])

                    if self.iteration > self.opt.densify_from_iter and self.iteration % self.opt.densification_interval == 0:
                        size_threshold = 20 if self.iteration > self.opt.opacity_reset_interval else None
                        threshold = 0.005 
                        self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, threshold, self.scene.cameras_extent, size_threshold)
                    
                    if self.iteration % self.opt.opacity_reset_interval == 0 or (
                            self.dataset.white_background and self.iteration == self.opt.densify_from_iter):
                        self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.update_learning_rate(self.iteration + self.cano_init_iter)
                self.gaussians.optimizer.zero_grad(set_to_none=True)
            if self.iteration >= self.opt.init_cano_steps:
                self.deform.optimizer.step()
                self.deform.optimizer.zero_grad()
                self.deform.update_learning_rate(self.iteration + self.deform_init_iter)
            
            self.deform.update(self.iteration)

        self.iteration += 1

    def visualize(self, image, gt_image, gt_depth, depth):
        torchvision.utils.save_image(image.detach(), "img.png")
        torchvision.utils.save_image(gt_image, "img_gt.png")
        torchvision.utils.save_image(vis_depth(gt_depth), "gt.png")
        torchvision.utils.save_image(vis_depth(depth.detach()), "pred.png")
