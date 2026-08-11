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

import sys
import tqdm
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, ContrastiveLoss
from gaussian_renderer import render_gsplat
from scene import Scene, GaussianModel
from scene.dataset_readers import fetchPly
from utils.general_utils import safe_state, get_linear_noise_func
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from pytorch_lightning import seed_everything
from utils.metrics import *
from utils.log_utils import prepare_output_and_logger


class Trainer:
    def __init__(self, args, dataset, opt, pipe, saving_iterations):
        self.dataset = dataset
        self.args = args
        self.opt = opt
        self.pipe = pipe
        self.saving_iterations = saving_iterations
        self.tb_writer = prepare_output_and_logger(dataset)
        self.gaussians = GaussianModel(dataset.sh_degree, fea_dim=args.feature_dim, use_2dgs=args.use_2dgs, use_marble=args.use_marble)

        self.scene = Scene(dataset, self.gaussians)
        
        self.gaussians.create_from_pcd(fetchPly(f'{args.source_path}/point_cloud.ply'))
        
        self.gaussians.training_setup(opt)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.iter_start = torch.cuda.Event(enable_timing=True)
        self.iter_end = torch.cuda.Event(enable_timing=True)
        self.iteration = 1

        self.viewpoint_stacks = [self.scene.getTrainCameras_canonical()]
        self.ema_loss_for_log = 0.0
        self.progress_bar = tqdm.tqdm(range(self.iteration-1, opt.iterations), desc="Training progress")
        self.smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)

        self.contrastive_loss = ContrastiveLoss(args)
        self.render_feat_step = 3000
        self.metric_depth_loss_weight = args.metric_depth_loss_weight

    def train(self, iters=5000):
        for i in tqdm.trange(iters):
            self.train_step()
    
    def train_step(self):
        self.iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if self.iteration % self.opt.oneupSHdegree_step == 0:
            self.gaussians.oneupSHdegree()
        id = randint(0, len(self.viewpoint_stacks[0]) - 1)
        viewpoint_cam = self.viewpoint_stacks[0][id]
        
        # Render
        random_bg = (not self.dataset.white_background and self.opt.random_bg_color) and viewpoint_cam.gt_alpha_mask is not None
        bg = self.background if not random_bg else torch.rand_like(self.background).cuda()
        d_xyz, d_rot = None, None
        render_features = self.args.feature_dim > 0
        render_pkg_re = render_gsplat(viewpoint_cam, self.gaussians, self.pipe, bg, d_xyz=d_xyz, d_rot=d_rot, is_training=True, render_features=render_features, use_2dgs=self.args.use_2dgs)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re["viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
        if random_bg:
            gt_image = gt_alpha_mask * gt_image + (1 - gt_alpha_mask) * bg[:, None, None]
        elif self.dataset.white_background and viewpoint_cam.gt_alpha_mask is not None:
            gt_image = gt_alpha_mask * gt_image + (1 - gt_alpha_mask) * self.background[:, None, None]

        if viewpoint_cam.human_mask != None:
            valid_mask = viewpoint_cam.human_mask < 0.5
            valid_mask = valid_mask.unsqueeze(0) # [1, H, W]
        else:
            valid_mask = torch.ones(1, gt_image.shape[1], gt_image.shape[2], dtype=torch.bool, device="cuda")
        
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # depth loss
        depth_loss = torch.tensor([0.])
        if gt_alpha_mask is not None:
            valid_mask = gt_alpha_mask > 0.5
        else:
            valid_mask = torch.ones_like(depth)
        if self.metric_depth_loss_weight > 0:
            depth = render_pkg_re['depth']
            gt_depth = viewpoint_cam.depth.cuda()
            depth_valid_mask = (gt_depth > 0.01) & valid_mask
            n_valid_pixel = depth_valid_mask.sum()
            if n_valid_pixel > 100:
                depth_loss = (torch.log(1 + torch.abs(depth - gt_depth)) * depth_valid_mask).sum() / n_valid_pixel
                loss = loss + depth_loss * self.metric_depth_loss_weight

        loss.backward()

        with torch.no_grad():
                # Keep track of max radii in image-space for pruning
            if self.gaussians.max_radii2D.shape[0] == 0:
                self.gaussians.max_radii2D = torch.zeros_like(radii)
            self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            # Densification
            if self.iteration < self.opt.densify_until_iter:
                self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, image.shape[2], image.shape[1])

                if self.iteration > self.opt.densify_from_iter and self.iteration % self.opt.densification_interval == 0:
                    size_threshold = 20 if self.iteration > self.opt.opacity_reset_interval else None
                    self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, 0.01, self.scene.cameras_extent, size_threshold)

                if self.iteration % self.opt.opacity_reset_interval == 0 or (
                        self.dataset.white_background and self.iteration == self.opt.densify_from_iter):
                    self.gaussians.reset_opacity()

            # Optimizer step
            if self.iteration < self.opt.iterations:
                self.gaussians.optimizer.step()
                self.gaussians.update_learning_rate(self.iteration)
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                    
        self.iter_end.record()

        with torch.no_grad():
            # Progress bar
            self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
            if self.iteration % 10 == 0:
                self.progress_bar.set_postfix({"Loss": f"{self.ema_loss_for_log:.{7}f}"})
                self.progress_bar.update(10)
        if self.iteration in self.saving_iterations:
            print("\n[ITER {}] Saving deformation model".format(self.iteration))
            self.scene.save(self.iteration)
        if self.iteration == self.opt.iterations:
            self.progress_bar.close()
            self.scene.save(self.iteration)
        self.iteration += 1


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[5_000, 10_000, 15_000, 20_000])
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args(sys.argv[1:])
    args.source_path = f"{args.source_path}/{args.dataset}/{args.subset}/{args.scene_name}"

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    seed_everything(args.seed)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    trainer = Trainer(args=args, dataset=lp.extract(args), opt=op.extract(args), pipe=pp.extract(args), saving_iterations=args.save_iterations)
    trainer.train(args.iterations)
