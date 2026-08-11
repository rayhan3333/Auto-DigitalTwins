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
import json
import tqdm
import torch
import numpy as np
from scene import DeformModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from pytorch_lightning import seed_everything
from utils.general_utils import safe_state
from utils.pointnet2_utils import farthest_point_sample, index_points


class Trainer:
    def __init__(self, args, dataset, opt, pipe, saving_iterations):
        self.dataset = dataset
        self.args = args
        self.opt = opt
        self.pipe = pipe
        self.saving_iterations = saving_iterations

        self.track_data = np.load(f"{self.args.source_path}/filtered.npz")
        self.track3d = torch.from_numpy(self.track_data["coords"]).float().cuda()
        self.vis_mask3d = torch.from_numpy(self.track_data["visibs"]).bool().cuda()
        # self.vis_mask3d = torch.ones_like(self.vis_mask3d).bool().cuda()
        self.num_frames = self.track3d.shape[0]
        self.track_loss_weight = args.track_loss_weight

        idx = farthest_point_sample(self.track3d[0:1], 512).repeat(self.track3d.shape[0], 1)
        self.track3d1 = index_points(self.track3d, idx)
        self.vis_mask3d1 = index_points(self.vis_mask3d.unsqueeze(-1), idx).squeeze(-1)

        self.deform = DeformModel(self.dataset)
        joint_infos = json.load(open(f"{self.args.source_path}/joint_infos.json", "r"))
        self.deform.init_from_joint_info(joint_infos)
        self.deform.train_setting(self.opt)
        self.deform.deform.max_window_size = len(self.track3d)
        self.deform.deform.window_size = len(self.track3d)

        self.iter_start = torch.cuda.Event(enable_timing=True)
        self.iter_end = torch.cuda.Event(enable_timing=True)
        self.iteration = 1
        self.ema_loss_for_log = 0.0
        self.progress_bar = tqdm.tqdm(range(self.iteration - 1, opt.iterations), desc="Training progress")
        self.cd_loss_weight = 1.0
    
    def train(self, iters=5000):
        for i in tqdm.trange(iters):
            self.train_step()

    def train_step(self):
        self.iter_start.record()
        track_loss = 0
        track_loss += self.deform.deform.track_loss_o2o(self.track3d, self.vis_mask3d)
        track_loss += self.deform.deform.track_loss_c2o(self.track3d1, self.vis_mask3d1)
        track_loss.backward()
        self.iter_end.record()

        with torch.no_grad():
            self.ema_loss_for_log = 0.4 * track_loss.item() + 0.6 * self.ema_loss_for_log
            if self.iteration % 10 == 0:
                self.progress_bar.set_postfix({"Loss": f"{self.ema_loss_for_log:.{6}f}"})
                self.progress_bar.update(10)
            if self.iteration == self.opt.iterations:
                self.progress_bar.close()

            if self.iteration in self.saving_iterations:
                print("\n[ITER {}] Saving deformation model".format(self.iteration))
                self.deform.save_weights(self.args.model_path, self.iteration)
                
            self.deform.optimizer.step()
            self.deform.optimizer.zero_grad()
            self.deform.update_learning_rate(self.iteration)
            self.deform.update(max(0, self.iteration))

        self.iteration += 1


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1000, 5_000, 10_000, 15_000, 20_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args(sys.argv[1:])
    args.source_path = f"{args.source_path}/{args.dataset}/{args.subset}/{args.scene_name}"
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    seed_everything(args.seed)

    joint_infos = json.load(open(f'{args.source_path}/joint_infos.json', 'r'))
    joint_types = [joint_info['joint_type'] for joint_info in joint_infos]
    args.joint_types = joint_types
    args.num_slots = len(joint_types)
    trainer = Trainer(args=args, dataset=lp.extract(args), opt=op.extract(args), pipe=pp.extract(args), saving_iterations=args.save_iterations)
    trainer.train(args.iterations)
    print("\nTraining complete.")
