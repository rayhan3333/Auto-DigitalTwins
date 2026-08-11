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
import torch
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from pytorch_lightning import seed_everything
from trainer import Trainer



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[5000, 10_000, 20_000, 40_000, 60_000, 80_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args(sys.argv[1:])
    args.source_path = f"{args.source_path}/{args.dataset}/{args.subset}/{args.scene_name}"
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    seed_everything(args.seed)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    joint_infos = json.load(open(f'{args.source_path}/joint_infos.json', 'r'))
    joint_types = [joint_info['joint_type'] for joint_info in joint_infos]
    args.joint_types = joint_types
    args.num_slots = len(joint_types)
    trainer = Trainer(args=args, dataset=lp.extract(args), opt=op.extract(args), pipe=pp.extract(args), saving_iterations=args.save_iterations)
    trainer.train(args.iterations)
    print("\nTraining complete.")
