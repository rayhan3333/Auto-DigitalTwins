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

from argparse import ArgumentParser, Namespace
import sys
import os


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if t == bool:
                group.add_argument("--" + key, default=value, action="store_true")
            else:
                group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self.K = 3
        self._source_path = "./data"
        self._model_path = "outputs/artgs/sapien/168_new/base_mi_nt"
        self.scene_name = '168_new'
        self.dataset = 'artgs'
        self.subset = 'sapien'
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.flow_resolution = 1
        self.flow_interval = 30
        self.flow_neighbors = 20
        self.data_device = "cuda"
        self.eval = True
        self.load2gpu_on_the_fly = False
        self.joint_types = ''
        self.slot_size = 32
        self.feature_dim = 0
        self.gumbel = True
        self.scale_factor = 1.
        self.use_art_type_prior = True
        self.dynamic_threshold_ratio = 0.02
        self.cd_min_steps = 1000
        self.cd_max_steps = 5000
        self.opacity_reg_weight = 0.01
        self.coarse_name = 'coarse_gs'
        self.deform_name = 'init'
        self.shift_weight = 0.1
        self.num_slots = 2
        self.tau_decay_steps = 10_000
        self.cano_state = 0
        self.num_dynamic_frames = 100
        self.noise_scale = 0.1
        self.feat_model = 'dvt'
        self.time_model_type = 'predict'
        self.num_states = 14
        self.max_time = None
        self.start_time = 90
        self.points_direction = False
        self.use_motion_grid = False
        self.cano_init_iter = 20000
        self.deform_init_iter = 10000
        self.mask_inv = True
        self.seg_type = 'hybrid'
        self.use_2dgs = False
        self.use_marble = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 20_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.deform_lr_max_steps = 40_000
        self.sh_lr = 0.0025
        self.feature_lr = 0.01
        self.opacity_lr = 0.05
        self.scaling_lr = 0.001
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0005
        self.oneupSHdegree_step = 1000
        self.random_bg_color = True
        self.load_iteration = -1

        self.deform_lr_scale = 1
        self.cd_loss_weight = 100
        self.mask_loss_weight = 0.2
        self.metric_depth_loss_weight = 1.0
        self.mono_depth_loss_weight = 0.0
        self.init_cano_steps = 0
        self.freeze_cano_steps = 0
        self.regroup_interval = 1000
        self.track_loss_weight = 1.0

        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
