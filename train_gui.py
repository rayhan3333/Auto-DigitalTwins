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
import sys
import time
import torch
from gaussian_renderer import render_gsplat
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from pytorch_lightning import seed_everything
import dearpygui.dearpygui as dpg
import datetime
from cam_utils import OrbitCamera
from PIL import Image
from utils.graphics_utils import fov2focal
from trainer import Trainer
import math
import numpy as np
import json


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 1 / tanHalfFovX
    P[1, 1] = 1 / tanHalfFovY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def landmark_interpolate(landmarks, steps, step, interpolation='log'):
    stage = (step >= np.array(steps)).sum()
    if stage == len(steps):
        return max(0, landmarks[-1])
    elif stage == 0:
        return 0
    else:
        ldm1, ldm2 = landmarks[stage-1], landmarks[stage]
        if ldm2 <= 0:
            return 0
        step1, step2 = steps[stage-1], steps[stage]
        ratio = (step - step1) / (step2 - step1)
        if interpolation == 'log':
            return np.exp(np.log(ldm1) * (1 - ratio) + np.log(ldm2) * ratio)
        elif interpolation == 'linear':
            return ldm1 * (1 - ratio) + ldm2 * ratio
        else:
            print(f'Unknown interpolation type: {interpolation}')
            raise NotImplementedError

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar, fid):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.fid = fid
        self.c2w = c2w

        self.fx = fov2focal(fovx, self.image_width)
        self.fy = fov2focal(fovy, self.image_height)
        self.cx = self.image_width / 2.0
        self.cy = self.image_height / 2.0

        w2c = np.linalg.inv(c2w)
        w2c[1:3, :3] *= -1
        w2c[:3, 3] *= -1

        self.world_view_transform = torch.tensor(w2c).transpose(0, 1).cuda().float()
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda().float()
        )
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = -torch.tensor(c2w[:3, 3]).cuda()

    def reset_extrinsic(self, R, T):
        self.world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
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

import open3d as o3d    
def read_pcd(path, num_states):
    xyzs = []
    for i in range(num_states):
        pcd = o3d.io.read_point_cloud(f"{path}/point_cloud_{i}.ply")
        xyz = torch.tensor(np.array(pcd.points), dtype=torch.float32, device="cuda")
        xyzs.append(xyz)
    return xyzs


class GUITrainer(Trainer):
    def __init__(self, args, dataset, opt, pipe, saving_iterations):
        super().__init__(args, dataset, opt, pipe, saving_iterations)
        
        self.visualization_mode = 'RGB'
        self.gui = True # enable gui
        self.W = args.W
        self.H = args.H
        self.cam = OrbitCamera(args.W, args.H, r=args.radius, fovy=args.fovy)
        self.vis_scale_const = None
        self.mode = "render"
        self.seed = "random"
        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.training = False
        self.video_speed = 1.

        self.animation_time = 0.
        self.is_animation = False
        self.need_update_overlay = False
        self.buffer_overlay = None
        self.animation_trans_bias = None
        self.animation_rot_bias = None
        self.animation_scaling_bias = None
        self.animate_tool = None
        self.is_training_animation_weight = False
        self.is_training_motion_analysis = False
        self.motion_genmodel = None
        self.motion_animation_d_values = None
        self.showing_overlay = True
        self.should_save_screenshot = False
        self.should_vis_trajectory = False
        self.screenshot_id = 0
        self.screenshot_sv_path = f'./screenshot/' + datetime.datetime.now().strftime('%Y-%m-%d')
        self.traj_overlay = None
        self.vis_traj_realtime = False
        self.last_traj_overlay_type = None
        self.view_animation = True
        self.n_rings_N = 2
        self.should_render_customized_trajectory = False
        self.should_render_customized_trajectory_spiral = False
        self.mask_id = -1
        self.feature_vis_dim = 0

        dpg.create_context()
        self.register_dpg()
        self.test_step()

    def register_dpg(self):
        ### register texture
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window
        # the rendered image, as the primary window
        with dpg.window(
            tag="_primary_window",
            width=self.W,
            height=self.H,
            pos=[0, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        # dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(
            label="Control",
            tag="_control_window",
            width=600,
            height=self.H,
            pos=[self.W, 0],
            no_move=True,
            no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            def callback_setattr(sender, app_data, user_data):
                setattr(self, user_data, app_data)

            # init stuff
            with dpg.collapsing_header(label="Initialize", default_open=True):

                # seed stuff
                def callback_set_seed(sender, app_data):
                    self.seed = app_data
                    self.seed_everything()

                dpg.add_input_text(
                    label="seed",
                    default_value=self.seed,
                    on_enter=True,
                    callback=callback_set_seed,
                )

                # input stuff
                def callback_select_input(sender, app_data):
                    # only one item
                    for k, v in app_data["selections"].items():
                        dpg.set_value("_log_input", k)
                        self.load_input(v)

                    self.need_update = True

                with dpg.file_dialog(
                    directory_selector=False,
                    show=False,
                    callback=callback_select_input,
                    file_count=1,
                    tag="file_dialog_tag",
                    width=700,
                    height=400,
                ):
                    dpg.add_file_extension("Images{.jpg,.jpeg,.png}")

                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="input",
                        callback=lambda: dpg.show_item("file_dialog_tag"),
                    )
                    dpg.add_text("", tag="_log_input")

                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Visualization: ")

                    def callback_vismode(sender, app_data, user_data):
                        self.visualization_mode = user_data

                    dpg.add_button(
                        label="RGB",
                        tag="_button_vis_rgb",
                        callback=callback_vismode,
                        user_data='RGB',
                    )
                    dpg.bind_item_theme("_button_vis_rgb", theme_button)

                    def callback_vis_traj_realtime():
                        self.vis_traj_realtime = not self.vis_traj_realtime
                        if not self.vis_traj_realtime:
                            self.traj_coor = None
                        print('Visualize trajectory: ', self.vis_traj_realtime)
                    dpg.add_button(
                        label="Traj",
                        tag="_button_vis_traj",
                        callback=callback_vis_traj_realtime,
                    )
                    dpg.bind_item_theme("_button_vis_traj", theme_button)

                    dpg.add_button(
                        label="MotionMask",
                        tag="_button_vis_motion_mask",
                        callback=callback_vismode,
                        user_data='MotionMask',
                    )
                    dpg.bind_item_theme("_button_vis_motion_mask", theme_button)

                    dpg.add_button(
                        label="NodeMotion",
                        tag="_button_vis_node_motion",
                        callback=callback_vismode,
                        user_data='MotionMask_Node',
                    )
                    dpg.bind_item_theme("_button_vis_node_motion", theme_button)

                    dpg.add_button(
                        label="Mask",
                        tag="_button_vis_mask",
                        callback=callback_vismode,
                        user_data='Mask',
                    )
                    dpg.bind_item_theme("_button_vis_mask", theme_button)

                with dpg.group(horizontal=True):
                    dpg.add_text("Scale Const: ")
                    def callback_vis_scale_const(sender):
                        self.vis_scale_const = 10 ** dpg.get_value(sender)
                        self.need_update = True
                    dpg.add_slider_float(
                        label="Log vis_scale_const (For debugging)",
                        default_value=-3,
                        max_value=-.5,
                        min_value=-5,
                        callback=callback_vis_scale_const,
                    )

                with dpg.group(horizontal=True):
                    dpg.add_text("Mask ID: ")
                    def callback_vis_mask_id(sender):
                        self.mask_id = dpg.get_value(sender)
                        self.need_update = True
                    dpg.add_slider_int(
                        label="Log mask ID",
                        default_value=-1,
                        max_value=self.args.num_slots-1,
                        min_value=-1,
                        callback=callback_vis_mask_id,
                    )

                with dpg.group(horizontal=True):
                    dpg.add_text("Feat: ")
                    def callback_vis_feature_vis_dim(sender):
                        self.feature_vis_dim = dpg.get_value(sender)
                        self.need_update = True
                    dpg.add_slider_int(
                        label="Log feature_vis_dim",
                        default_value=-1,
                        max_value=self.args.feature_dim-1,
                        min_value=0,
                        callback=callback_vis_feature_vis_dim,
                    )

                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Temporal Speed: ")
                    self.video_speed = 1.
                    def callback_speed_control(sender):
                        self.video_speed = 10 ** dpg.get_value(sender)
                        self.need_update = True
                    dpg.add_slider_float(
                        label="Play speed",
                        default_value=0.,
                        max_value=3.,
                        min_value=-3.,
                        callback=callback_speed_control,
                    )
                
                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Save: ")

                    def callback_save(sender, app_data, user_data):
                        print("\n[ITER {}] Saving Gaussians".format(self.iteration))
                        self.scene.save(self.iteration)
                        self.deform.save_weights(self.args.model_path, self.iteration)
                    dpg.add_button(
                        label="model",
                        tag="_button_save_model",
                        callback=callback_save,
                        user_data='model',
                    )
                    dpg.bind_item_theme("_button_save_model", theme_button)

                    def callback_screenshot(sender, app_data):
                        self.should_save_screenshot = True
                    dpg.add_button(
                        label="screenshot", tag="_button_screenshot", callback=callback_screenshot
                    )
                    dpg.bind_item_theme("_button_screenshot", theme_button)

                    def callback_render_traj(sender, app_data):
                        self.should_render_customized_trajectory = True
                    dpg.add_button(
                        label="render_traj", tag="_button_render_traj", callback=callback_render_traj
                    )
                    dpg.bind_item_theme("_button_render_traj", theme_button)

                    def callback_render_traj(sender, app_data):
                        self.should_render_customized_trajectory_spiral = not self.should_render_customized_trajectory_spiral
                        if self.should_render_customized_trajectory_spiral:
                            dpg.configure_item("_button_render_traj_spiral", label="camera")
                        else:
                            dpg.configure_item("_button_render_traj_spiral", label="spiral")
                    dpg.add_button(
                        label="spiral", tag="_button_render_traj_spiral", callback=callback_render_traj
                    )
                    dpg.bind_item_theme("_button_render_traj_spiral", theme_button)
                    
                    def callback_cache_nn(sender, app_data):
                        self.deform.deform.cached_nn_weight = not self.deform.deform.cached_nn_weight
                        print(f'Cached nn weight for higher rendering speed: {self.deform.deform.cached_nn_weight}')
                    dpg.add_button(
                        label="cache_nn", tag="_button_cache_nn", callback=callback_cache_nn
                    )
                    dpg.bind_item_theme("_button_cache_nn", theme_button)

            # training stuff
            with dpg.collapsing_header(label="Train", default_open=True):
                # lr and train button
                with dpg.group(horizontal=True):
                    dpg.add_text("Train: ")

                    def callback_train(sender, app_data):
                        if self.training:
                            self.training = False
                            dpg.configure_item("_button_train", label="start")
                        else:
                            # self.prepare_train()
                            self.training = True
                            dpg.configure_item("_button_train", label="stop")

                    dpg.add_button(
                        label="start", tag="_button_train", callback=callback_train
                    )
                    dpg.bind_item_theme("_button_train", theme_button)

                with dpg.group(horizontal=True):
                    dpg.add_text("", tag="_log_train_psnr")
                with dpg.group(horizontal=True):
                    dpg.add_text("", tag="_log_train_log")

            # rendering options
            with dpg.collapsing_header(label="Rendering", default_open=True):
                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(
                    ("render", "depth", "alpha", "normal_dep", "feat"),
                    label="mode",
                    default_value=self.mode,
                    callback=callback_change_mode,
                )

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = np.deg2rad(app_data)
                    self.need_update = True

                dpg.add_slider_int(
                    label="FoV (vertical)",
                    min_value=1,
                    max_value=120,
                    format="%d deg",
                    default_value=np.rad2deg(self.cam.fovy),
                    callback=callback_set_fovy,
                )
            
        def callback_set_mouse_loc(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            self.mouse_loc = np.array(app_data)

        def callback_keypoint_add(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            ##### select keypoints by shift + click
            if dpg.is_key_down(dpg.mvKey_S) or dpg.is_key_down(dpg.mvKey_D) or dpg.is_key_down(dpg.mvKey_F) or dpg.is_key_down(dpg.mvKey_A) or dpg.is_key_down(dpg.mvKey_Q):
                if not self.is_animation:
                    print("Please switch to animation mode!")
                    return
                # Rendering the image with node gaussians to select nodes as keypoints
                fid = torch.tensor(self.animation_time).cuda().float()
                cur_cam = MiniCam(
                    self.cam.pose,
                    self.W,
                    self.H,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                    fid = fid
                )
                with torch.no_grad():
                    time_input = self.deform.deform.expand_time(fid)
                    d_values = self.deform.step(self.gaussians.get_xyz.detach(), time_input, feature=self.gaussians.feature, is_training=False, motion_mask=self.gaussians.motion_mask, camera_center=cur_cam.camera_center, node_trans_bias=self.animation_trans_bias, node_rot_bias=self.animation_rot_bias, node_scaling_bias=self.animation_scaling_bias)
                    gaussians = self.gaussians
                    d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], d_values['d_scaling'], d_values['d_opacity'], d_values['d_color']

                    out = render(viewpoint_camera=cur_cam, pc=gaussians, pipe=self.pipe, bg_color=self.background, d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling, d_opacity=d_opacity, d_color=d_color, d_rot_as_res=self.deform.d_rot_as_res, use_2dgs=self.args.use_2dgs)

                    # Project mouse_loc to points_3d
                    pw, ph = int(self.mouse_loc[0]), int(self.mouse_loc[1])

                    d = out['depth'][0][ph, pw]
                    z = cur_cam.zfar / (cur_cam.zfar - cur_cam.znear) * d - cur_cam.zfar * cur_cam.znear / (cur_cam.zfar - cur_cam.znear)
                    uvz = torch.tensor([((pw-.5)/self.W * 2 - 1) * d, ((ph-.5)/self.H*2-1) * d, z, d]).cuda().float().view(1, 4)
                    p3d = (uvz @ torch.inverse(cur_cam.full_proj_transform))[0, :3]

                    # Pick the closest node as the keypoint
                    node_trans = self.deform.deform.node_deform(time_input)['d_xyz']
                    if self.animation_trans_bias is not None:
                        node_trans = node_trans + self.animation_trans_bias
                    nodes = self.deform.deform.nodes[..., :3] + node_trans
                    keypoint_idxs = torch.tensor([(p3d - nodes).norm(dim=-1).argmin()]).cuda()

                if dpg.is_key_down(dpg.mvKey_A):
                    if True:
                        keypoint_idxs = self.animate_tool.add_n_ring_nbs(keypoint_idxs, n=self.n_rings_N)
                    keypoint_3ds = nodes[keypoint_idxs]
                    self.deform_keypoints.add_kpts(keypoint_3ds, keypoint_idxs)
                    print(f'Add kpt: {self.deform_keypoints.selective_keypoints_idx_list}')

                elif dpg.is_key_down(dpg.mvKey_S):
                    self.deform_keypoints.select_kpt(keypoint_idxs.item())

                elif dpg.is_key_down(dpg.mvKey_D):
                    if True:
                        keypoint_idxs = self.animate_tool.add_n_ring_nbs(keypoint_idxs, n=self.n_rings_N)
                    keypoint_3ds = nodes[keypoint_idxs]
                    self.deform_keypoints.add_kpts(keypoint_3ds, keypoint_idxs, expand=True)
                    print(f'Expand kpt: {self.deform_keypoints.selective_keypoints_idx_list}')

                elif dpg.is_key_down(dpg.mvKey_F):
                    keypoint_idxs = torch.arange(nodes.shape[0]).cuda()
                    keypoint_3ds = nodes[keypoint_idxs]
                    self.deform_keypoints.add_kpts(keypoint_3ds, keypoint_idxs, expand=True)
                    print(f'Add all the control points as kpt: {self.deform_keypoints.selective_keypoints_idx_list}')

                elif dpg.is_key_down(dpg.mvKey_Q):
                    self.deform_keypoints.select_rotation_kpt(keypoint_idxs.item())
                    print(f"select rotation control points: {keypoint_idxs.item()}")

                self.need_update_overlay = True

        self.callback_keypoint_add = callback_keypoint_add

        def callback_camera_drag_rotate_or_draw_mask(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True
                
        with dpg.handler_registry():
            # for camera moving
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=callback_camera_drag_rotate_or_draw_mask,
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

            dpg.add_mouse_move_handler(callback=callback_set_mouse_loc)
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left, callback=callback_keypoint_add)

        dpg.create_viewport(
            title="Gaussian3D",
            width=self.W + 600,
            height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        if os.path.exists("LXGWWenKai-Regular.ttf"):
            with dpg.font_registry():
                with dpg.font("LXGWWenKai-Regular.ttf", 18) as default_font:
                    dpg.bind_font(default_font)

        dpg.show_viewport()
    
    def render(self):
        assert self.gui
        while dpg.is_dearpygui_running():
            # update texture every frame
            if self.training:
                self.train_step()
            self.test_step()
            dpg.render_dearpygui_frame()

    @torch.no_grad()
    def test_step(self, specified_cam=None):

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        if not hasattr(self, 't0'):
            self.t0 = int(time.time())
            self.fps_of_fid = 10
        
        if self.is_animation:
            if not self.showing_overlay:
                self.buffer_overlay = None
            else:
                self.update_control_point_overlay()
            fid = torch.tensor(self.animation_time).cuda().float()
        else:
            fid = torch.remainder(torch.tensor((int(time.time())-self.t0) * self.fps_of_fid).float().cuda() * self.video_speed / self.T_max, 1.0)
        if specified_cam is not None:
            cur_cam = specified_cam
        else:
            cur_cam = MiniCam(
                self.cam.pose,
                self.W,
                self.H,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
                fid = fid
            )
        fid = cur_cam.fid

        mask = None
        vis_mask = None
        vis_scale_const = None
        gaussians = self.gaussians


        # d_values = self.deform.deform.one_transform(gaussians, fid[None], state=int(fid * self.args.num_slots), is_training=False)
        d_values = self.deform.deform.one_transform(gaussians, fid[None], state=None, is_training=False)
        d_xyz, d_rotation = d_values['d_xyz'], d_values['d_rotation']
        mask = d_values.get('mask', None)
        if self.mask_id >= 0:
            vis_mask = mask == self.mask_id
        
        if self.vis_traj_realtime:
            if 'Node' in self.visualization_mode:
                if self.last_traj_overlay_type != 'node':
                    self.traj_coor = None
                self.update_trajectory_overlay(gs_xyz=gaussians.get_xyz+d_xyz, camera=cur_cam, gs_num=512)
                self.last_traj_overlay_type = 'node'
            else:
                if self.last_traj_overlay_type != 'gs':
                    self.traj_coor = None
                self.update_trajectory_overlay(gs_xyz=gaussians.get_xyz+d_xyz, camera=cur_cam)
                self.last_traj_overlay_type = 'gs'
        
        # if self.visualization_mode == 'Dynamic' or self.visualization_mode == 'Static':
        #     d_opacity = torch.zeros_like(self.gaussians.motion_mask)
        #     if self.visualization_mode == 'Dynamic':
        #         d_opacity[self.gaussians.motion_mask < .9] = - 1e3
        #     else:
        #         d_opacity[self.gaussians.motion_mask > .1] = - 1e3
        
        render_motion = "Motion" in self.visualization_mode
        if render_motion or 'Mask' in self.visualization_mode:
            vis_scale_const = self.vis_scale_const
        else:
            mask = None
        if type(d_rotation) is not float and gaussians._rotation.shape[0] != d_rotation.shape[0]:
            d_xyz, d_rotation, d_scaling = 0, 0, 0
            print('Async in Gaussian Switching')
        render_features = self.args.feature_dim > 0
        out = render_gsplat(viewpoint_camera=cur_cam, pc=gaussians, pipe=self.pipe, bg_color=self.background, d_xyz=d_xyz, d_rot=d_rotation, scale_const=vis_scale_const, mask=mask, vis_mask=vis_mask, render_features=render_features, use_2dgs=self.args.use_2dgs)

        if self.mode == "normal_dep":
            from utils.other_utils import depth2normal
            normal = depth2normal(out["depth"])
            out["normal_dep"] = (normal + 1) / 2

        buffer_image = out[self.mode]  # [3, H, W]

        if self.should_save_screenshot:
            alpha = out['alpha']
            sv_image = torch.cat([buffer_image, alpha], dim=0).clamp(0,1).permute(1,2,0).detach().cpu().numpy()
            def save_image(image, image_dir):
                os.makedirs(image_dir, exist_ok=True)
                idx = len(os.listdir(image_dir))
                print('>>> Saving image to %s' % os.path.join(image_dir, '%06d.png'%idx))
                Image.fromarray((image * 255).astype('uint8')).save(os.path.join(image_dir, '%06d.png'%idx))
            save_image(sv_image, self.screenshot_sv_path)
            self.should_save_screenshot = False

        if self.mode in ['depth', 'alpha']:
            buffer_image = buffer_image.repeat(3, 1, 1)
            if self.mode == 'depth':
                buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)
        elif self.mode == 'feat':
            buffer_image = self.deform.feat_decoder(buffer_image[None])[0]
            buffer_image = buffer_image[self.feature_vis_dim*3:(self.feature_vis_dim+1)*3]
            buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)

        buffer_image = torch.nn.functional.interpolate(
            buffer_image.unsqueeze(0),
            size=(self.H, self.W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        self.buffer_image = (
            buffer_image.permute(1, 2, 0)
            .contiguous()
            .clamp(0, 1)
            .contiguous()
            .detach()
            .cpu()
            .numpy()
        )

        self.need_update = True

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        if self.is_animation and self.buffer_overlay is not None:
            overlay_mask = self.buffer_overlay.sum(axis=-1, keepdims=True) == 0
            try:
                buffer_image = self.buffer_image * overlay_mask + self.buffer_overlay
            except:
                buffer_image = self.buffer_image
        else:
            buffer_image = self.buffer_image

        if self.vis_traj_realtime:
            buffer_image = buffer_image * (1 - self.traj_overlay[..., 3:]) + self.traj_overlay[..., :3] * self.traj_overlay[..., 3:]

        if self.gui:
            dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)} FPS FID: {fid.item()})")
            dpg.set_value(
                "_texture", buffer_image
            )  # buffer must be contiguous, else seg fault!
        return buffer_image


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
    parser.add_argument('--W', type=int, default=1600, help="GUI width")
    parser.add_argument('--H', type=int, default=1600, help="GUI height")
    parser.add_argument('--elevation', type=float, default=0, help="default GUI camera elevation")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)

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
    trainer = GUITrainer(args=args, dataset=lp.extract(args), opt=op.extract(args), pipe=pp.extract(args), saving_iterations=args.save_iterations)
    trainer.render()
    print("\nTraining complete.")
