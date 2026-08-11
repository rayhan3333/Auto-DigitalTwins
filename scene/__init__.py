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
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.deform_model import DeformModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos


class Scene:
    gaussians: GaussianModel
    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.args = args

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = []
        self.test_cameras = []
        if os.path.exists(os.path.join(args.source_path, "transforms.json")):
            print("Found transforms.json file, assuming Nerf data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, load_feat=args.feature_dim > 0, args=args)
        elif os.path.exists(os.path.join(args.source_path, "sparse")):
            print("Found sparse folder, assuming Colmap data set!")
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, eval=args.eval, args=args)
        elif os.path.exists(os.path.join(args.source_path, "data.npz")):
            print("Found vggt_result.npz file, assuming VGGT data set!")
            scene_info = sceneLoadTypeCallbacks["VGGT"](args.source_path, white_background=args.white_background)
        else:
            raise ValueError("No scene info file found at {}".format(args.source_path))
    
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print("Cameras extent: ", self.cameras_extent)
        print("Loading Cameras")
        train_cameras = {}
        self.num_frames = len(scene_info.train_cameras)
        for train_camera in scene_info.train_cameras:
            if train_camera.state_id not in train_cameras:
                train_cameras[train_camera.state_id] = []
            train_cameras[train_camera.state_id].append(train_camera)
        # no test cameras for now

        for state_id in train_cameras.keys():
            self.train_cameras.append(cameraList_from_camInfos(train_cameras[state_id], resolution_scale=1.0, args=args))

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),
                                                 "point_cloud.ply"),
                                    og_number_points=len(scene_info.point_cloud.points))
        
    def save(self, iteration, is_best=False):
        if is_best:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_best")
            self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
            with open(os.path.join(point_cloud_path, "iter.txt"), 'w') as f:
                f.write(f"iteration: {iteration}")
        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
            self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    
    def getTrainCameras_canonical(self):
        return self.train_cameras[0]
    
    def getTestCameras_canonical(self):
        return self.test_cameras[0]
    
    def getTrainCameras_dynamic(self):
        return self.train_cameras[1:]
    
    def getTestCameras_dynamic(self):
        return self.test_cameras[1:]

    def getTrainCameras(self):
        return self.train_cameras

    def getTestCameras(self):
        return self.test_cameras
