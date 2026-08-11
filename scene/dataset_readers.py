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
from tqdm import tqdm
from PIL import Image
from typing import NamedTuple, Optional
from utils.graphics_utils import getWorld2View2, focal2fov
import numpy as np
import json
import cv2 as cv
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text

import imageio
import glob

IMAGE_DIR = 'color'
IMAGE_DIR = 'rgba'
IMAGE_DIR = 'images'
EXTENSION = '.jpg'
EXTENSION = '.png'


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fid: float
    depth: Optional[np.array] = None
    mono_depth: Optional[np.array] = None
    feat: Optional[np.array] = None
    part_mask: Optional[np.array] = None
    state_id: Optional[int] = None
    human_mask: Optional[np.array] = None
    conf: Optional[np.array] = None


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info, apply=False):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal
    cam_centers = []
    if apply:
        c2ws = []
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        if apply:
            c2ws.append(C2W)
        cam_centers.append(C2W[:3, 3:4])
    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal
    translate = -center
    if apply:
        c2ws = np.stack(c2ws, axis=0)
        c2ws[:, :3, -1] += translate
        c2ws[:, :3, -1] /= radius
        w2cs = np.linalg.inv(c2ws)
        for i in range(len(cam_info)):
            cam = cam_info[i]
            cam_info[i] = cam._replace(R=w2cs[i, :3, :3].T, T=w2cs[i, :3, 3])
        apply_translate = translate
        apply_radius = radius
        translate = 0
        radius = 1.
        return {"translate": translate, "radius": radius, "apply_translate": apply_translate, "apply_radius": apply_radius}
    else:
        return {"translate": translate, "radius": radius}
    

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'],
                        vertices['blue']]).T / 255.0
    try:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readRaftExhaustiveDataCamera(image_path, raft_dir="raft_exhaustive", raft_masks_dir="raft_masks"):
    basedir ="/".join(image_path.split("/")[:-3])
    image_name = os.path.basename(image_path).split(".")[0]
    exhaustive_raft_dirs = sorted(glob.glob(os.path.join(basedir,raft_dir,image_name+".png"+"*.npy")))
    exhaustive_raft_mask_dirs = sorted(glob.glob(os.path.join(basedir,raft_masks_dir,image_name+".png"+"*.png")))
    assert len(exhaustive_raft_dirs)==len(exhaustive_raft_mask_dirs), "raft and mask not match"
    raft_dict = {}
    raft_msks_dict = {}
    for raft_dir, msk_dir in zip(exhaustive_raft_dirs,exhaustive_raft_mask_dirs):
        raft_name  = ''.join(os.path.basename(raft_dir).split(".")[:-1]).replace("png","").replace("jpg","")
        raft_msk_name  = ''.join(os.path.basename(msk_dir).split(".")[:-1]).replace("png","").replace("jpg","")
        assert raft_name == raft_msk_name , "raft and mask not match"
        # raft_np = np.load(raft_dir)
        # raft_dict[raft_name] = raft_np
        # raft_msks_dict[raft_msk_name] = np.asarray(imageio.imread(msk_dir)>0)
        raft_dict[raft_name] = raft_dir
        raft_msks_dict[raft_msk_name] = msk_dir
    return {"rafts": raft_dict, "raft_msks": raft_msks_dict}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, load_depth=True, load_mono_depth=True, load_feat=True, load_part_mask=True):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        fid = idx / len(cam_extrinsics)
        state_id = 0
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        
        # Extract alpha mask if available
        im_data = np.array(image)
        if im_data.shape[2] == 4:
            alpha_mask = im_data[..., 3:4] / 255.0
        else:
            alpha_mask = np.ones(im_data.shape[:2] + (1,))
        
        # Load depth map if available
        depth_path = image_path.replace('images', 'depth').replace(EXTENSION, '.png')
        if load_depth and os.path.exists(depth_path):
            depth = cv.imread(depth_path, -1) / 1e3
            h, w = depth.shape
            if depth.size == alpha_mask.size:
                depth[alpha_mask[..., 0] < 0.5] = 0
            else:
                depth[cv.resize(alpha_mask[..., 0], [w, h], interpolation=cv.INTER_NEAREST) < 0.5] = 0
            depth[depth < 0.1] = 0
        else:
            depth = None

        # Load mono depth if available
        mono_depth_path = image_path.replace('images', 'mono_depth').replace(EXTENSION, '.png')
        if load_mono_depth and os.path.exists(mono_depth_path):
            mono_depth = cv.imread(mono_depth_path, cv.IMREAD_GRAYSCALE) / 255
            h, w = mono_depth.shape
            if mono_depth.size == alpha_mask.size:
                mono_depth[alpha_mask[..., 0] < 0.5] = 0
            else:
                mono_depth[cv.resize(alpha_mask[..., 0], [w, h], interpolation=cv.INTER_NEAREST) < 0.5] = 0
        else:
            mono_depth = None

        # Load feature maps if available
        feat_path = image_path.replace('images', 'featup_feat_pca96').replace(EXTENSION, '.npy')
        if load_feat and os.path.exists(feat_path):
            feat = np.load(feat_path)
        else:
            feat = None

        # Load part mask if available
        mask_path = image_path.replace('images', 'masks').replace(EXTENSION, '.npy')
        if load_part_mask and os.path.exists(mask_path):
            part_mask = np.load(mask_path)
            part_mask = np.concatenate([np.zeros((1, *part_mask.shape[1:])), part_mask], axis=0)
            part_mask = np.argmax(part_mask, axis=0).squeeze()
        else:
            part_mask = None

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              fid=fid, state_id=state_id, depth=depth, mono_depth=mono_depth, feat=feat, part_mask=part_mask)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readColmapSceneInfo(path, eval=False, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images"
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, 
                                          images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    normalize = True
    nerf_normalization = getNerfppNorm(train_cam_infos, apply=normalize)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        if normalize:
            xyz += nerf_normalization['apply_translate']
            xyz /= nerf_normalization['apply_radius']
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, no_bg=False, load_depth=True, load_mono_depth=True, load_feat=True, load_part_mask=True, neighbors=10, interval=2, resolution=2):
    cam_infos = []

    raft_dir = f"raft_exhaustive_n{neighbors}_i{interval}_r{resolution}"
    raft_masks_dir = f"raft_masks_n{neighbors}_i{interval}_r{resolution}"
    print(f"reading flows with neighbors={neighbors}, interval={interval}, resolution={resolution}")
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]
        fovy = contents["camera_angle_y"]

        frames = contents["frames"]
        frames = sorted(frames, key=lambda x: x['file_path'])
        for idx, frame in tqdm(enumerate(frames), desc="Reading cameras"):
            cam_name = frame["file_path"]
            if cam_name.endswith('.png'):
                EXTENSION = '.png'
            elif cam_name.endswith('.jpg'):
                EXTENSION = '.jpg'
            else:
                EXTENSION = '.png'
                cam_name = cam_name + EXTENSION
            # cam_name = cam_name.replace(f'{idx:04d}', f'{idx:06d}')
            frame_time = frame['time']
            state_id = frame['state']
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            bg = np.array(
                [1, 1, 1]) if white_background else np.array([0, 0, 0])
            im_data = np.array(image)
            if im_data.shape[2] == 4 and no_bg:
                norm_data = im_data / 255.0
                alpha_mask = norm_data[..., 3:4]
                norm_data[:, :, :3] = norm_data[:, :, 3:4] * norm_data[:, :, :3] + bg * (1 - norm_data[:, :, 3:4])
                image = Image.fromarray(np.array(norm_data * 255.0, dtype=np.uint8), "RGBA")
            else:
                alpha_mask = np.ones(im_data.shape[:2])

            FovY = fovy
            FovX = fovx

            idx = str(int(image_name)).zfill(3)
            depth_path = image_path.replace(IMAGE_DIR, 'depth').replace(EXTENSION, '.png')
            if load_depth and os.path.exists(depth_path):
                depth = cv.imread(depth_path, -1) / 1e3
                h, w = depth.shape
                if depth.size == alpha_mask.size:
                    depth[alpha_mask[..., 0] < 0.5] = 0
                else:
                    depth[cv.resize(alpha_mask[..., 0], [w, h], interpolation=cv.INTER_NEAREST) < 0.5] = 0
                depth[depth < 0.1] = 0
            else:
                depth = None

            mono_depth_path = image_path.replace(IMAGE_DIR, 'mono_depth').replace(EXTENSION, '.png')
            if load_mono_depth and os.path.exists(mono_depth_path):
                mono_depth = cv.imread(mono_depth_path, cv.IMREAD_GRAYSCALE) / 255
                h, w = mono_depth.shape
                if mono_depth.size == alpha_mask.size:
                    mono_depth[alpha_mask[..., 0] < 0.5] = 0
                else:
                    mono_depth[cv.resize(alpha_mask[..., 0], [w, h], interpolation=cv.INTER_NEAREST) < 0.5] = 0
            else:
                mono_depth = None

            feat_path = image_path.replace(IMAGE_DIR, 'featup_feat_pca96').replace(EXTENSION, '.npy')
            if load_feat and os.path.exists(feat_path):
                feat = np.load(feat_path)
            else:
                feat = None

            mask_path = image_path.replace(IMAGE_DIR, 'mask').replace(EXTENSION, '.npy')
            if load_part_mask and os.path.exists(mask_path):
                part_mask = np.load(mask_path)
                part_mask = np.concatenate([np.zeros((1, *part_mask.shape[1:])), part_mask], axis=0)
                part_mask = np.argmax(part_mask, axis=0).squeeze()
            else:
                part_mask = None
            
            conf_path = image_path.replace(IMAGE_DIR, 'conf').replace(EXTENSION, '.npy')
            if os.path.exists(conf_path):
                conf = np.load(conf_path)
                h, w = conf.shape
                if conf.size == alpha_mask.size:
                    conf[alpha_mask[..., 0] < 0.5] = 0
                else:
                    conf[cv.resize(alpha_mask[..., 0], [w, h], interpolation=cv.INTER_NEAREST) < 0.5] = 0
            else:
                conf = None

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth=depth, mono_depth=mono_depth,
                                        image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], 
                                        fid=frame_time, state_id=state_id, feat=feat, part_mask=part_mask, conf=conf))

    return cam_infos


def readInfo(path, white_background=False, eval=False, no_bg=True, load_feat=True, args=None):
    print("Reading Training Transforms")
    train_infos = readCamerasFromTransforms(
        path, f"transforms.json", white_background, no_bg=no_bg, load_feat=load_feat, neighbors=args.flow_neighbors, interval=args.flow_interval, resolution=args.flow_resolution)
    try:
        test_infos = readCamerasFromTransforms(
        path, f"transforms_test.json", white_background, no_bg=no_bg, load_feat=load_feat, neighbors=args.flow_neighbors, interval=args.flow_interval, resolution=args.flow_resolution)
    except:
        test_infos = []
    if not eval:
        train_infos.extend(test_infos)
    print(f"Read train transforms with {len(train_infos)} cameras")
    print(f"Read test transforms with {len(test_infos)} cameras")

    nerf_normalization = getNerfppNorm(train_infos)

    ply_path = os.path.join(path, "point_cloud.ply")
    pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_infos,
                           test_cameras=test_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           )
    return scene_info


def vggt_reader(path, white_background=False, no_bg=True, original_resolution=True):
    vggt_results = np.load(f'{path}/data.npz', allow_pickle=True)
    video = vggt_results['video'].transpose(0, 2, 3, 1)
    N, H, W, _ = video.shape
    depths = vggt_results['depths']
    intrinsics = vggt_results['intrinsics']
    poses = vggt_results['poses']
    masks = vggt_results['masks']
    depths[masks[:, 0] == 0] = 0

    if original_resolution:
        img_files = sorted(glob.glob(f'{path}/images/*.png'))
        video = np.stack([np.array(Image.open(img_file)) for img_file in img_files])
        mask_files = sorted(glob.glob(f'{path}/masks/*.npy'))
        masks = np.stack([np.load(mask_file) for mask_file in mask_files]).squeeze()

    if masks.shape[1] >= 2:
        human_masks = masks[:, -1] * 1.0 # [T, H, W]
    else:
        human_masks = np.zeros_like(masks[:, 0]) # [T, H, W]
    alpha_masks = masks[:, 0][..., None] # [T, H, W, 1]
    bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
    cam_infos = []
    N_cano = 100
    for idx, (image, alpha_mask, human_mask, pose, depth, intrinsic) in enumerate(zip(video, alpha_masks, human_masks, poses, depths, intrinsics)):
        w2c = np.linalg.inv(pose)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        FovY = focal2fov(intrinsic[1, 1], H)
        FovX = focal2fov(intrinsic[0, 0], W)
        if idx <= N_cano:
            state_id = 0
            fid = 0
        else:
            state_id = 1
            fid = (idx - N_cano) / (N - N_cano)
        image_path = f'{path}/images/{idx:06d}.png'
        image_name = f'{idx:06d}.png'
        if no_bg:
            image = image * alpha_mask + bg * (1 - alpha_mask)
        image = np.concatenate([image, alpha_mask * 255.0], axis=2)
        image = Image.fromarray(np.array(image, dtype=np.uint8), "RGBA")
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth=depth,
                              image_path=image_path, image_name=image_name, width=W, height=H, 
                              fid=fid, state_id=state_id, human_mask=human_mask)
        cam_infos.append(cam_info)
    return cam_infos


def readVGGT(path, white_background=False, eval=False, no_bg=True, load_feat=True, args=None):
    train_infos = vggt_reader(path, white_background, no_bg)
    print(f"Read {len(train_infos)} train cameras")

    nerf_normalization = getNerfppNorm(train_infos)

    ply_path = os.path.join(path, "point_cloud.ply")
    pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_infos,
                           test_cameras=[],
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           )
    return scene_info

sceneLoadTypeCallbacks = {
    "Blender": readInfo,
    "Colmap": readColmapSceneInfo,
    "VGGT": readVGGT,
}
