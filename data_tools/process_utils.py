import os
import cv2
import math
import json
import torch
import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm
from plyfile import PlyData, PlyElement
from utils.geo_utils import *
from pytorch3d.loss import chamfer_distance


def visualize_point_cloud(pcd, save_path=None, show_normals=False, normal_scale=0.02):
    """
    pcd: (N, 3) or (N, 6) or (N, 9) numpy array
         - (N, 3): xyz coordinates
         - (N, 6): xyz + rgb colors
         - (N, 9): xyz + rgb + normals
    show_normals: bool, whether to display normal vectors
    normal_scale: float, scale factor for normal vector display
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pcd[:, :3])
    
    geometries = [point_cloud]
    
    if pcd.shape[1] >= 6:
        colors = pcd[:, 3:6]
        if colors.max() > 1:
            colors = colors / 255.0
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    # Check if normals are provided
    if pcd.shape[1] >= 9 and show_normals:
        normals = pcd[:, 6:9]
        point_cloud.normals = o3d.utility.Vector3dVector(normals)
        
        # Create normal vectors as line segments
        step = max(1, len(pcd) // 200)  # Avoid displaying too many normals
        lines = []
        line_points = []
        line_colors = []
        
        for i in range(0, len(pcd), step):
            start_point = pcd[i, :3]
            end_point = pcd[i, :3] + normals[i] * normal_scale
            
            line_points.extend([start_point, end_point])
            lines.append([len(line_points) - 2, len(line_points) - 1])
            line_colors.append([1, 0, 0])  # Red normal vectors
        
        if lines:  # Only create LineSet if we have lines
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(line_points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(line_colors)
            geometries.append(line_set)
    
    elif not point_cloud.has_normals() and show_normals:
        # Estimate normals if not provided but requested
        point_cloud.estimate_normals()
        print("Estimated normals for the point cloud")
    
    if show_normals:
        print("Press 'N' in the visualization window to toggle normal vector display")
    
    o3d.visualization.draw_geometries(geometries)
    
    if save_path is not None:
        o3d.io.write_point_cloud(save_path, point_cloud)


def find_closest_point(c, xyz):
    # c: [1, 3], xyz: [N, 3]
    dist = np.linalg.norm(xyz - c, axis=-1)
    return xyz[np.argmin(dist)][None] # [1, 3]


def storePly(path, xyz, rgb, normal=None):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    if normal is None:
        normals = np.zeros_like(xyz)
    else:
        normals = normal

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def gen_pcd_cano(path, K, poses, rgbs, depths, masks=None, reprocess=False, cluster=False, visualize=False, eps=0.05, coord_trans=True):
    if not os.path.exists(path) or reprocess:
        if depths[0].shape != rgbs[0].shape[:2]:
            rgbs = np.stack([cv2.resize(rgb, (depths[0].shape[1], depths[0].shape[0]), interpolation=cv2.INTER_LINEAR) for rgb in rgbs], 0)
            if masks is not None:
                masks = np.stack([cv2.resize(mask, (rgbs[0].shape[1], rgbs[0].shape[0]), interpolation=cv2.INTER_NEAREST) for mask in masks], 0)
        poses = torch.from_numpy(poses).float().cuda()
        rgbs = torch.from_numpy(rgbs).float().cuda()
        depths = torch.from_numpy(depths).float().cuda()
        if masks is not None:
            masks = torch.from_numpy(masks).float().cuda()
        K = torch.from_numpy(K).float().cuda()
        if K.ndim == 2:
            K = K[None].repeat(len(poses), 1, 1)
        xyz, color, normal = compute_pcd_torch_batch(poses, K, rgbs, depths, masks, eps=eps, cluster=cluster, coord_trans=coord_trans)
        print("Saving point clouds to", path, xyz.shape[0], "points")
        storePly(path, xyz, color * 255, normal)

    if visualize:
        pcd = o3d.io.read_point_cloud(path)
        xyz = np.asarray(pcd.points, np.float32)
        color = np.asarray(pcd.colors, np.float32)
        normal = np.asarray(pcd.normals, np.float32)
        visualize_point_cloud(np.concatenate([xyz, color * 255, normal], -1), show_normals=True, normal_scale=0.05)


def gen_pcd_frames(scene, K, poses, rgbs, depths, masks=None, reprocess=False, cluster=False, visualize=False, eps=0.075, coord_trans=True):
    path = f'{scene}/pcd_frames.npz'
    scene_name = os.path.basename(scene)
    if not os.path.exists(path) or reprocess:
        if depths[0].shape != rgbs[0].shape[:2]:
            rgbs = np.stack([cv2.resize(rgb, (depths[0].shape[1], depths[0].shape[0]), interpolation=cv2.INTER_LINEAR) for rgb in rgbs], 0)
            if masks is not None:
                masks = np.stack([cv2.resize(mask, (rgbs[0].shape[1], rgbs[0].shape[0]), interpolation=cv2.INTER_NEAREST) for mask in masks], 0)
        poses = torch.from_numpy(poses).float().cuda()
        rgbs = torch.from_numpy(rgbs).float().cuda()
        depths = torch.from_numpy(depths).float().cuda()
        if masks is not None:
            masks = torch.from_numpy(masks).float().cuda()
        K = torch.from_numpy(K).float().cuda()
        if K.ndim == 2:
            K = K[None].repeat(len(poses), 1, 1)
        xyz, color, normal, start_idx = compute_pcd_frames_batch(poses, K, rgbs, depths, masks, eps=eps, cluster=cluster, coord_trans=coord_trans)
        print("Saving point clouds for", scene_name)
        np.savez(path, xyz=xyz, color=color, normal=normal, start_idx=start_idx)

    if visualize:
        data = np.load(path)
        xyz = data['xyz']
        color = data['color']
        normal = data['normal']
        visualize_point_cloud(np.concatenate([xyz, color * 255, normal], -1), show_normals=True, normal_scale=0.05)


def saveTransformFilesCanoMono(poses_s, poses_d, split, fov_x, fov_y, scene_path):
    transforms = {
            "camera_angle_x": fov_x,
            "camera_angle_y": fov_y,
            "frames": [],
        }
    with open(f'{scene_path}/transforms_{split}.json', 'w') as f:
        for i, pose in enumerate(poses_s):
            info = {
                "file_path": f'{split}/rgba/{str(i+1).zfill(4)}',
                "time": 0.,
                "transform_matrix": pose.tolist(),
            }
            transforms["frames"].append(info)
        for i, pose in enumerate(poses_d):
            info = {
                "file_path": f'{split}/rgba/{str(i+len(poses_s)+1).zfill(4)}',
                "time": i / len(poses_d),
                "transform_matrix": pose.tolist(),
            }
            transforms["frames"].append(info)
        json.dump(transforms, f, indent=4)


def saveTransformFilesCanoMono1(poses_s, poses_d, intrinsic_info, scene_path, suffix='png'):
    transforms = {
            "camera_angle_x": intrinsic_info['fov_x'],
            "camera_angle_y": intrinsic_info['fov_y'],
            "focal_x": intrinsic_info['focal_x'],
            "focal_y": intrinsic_info['focal_y'],
            "cx": intrinsic_info['cx'],
            "cy": intrinsic_info['cy'],
            "w": intrinsic_info['w'],
            "h": intrinsic_info['h'],
            "frames": [],
        }
    with open(f'{scene_path}/transforms.json', 'w') as f:
        for i, pose in enumerate(poses_s):
            info = {
                "file_path": f'images/{str(i).zfill(6)}.{suffix}',
                "state": 0,
                "time": 0.,
                "transform_matrix": pose.tolist(),
            }
            transforms["frames"].append(info)
        for i, pose in enumerate(poses_d):
            info = {
                "file_path": f'images/{str(i+len(poses_s)).zfill(6)}.{suffix}',
                "state": 1,
                "time": i / len(poses_d),
                "transform_matrix": pose.tolist(),
            }
            transforms["frames"].append(info)
        json.dump(transforms, f, indent=4)