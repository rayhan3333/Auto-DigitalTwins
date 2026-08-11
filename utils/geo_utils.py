# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# copy from https://github.com/NVlabs/DigitalTwinArt/tree/master, modified by Yu Liu

import open3d as o3d
import numpy as np
import os
import joblib
import torch
from sklearn.cluster import DBSCAN
from pytorch3d.ops import knn_points

os.environ['PYOPENGL_PLATFORM'] = 'egl'


glcam_in_cvcam = np.array([[1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, -1, 0],
                           [0, 0, 0, 1]])
# glcam_in_cvcam = np.eye(4)


def toOpen3dCloud(points, colors=None, normals=None):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        if colors.max() > 1:
            colors = colors / 255.0
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    return cloud


def depth2xyzmap(depth, K):
    H, W = depth.shape[:2]
    vs, us = np.meshgrid(np.arange(0, H), np.arange(0, W), sparse=False, indexing='ij')
    vs = vs.reshape(-1)
    us = us.reshape(-1)
    zs = depth.reshape(-1)
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    pts = np.stack((xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)), 1)  # (N,3)
    xyz_map = pts.reshape(H, W, 3).astype(np.float32)
    return xyz_map.astype(np.float32)


def depth2xyzmap_torch(depth, K):
    # depth: (H, W), K: (3, 3)
    H, W = depth.shape[:2]
    vs, us = torch.meshgrid(torch.arange(0, H), torch.arange(0, W), indexing='ij')
    vs = vs.reshape(-1).to(depth.device)
    us = us.reshape(-1).to(depth.device)
    zs = depth.reshape(-1)
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    pts = torch.stack((xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)), 1)  # (N,3)
    xyz_map = pts.reshape(H, W, 3)
    return xyz_map


def depth2xyzmap_batch(depth, K):
    # depth: (B, H, W), K: (B, 3, 3)
    B, H, W = depth.shape
    vs, us = torch.meshgrid(torch.arange(0, H), torch.arange(0, W), indexing='ij')
    vs = vs.reshape(1, -1).to(depth.device).repeat(B, 1)
    us = us.reshape(1, -1).to(depth.device).repeat(B, 1)
    zs = depth.reshape(B, -1)
    xs = (us - K[:, 0, 2][..., None]) * zs / K[:, 0, 0][..., None] 
    ys = (vs - K[:, 1, 2][..., None]) * zs / K[:, 1, 1][..., None]
    pts = torch.stack((xs.reshape(B, -1), ys.reshape(B, -1), zs.reshape(B, -1)), -1)  # (B, N, 3)
    xyz_map = pts.reshape(B, H, W, 3)
    return xyz_map


def unproject_from_depthmap_torch(c2w, K, depth):
    mask_depth = depth > 0.1
    H, W = depth.shape
    vs, us = torch.meshgrid(torch.arange(0, H), torch.arange(0, W), indexing='ij')
    vs = vs.reshape(-1).to(depth.device)
    us = us.reshape(-1).to(depth.device)
    zs = depth.reshape(-1)
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    pts = torch.stack((xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)), 1)  # (N,3)
    xyz_map = pts.reshape(H, W, 3).float()
    xyz_map[~mask_depth] = 0
    xyz_map = torch.cat((xyz_map, torch.ones_like(xyz_map[:, :, :1])), -1)  # (H, W, 4)
    xyz_map = xyz_map @ c2w.T  # (H, W, 4)
    return xyz_map[..., :3], mask_depth


def find_biggest_cluster(pts, eps=0.005, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    dbscan.fit(pts)
    ids, cnts = np.unique(dbscan.labels_, return_counts=True)
    best_id = ids[cnts.argsort()[-1]]
    keep_mask = dbscan.labels_ == best_id
    return keep_mask


def xyzmap2depth_batch(xyzs, poses):
    # xyzs in world coordinate: (B, N, 3), poses: (B, 4, 4)
    B, N, _ = xyzs.shape
    T = torch.from_numpy(glcam_in_cvcam).float().cuda()[None].repeat(B, 1, 1)
    c2w = poses @ T
    xyzs = (xyzs - c2w[:, :3, 3].unsqueeze(1)) @ c2w[:, :3, :3]
    depths = xyzs[:, :, 2]
    return depths # (B, N)


def compute_pcd_worker(K, glcam_in_world, rgb, depth, mask=None):
    xyz_map = depth2xyzmap(depth, K)
    valid = depth >= 0.1
    if mask is not None:
        valid = valid & (mask > 0)
    pts = xyz_map[valid].reshape(-1, 3)
    if len(pts) == 0:
        return None
    colors = rgb[valid].reshape(-1, 3)

    pcd = toOpen3dCloud(pts, colors)

    pcd = pcd.voxel_down_sample(0.01)
    new_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
    cam_in_world = glcam_in_world @ glcam_in_cvcam
    new_pcd.transform(cam_in_world)

    return np.asarray(new_pcd.points).copy(), np.asarray(new_pcd.colors).copy()


def compute_pcd(glcam_in_worlds, K, rgbs=None, depths=None,
                         masks=None, cluster=False, eps=0.075, min_samples=5):

    args = []
    for i in range(len(rgbs)):
        args.append((K, glcam_in_worlds[i], rgbs[i], depths[i], masks[i]))

    ret = joblib.Parallel(n_jobs=6, prefer="threads")(joblib.delayed(compute_pcd_worker)(*arg) for arg in args)

    pcd_all = None
    for r in ret:
        if r is None or len(r[0]) == 0:
            continue
        if pcd_all is None:
            pcd_all = toOpen3dCloud(r[0], r[1])
        else:
            pcd_all += toOpen3dCloud(r[0], r[1])

    pcd = pcd_all.voxel_down_sample(eps / 5)
    pts = np.asarray(pcd.points).copy()
    if cluster:
        keep_mask = find_biggest_cluster(pts, eps, min_samples)
    else:
        keep_mask = np.ones((len(pts)), dtype=bool)
    return pts[keep_mask], np.asarray(pcd.colors)[keep_mask]


def depthpts_to_normal(points):
    # points: (H, W, 3), torch.Tensor
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output # (H, W, 3), torch.Tensor


def depthpts_to_normal_batch(points):
    # points: (B, H, W, 3), torch.Tensor
    output = torch.zeros_like(points)
    dx = torch.cat([points[:, 2:, 1:-1] - points[:, :-2, 1:-1]], dim=1)
    dy = torch.cat([points[:, 1:-1, 2:] - points[:, 1:-1, :-2]], dim=2)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[:, 1:-1, 1:-1, :] = normal_map
    return output # (B, H, W, 3), torch.Tensor


def compute_pcd_torch(K, pose, rgb, depth, mask=None):
    xyz_map = depth2xyzmap_torch(depth, K)
    # coordinate transform: blender -> colmap
    c2w = pose @ glcam_in_cvcam
    xyz_map = xyz_map @ c2w[:3, :3].T + c2w[:3, 3]
    normal_map = depthpts_to_normal(xyz_map)
    
    valid = depth > 0.1
    if mask is not None:
        valid = valid & (mask > 0)
    # remove the points on the border because the normal is zero
    valid[:, 0] = False
    valid[:, -1] = False
    valid[0] = False
    valid[-1] = False
    
    pts = xyz_map[valid].reshape(-1, 3)
    colors = rgb[valid].reshape(-1, 3)
    normals = normal_map[valid].reshape(-1, 3)
    return pts, colors, normals


def compute_pcd_torch_batch(c2w, K, rgbs=None, depths=None,
                         masks=None, eps=0.075, cluster=False, min_samples=5, coord_trans=True):

    B, H, W = depths.shape
    xyz_map = depth2xyzmap_batch(depths, K) # (B, H, W, 3)
    # xyz_map = torch.stack([depth2xyzmap_torch(depth, K[0]) for depth in depths], 0)

    if coord_trans:
        T = torch.from_numpy(glcam_in_cvcam).float().cuda()[None].repeat(B, 1, 1)
        c2w = c2w @ T
    xyz_map = xyz_map.reshape(B, -1, 3) @ c2w[:, :3, :3].transpose(1, 2) + c2w[:, :3, 3].unsqueeze(1)
    xyz_map = xyz_map.reshape(B, H, W, 3)
    normal_map = depthpts_to_normal_batch(xyz_map)
    valid = depths > 0.01
    if masks is not None:
        valid = valid & (masks > 0)
    # remove the points on the border because the normal is zero
    valid[:, :, 0] = False
    valid[:, :, -1] = False
    valid[:, 0, :] = False
    valid[:, -1, :] = False

    pts = xyz_map[valid].reshape(-1, 3).cpu().numpy()
    normals = normal_map[valid].reshape(-1, 3).cpu().numpy()
    colors = rgbs[valid].reshape(-1, 3).cpu().numpy()

    pcd = toOpen3dCloud(pts, colors, normals)
    pcd = pcd.voxel_down_sample(eps / 5)
    if cluster:
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
    pts = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    normals = np.asarray(pcd.normals)
    if cluster:
        keep_mask = find_biggest_cluster(pts, eps, min_samples)
    else:
        keep_mask = np.ones((len(pts)), dtype=bool)
    return pts[keep_mask], colors[keep_mask], normals[keep_mask]


def compute_pcd_frames_batch(c2w, K, rgbs=None, depths=None,
                         masks=None, eps=0.075, cluster=False, min_samples=5, coord_trans=True):

    B, H, W = depths.shape
    xyz_map = depth2xyzmap_batch(depths, K) # (B, H, W, 3)
    # xyz_map = torch.stack([depth2xyzmap_torch(depth, K[0]) for depth in depths], 0)

    if coord_trans:
        T = torch.from_numpy(glcam_in_cvcam).float().cuda()[None].repeat(B, 1, 1)
        c2w = c2w @ T
    xyz_map = xyz_map.reshape(B, -1, 3) @ c2w[:, :3, :3].transpose(1, 2) + c2w[:, :3, 3].unsqueeze(1)
    xyz_map = xyz_map.reshape(B, H, W, 3)
    normal_map = depthpts_to_normal_batch(xyz_map)
    valid = depths > 0.01
    if masks is not None:
        valid = valid & (masks > 0)
    # remove the points on the border because the normal is zero
    valid[:, :, 0] = False
    valid[:, :, -1] = False
    valid[:, 0, :] = False
    valid[:, -1, :] = False

    pts_all = []
    colors_all = []
    normals_all = []
    start_idx = [0]
    for i in range(B):
        pts = xyz_map[i, valid[i]].reshape(-1, 3).cpu().numpy()
        normals = normal_map[i, valid[i]].reshape(-1, 3).cpu().numpy()
        colors = rgbs[i, valid[i]].reshape(-1, 3).cpu().numpy()

        pcd = toOpen3dCloud(pts, colors, normals)
        pcd = pcd.voxel_down_sample(eps / 5)
        # pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
        pts = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        normals = np.asarray(pcd.normals)
        if cluster:
            keep_mask = find_biggest_cluster(pts, eps, min_samples)
        else:
            keep_mask = np.ones((len(pts)), dtype=bool)
        pts_all.append(pts[keep_mask])
        colors_all.append(colors[keep_mask])
        normals_all.append(normals[keep_mask])
        start_idx.append(start_idx[-1] + len(pts_all[-1]))
    pts_all = np.concatenate(pts_all, 0)
    colors_all = np.concatenate(colors_all, 0)
    normals_all = np.concatenate(normals_all, 0)
    start_idx = np.array(start_idx)
    return pts_all, colors_all, normals_all, start_idx


def compute_pcd_frames_v2a(poses, K, rgbs, depths, bs=64):
    B, H, W = depths.shape
    pcd_list = []
    for i in range(0, B, bs):
        poses_batch = poses[i:i+bs]
        rgb_batch = rgbs[i:i+bs]
        depth_batch = depths[i:i+bs]
        K_batch = K[i:i+bs]
        b = len(rgb_batch)
        xyz_map = depth2xyzmap_batch(depth_batch, K_batch) # (B, H, W, 3)
        c2w = poses_batch @ torch.from_numpy(glcam_in_cvcam).float().cuda()[None].repeat(b, 1, 1)
        xyz_map = xyz_map.reshape(b, -1, 3) @ c2w[:, :3, :3].transpose(1, 2) + c2w[:, :3, 3].unsqueeze(1)
        xyz_map = xyz_map.reshape(b, H, W, 3)
        valid = depth_batch > 0.01
        for j in range(len(rgb_batch)):
            xyz = (xyz_map[j, valid[j]]).cpu().numpy()
            colors = (rgb_batch[j, valid[j]]).cpu().numpy()
            pcd_list.append(np.concatenate([xyz, colors], -1))    
    return pcd_list


def compute_xyz_frames_v2a(poses, K, depths, bs=64):
    B, H, W = depths.shape
    xyz_list = []
    for i in range(0, B, bs):
        poses_batch = poses[i:i+bs]
        depth_batch = depths[i:i+bs]
        K_batch = K[i:i+bs]
        b = len(depth_batch)
        xyz_map = depth2xyzmap_batch(depth_batch, K_batch) # (B, H, W, 3)
        c2w = poses_batch @ torch.from_numpy(glcam_in_cvcam).float().cuda()[None].repeat(b, 1, 1)
        xyz_map = xyz_map.reshape(b, -1, 3) @ c2w[:, :3, :3].transpose(1, 2) + c2w[:, :3, 3].unsqueeze(1)
        xyz_map = xyz_map.reshape(b, H, W, 3)
        xyz_list.append(xyz_map.cpu().numpy())   
    return np.concatenate(xyz_list, 0)


def compute_pcd_oneframe(glcam_in_world, K, rgb=None, depth=None,
                         mask=None, cluster=False, eps=0.075, min_samples=5):

    xyz_map = depth2xyzmap(depth, K)
    valid = depth >= 0.1
    if mask is not None:
        valid = valid & (mask > 0)
    pts = xyz_map[valid].reshape(-1, 3)
    if len(pts) == 0:
        return None
    colors = rgb[valid].reshape(-1, 3)

    pcd = toOpen3dCloud(pts, colors)
    pcd = pcd.voxel_down_sample(eps / 5)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
    cam_in_world = glcam_in_world @ glcam_in_cvcam
    pcd.transform(cam_in_world)
    pts = np.asarray(pcd.points).copy()
    if cluster:
        keep_mask = find_biggest_cluster(pts, eps, min_samples)
    else:
        keep_mask = np.ones((len(pts)), dtype=bool)
    return pts[keep_mask], np.asarray(pcd.colors)[keep_mask]


def sample_by_index(pts, colors, ind, masks=None):
    pts = pts[ind]
    colors = colors[ind]
    if masks is not None:
        masks = masks[ind]
    return pts, colors, masks


def voxel_down_sample(pts, colors, voxel_size=0.01, masks=None):
    pcd = toOpen3dCloud(pts, colors)
    pcd, _, inds = pcd.voxel_down_sample_and_trace(voxel_size=voxel_size, min_bound=pcd.get_min_bound(), max_bound=pcd.get_max_bound())
    ind = np.array([item[0] for item in inds]) # select the first point in each voxel
    pts, colors, masks = sample_by_index(pts, colors, ind, masks)
    return pts, colors, masks


def remove_statistical_outlier(pts, colors, masks=None, nb_neighbors=30, std_ratio=2.0):
    pcd = toOpen3dCloud(pts, colors)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pts, colors, masks = sample_by_index(pts, colors, ind, masks)
    return pts, colors, masks


def compute_pcd_with_mask_oneframe(pose, K, rgb=None, depth=None,
                         mask=None, cluster=False, eps=0.075, min_samples=5):
    # rgb: (H, W, 3)
    # depth: (H, W)
    # mask: (H, W) # id of the part
    xyz_map = depth2xyzmap(depth, K)
    valid = depth >= 0.1
    assert valid.sum() > 0, 'No valid points'
    pts = xyz_map[valid].reshape(-1, 3)
    colors = rgb[valid].reshape(-1, 3)
    mask = mask[valid].reshape(-1)
    pts, colors, mask = voxel_down_sample(pts, colors, voxel_size=eps / 5, masks=mask)
    pts, colors, mask = remove_statistical_outlier(pts, colors, mask, nb_neighbors=30, std_ratio=2.0)
    cam_in_world = pose @ glcam_in_cvcam
    pts = pts @ cam_in_world[:3, :3].T + cam_in_world[:3, 3]

    if cluster:
        keep_mask = find_biggest_cluster(pts, eps, min_samples)
        pts = pts[keep_mask]
        mask = mask[keep_mask]
        colors = colors[keep_mask]
    return pts, colors, mask # [N, 3], [N, 3], [N]


def compute_pcd_with_mask_worker(K, pose, rgb, depth, mask):
    xyz_map = depth2xyzmap(depth, K)
    valid = depth >= 0.1
    assert valid.sum() > 0, 'No valid points'
    pts = xyz_map[valid].reshape(-1, 3)
    colors = rgb[valid].reshape(-1, 3)
    mask = mask[valid].reshape(-1)
    pts, colors, mask = voxel_down_sample(pts, colors, voxel_size=0.01, masks=mask)
    pts, colors, mask = remove_statistical_outlier(pts, colors, mask, nb_neighbors=30, std_ratio=2.0)
    cam_in_world = pose @ glcam_in_cvcam
    pts = pts @ cam_in_world[:3, :3].T + cam_in_world[:3, 3]
    return pts, colors, mask # [N, 3], [N, 3], [N]


def compute_pcd_with_mask(poses, K, rgbs, depths, masks, eps=0.075, min_samples=5, cluster=False):
    args = []
    for i in range(len(rgbs)):
        args.append((K, poses[i], rgbs[i], depths[i], masks[i]))

    ret = joblib.Parallel(n_jobs=6, prefer="threads")(joblib.delayed(compute_pcd_with_mask_worker)(*arg) for arg in args)
    pts, colors, masks = [], [], []
    for r in ret:
        if r is None or len(r[0]) == 0:
            continue
        pts.append(r[0])
        colors.append(r[1])
        masks.append(r[2])
    pts = np.concatenate(pts, 0)
    colors = np.concatenate(colors, 0)
    masks = np.concatenate(masks, 0)
    xyz, colors, masks = voxel_down_sample(pts, colors, voxel_size=eps / 5, masks=masks)
    if cluster:
        keep_mask = find_biggest_cluster(xyz, eps, min_samples)
        xyz = xyz[keep_mask]
        colors = colors[keep_mask]
        masks = masks[keep_mask]
    return xyz, colors, masks # [N, 3], [N, 3], [N]


def find_nearest_points_knn(p, tgt):
    """
    p: (3) tensor
    tgt: (M, 3) tensor
    """
    dists, idx, _ = knn_points(p.reshape(1, 1, 3), tgt[None], K=1)
    return tgt[idx.squeeze()]
