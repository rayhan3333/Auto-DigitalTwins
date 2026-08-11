import torch
import os
import trimesh
from pytorch3d.ops import knn_points
from glob import glob
import numpy as np
import json
from utils.dual_quaternion import *
from scene.deform_model import DeformModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from utils.metrics import *
import pandas as pd


def find_nearest_point(x, y):
    # x: [N, 3], y: [M, 3]
    dists, idx, _ = knn_points(x[None], y[None], K=1)
    return idx.squeeze()


def cal_track_metrics(pred_track, gt_track):
    # pred_track: [T, N, 3], gt_track: [T, N, 3]
    epe = torch.norm(pred_track - gt_track, dim=-1)
    delta_05 = (epe < 0.05).float().mean() * 100
    delta_10 = (epe < 0.1).float().mean() * 100
    return epe.mean(), delta_05, delta_10


def get_gt_mask(gt_path):
    part_meshes = sorted(glob(os.path.join(gt_path, "part_*.ply")))
    gt_points = []
    gt_mask = []
    for i, part_mesh in enumerate(part_meshes):
        mesh = trimesh.load(part_mesh)
        gt_points.append(torch.from_numpy(np.asarray(mesh.vertices)).float())
        gt_mask.append((torch.ones(mesh.vertices.shape[0]) * i).long())
    gt_points = torch.cat(gt_points, dim=0).cuda()
    gt_mask = torch.cat(gt_mask, dim=0).cuda()
    return gt_points, gt_mask


def cal_query_mask(query_points, gt_points, gt_mask):
    # query_points: [N, 3], gt_points: [M, 3], gt_mask: [M]
    idx = find_nearest_point(query_points, gt_points)
    return gt_mask[idx] # [N]


def read_state_sequence(state_sequence_path):
    with open(state_sequence_path, 'r') as f:
        info = json.load(f)
    n_canonical = 150
    num_frames = len(info[0]['part_move']) - n_canonical
    state_sequence_dict = {}
    state_sequence = []
    idx_list = []
    for item in info:
        sequence = np.array(list(item['part_move'].values())) # [num_frames, 3] 3 dim, only 1 dim has value and others are 0
        sequence = sequence.sum(-1) # [num_frames]
        sequence -= sequence[0]
        idx = item['idx']
        idx_list.append(idx)
        state_sequence_dict[idx] = sequence[n_canonical:]
    for idx in sorted(idx_list):
        state_sequence.append(state_sequence_dict[idx])
    return torch.from_numpy(np.stack(state_sequence, axis=1)).float().cuda() # [num_frames, num_joints]


def joint_info_to_dq(gt_joint_list, state_sequence):
    # gt_joint_list: list of joint info, each item is a dict with 'origin', 'direction'
    # state_sequence: [num_frames, num_joints] num_parts = num_joints + 1
    T = state_sequence.shape[0]
    qr_base, qd_base = torch.zeros(T, 4).to(state_sequence.device), torch.zeros(T, 4).to(state_sequence.device)
    qr_base[:, 0] = 1
    qrs, qds = [qr_base], [qd_base]
    for i, joint_info in enumerate(gt_joint_list):
        direction = torch.from_numpy(np.array(joint_info['direction'])).float()[None].to(state_sequence.device) # [1, 3]
        origin = torch.from_numpy(np.array(joint_info['origin'])).float()[None].to(state_sequence.device) # [1, 3]
        theta = state_sequence[:, i:i+1] # [T, 1]
        if joint_info['joint_type'] == 'p':
            qr = qr_base.clone()
            t0 = torch.cat([torch.zeros(T, 1).to(qr.device), direction * theta], dim=-1) # [T, 4]
            qd = 0.5 * quaternion_mul(t0, qr)
        else:
            qr = axis2qr(direction, theta)
            t0 = torch.cat([torch.zeros(T, 1).to(qr.device), origin.repeat(T, 1)], dim=-1)
            qd = 0.5 * (quaternion_mul(t0, qr) - quaternion_mul(qr, t0))
        qrs.append(qr)
        qds.append(qd)
    qrs = torch.stack(qrs, dim=1) # [T, num_parts, 4]
    qds = torch.stack(qds, dim=1) # [T, num_parts, 4]
    return qrs, qds


def slotdq_to_gsdq_batch(slot_qr, slot_qd, mask):
    # slot_qr: [B, K, 4], slot_qd: [B, K, 4], mask: [B, N, K]
    qr = torch.einsum('bnk, bkl->bnl', mask, slot_qr)   # [B, N, 4]
    qd = torch.einsum('bnk, bkl->bnl', mask, slot_qd)   # [B, N, 4]
    
    return normalize_dualquaternion(qr, qd)


def deform_pts_batch(xc, mask, slot_qr, slot_qd):
    # xc: [B, N, 3], mask: [B, N, K], slot_qr: [B, K, 4], slot_qd: [B, K, 4]
    gs_qr, gs_qd = slotdq_to_gsdq_batch(slot_qr, slot_qd, mask) # [B, N, 4]
    xt = dual_quaternion_apply((gs_qr, gs_qd), xc) # [B, N, 3]
    return xt


def cal_gt_track(query_points, args):
    # query_points: [N, 3]
    gt_path = os.path.join(args.source_path, "gt")
    gt_points, gt_mask = get_gt_mask(gt_path)
    query_mask = cal_query_mask(query_points, gt_points, gt_mask) # [N]
    gt_joint_list = read_gt(f'{gt_path}/mobility_v2.json')
    gt_joint_list = sorted(gt_joint_list, key=lambda x: x['idx'])
    state_path = args.source_path.replace('sapien', 'new_version_1114')
    state_sequence = read_state_sequence(f'{state_path}/part_info.json')

    qr, qd = joint_info_to_dq(gt_joint_list, state_sequence) # [T, num_parts, 4]
    query_mask = F.one_hot(query_mask, num_classes=qr.shape[1]).float()[None].repeat(qr.shape[0], 1, 1) # [T, N, num_parts]
    xt = deform_pts_batch(query_points[None].repeat(qr.shape[0], 1, 1), query_mask, qr, qd) # [T, N, 3]
    return xt


def cal_pred_track(query_points, num_frames, deform_model: DeformModel):
    # query_points: [N, 3]
    t = torch.arange(num_frames, device=query_points.device)[:, None] / num_frames # [T, 1]
    qr, qd, _ = deform_model.deform.get_slot_deform(t) # [T, num_parts, 4]
    mask = deform_model.deform.get_mask(query_points, is_training=False) # [N, K]
    mask = mask[None].repeat(num_frames, 1, 1) # [T, N, K]
    xt = deform_pts_batch(query_points[None].repeat(num_frames, 1, 1), mask, qr, qd) # [T, N, 3]
    return xt


def eval_track(args):
    joint_infos = read_gt(f'{args.source_path}/gt/mobility_v2.json')
    nq = 4 + len(joint_infos) // 2
    tapip3d_path = os.path.join(args.source_path, f"{args.scene_name}.n{nq}.npz")
    tapip3d_data = np.load(tapip3d_path)
    tapip3d_track = torch.from_numpy(tapip3d_data['coords']).float().cuda()
    new_data = {}
    for key in tapip3d_data.keys():
        new_data[key] = tapip3d_data[key]   
    query_points = tapip3d_track[0]
    num_frames = tapip3d_track.shape[0]
    deform_model = DeformModel(args)
    loaded = deform_model.load_weights(args.model_path, iteration=20000)
    if not loaded:
        raise ValueError(f"Failed to load weights from {args.model_path}")
    deform_model.update(20000)

    gt_track = cal_gt_track(query_points, args)

    metrics = {}

    epe, delta_05, delta_10 = cal_track_metrics(tapip3d_track, gt_track)
    metrics['Tapip3d_raw_EPE'] = [epe.item()]
    metrics['Tapip3d_raw_Delta 0.05'] = [delta_05.item()]
    metrics['Tapip3d_raw_Delta 0.1'] = [delta_10.item()]
    print(f"Tapip3d_raw: EPE: {epe.item():.4f}, Delta 0.05: {delta_05.item():.4f}, Delta 0.1: {delta_10.item():.4f}")

    tapip3d_path = os.path.join(args.source_path, f"filtered.npz")
    tapip3d_data = np.load(tapip3d_path)
    tapip3d_track = torch.from_numpy(tapip3d_data['coords']).float().cuda()

    query_points = tapip3d_track[0]
    pred_track = cal_pred_track(query_points, num_frames, deform_model)
    gt_track = cal_gt_track(query_points, args)
    new_data['coords'] = gt_track.detach().cpu().numpy()
    np.savez(os.path.join(args.model_path, f'train/ours_{args.iteration}/tapip3d_data_gt.npz'), **new_data)
    new_data['coords'] = pred_track.detach().cpu().numpy()
    np.savez(os.path.join(args.model_path, f'train/ours_{args.iteration}/tapip3d_data_pred.npz'), **new_data)

    epe, delta_05, delta_10 = cal_track_metrics(tapip3d_track, gt_track)
    metrics['Tapip3d_filtered_EPE'] = [epe.item()]
    metrics['Tapip3d_filtered_Delta 0.05'] = [delta_05.item()]
    metrics['Tapip3d_filtered_Delta 0.1'] = [delta_10.item()]
    print(f"Tapip3d_filtered: EPE: {epe.item():.4f}, Delta 0.05: {delta_05.item():.4f}, Delta 0.1: {delta_10.item():.4f}")

    epe, delta_05, delta_10 = cal_track_metrics(pred_track, gt_track)
    metrics['Ours_EPE'] = [epe.item()]
    metrics['Ours_Delta 0.05'] = [delta_05.item()]
    metrics['Ours_Delta 0.1'] = [delta_10.item()]   
    print(f"Ours: EPE: {epe.item():.4f}, Delta 0.05: {delta_05.item():.4f}, Delta 0.1: {delta_10.item():.4f}")

    # save track_metrics
    df = pd.DataFrame(metrics).transpose()
    df.index.name = 'Metric'
    df.to_csv(os.path.join(args.model_path, f'train/ours_{args.iteration}/track_metrics.csv'), index=True, header=['Value'])


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mode", default='recon', choices=['render', 'recon', 'mask'])
    args = get_combined_args(parser)

    seed_everything(0)
    model.extract(args)
    pipeline.extract(args)
    eval_track(args)