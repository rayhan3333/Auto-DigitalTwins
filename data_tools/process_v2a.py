import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
import yaml
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from data_tools.process_utils import *
from utils.dual_quaternion import quaternion_to_matrix
from utils.geo_utils import xyzmap2depth_batch
from data_tools.v2a_data_utils import get_gt_mesh
from utils.other_utils import vis_depth


N_CANO = 24
H, W = 480, 640


def cal_pose(camera_pose, obj_pose):
    rot = quaternion_to_matrix(torch.from_numpy(camera_pose[3:]).float()).numpy()
    t = camera_pose[:3]
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = t
    pose = np.dot(obj_pose, pose)
    pose[:3, :3] = pose[:3, :3][:, [1, 2, 0]] * [[-1, 1, -1]]
    return pose


def cal_obj_pose(init_base_pose: np.ndarray) -> np.ndarray:
    rot = quaternion_to_matrix(torch.from_numpy(init_base_pose[3:]).float()).numpy()
    t = init_base_pose[:3]
    object2origin = np.eye(4)
    object2origin[:3, :3] = rot.T
    object2origin[:3, 3] = -np.dot(rot.T, t)

    # origin2label = np.array([[0, -1, 0, 0], 
    #                          [0, 0, 1, 0], 
    #                          [-1, 0, 0, 0], 
    #                          [0, 0, 0, 1]]) # coodinate transform
    # obj_pose = np.dot(origin2label, object2origin)
    obj_pose = object2origin
    return obj_pose


def process_cano_data(cano_path, obj_pose):
    # cano pcd
    cano_xyz_paths = os.listdir(os.path.join(cano_path, 'xyz'))
    assert len(cano_xyz_paths) == N_CANO, f'{len(cano_xyz_paths)} != {N_CANO}'
    cano_xyzs = []
    cano_colors = []
    cano_segs = []
    for i in range(len(cano_xyz_paths)):
        seg = np.load(f'{cano_path}/segment/{i:06d}.npz')['a'].reshape(-1) # [H*W]
        xyz = np.load(f'{cano_path}/xyz/{i:06d}.npz')['a'] # [H*W, 3]
        xyz = xyz @ obj_pose[:3, :3].T + obj_pose[:3, 3]
        color = np.array(Image.open(f'{cano_path}/rgb/{i:06d}.png').convert('RGB')).reshape(-1, 3) # [H*W, 3]
        cano_xyzs.append(xyz)
        cano_colors.append(color)
        cano_segs.append(seg)
    cano_xyzs = np.stack(cano_xyzs, axis=0)
    cano_colors = np.stack(cano_colors, axis=0)
    cano_segs = np.stack(cano_segs, axis=0)
    cano_masks = cano_segs > 5 # extracting foreground objects

    cano_pcd = o3d.geometry.PointCloud()
    cano_pcd.points = o3d.utility.Vector3dVector(cano_xyzs[cano_masks].reshape(-1, 3))
    cano_pcd.colors = o3d.utility.Vector3dVector(cano_colors[cano_masks].reshape(-1, 3) / 255)
    cano_pcd = cano_pcd.voxel_down_sample(voxel_size=0.05)
    cano_pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
    cano_poses = np.load(f'{cano_path}/camera_pose.npy')
    cano_poses = np.array([np.dot(obj_pose, p) for p in cano_poses])
    cano_depths = xyzmap2depth_batch(torch.from_numpy(cano_xyzs).float().cuda(), torch.from_numpy(cano_poses).float().cuda())
    cano_depths = cano_depths.reshape(N_CANO, H, W).cpu().numpy()
    return cano_pcd, cano_colors, cano_masks, cano_depths, cano_poses


def save_data(input_scene_path, output_scene_path, cano_colors, cano_masks, cano_depths, visualize_depth=False):
    os.makedirs(f'{output_scene_path}/rgb', exist_ok=True)
    os.makedirs(f'{output_scene_path}/depth', exist_ok=True)
    os.makedirs(f'{output_scene_path}/images', exist_ok=True)
    if visualize_depth:
        os.makedirs(f'{output_scene_path}/vis_depth', exist_ok=True)
    for i in range(N_CANO):
        rgb = cano_colors[i].reshape(H, W, 3).astype(np.uint8)
        img = Image.fromarray(rgb)
        img.save(f'{output_scene_path}/rgb/{i:06d}.jpg')
        depth_img = Image.fromarray((cano_depths[i] * 1000).astype(np.uint32))
        depth_img.save(f'{output_scene_path}/depth/{i:06d}.png')
        img_rgba = Image.fromarray(np.concatenate([rgb, cano_masks[i].reshape(H, W, 1).astype(np.uint8) * 255], axis=-1))
        img_rgba.save(f'{output_scene_path}/images/{i:06d}.png')
        if visualize_depth:
            vis_depth(cano_depths[i], f'{output_scene_path}/vis_depth/{i:06d}.png')

    rgb_files = os.listdir(f'{input_scene_path}/rgb')
    for j in range(len(rgb_files)):
        os.system(f'cp {input_scene_path}/rgb/{j:06d}.jpg {output_scene_path}/rgb/{j+N_CANO:06d}.jpg')
        depth = np.load(f'{input_scene_path}/depth/{j:06d}.npz')['a']
        if depth.max() < 1000: # some data is in meters and some is in mm
            depth = depth * 1000
        depth_img = Image.fromarray(depth.astype(np.uint32))
        depth_img.save(f'{output_scene_path}/depth/{j+N_CANO:06d}.png')
        img = np.array(Image.open(f'{input_scene_path}/rgb/{j:06d}.jpg').convert('RGB'))
        seg = np.load(f'{input_scene_path}/segment/{j:06d}.npz')['a'].reshape(-1) # [H*W]
        mask = (seg > 5).reshape(H, W, 1).astype(np.uint8) * 255 # extracting foreground objects
        img_rgba = Image.fromarray(np.concatenate([img, mask], axis=-1))
        img_rgba.save(f'{output_scene_path}/images/{j+N_CANO:06d}.png')
        if visualize_depth:
            vis_depth(depth, f'{output_scene_path}/vis_depth/{j+N_CANO:06d}.png')


def remove_vertices_below_ground(mesh, obj_pose):
    # remove vertices below ground because they are occluded by the ground in v2a data
    vertices = np.asarray(mesh.vertices)
    plane_normal = np.array([0, 0, 1])
    plane_origin = np.array([0, 0, obj_pose[2, 3]])
    if vertices.min(axis=0)[2] < plane_origin[2]:
        clipped_mesh = mesh.slice_plane(plane_origin, plane_normal)
    else:
        clipped_mesh = mesh
    return clipped_mesh


def process_gt_data(obj_name, joint_name, joint_data_dir, output_scene_path, obj_pose):
    os.makedirs(f'{output_scene_path}/gt', exist_ok=True)

    # gt mesh
    partnet_data_path = "/mnt/fillipo/Datasets/partnet-mobility-v0/dataset"
    urdf_path = Path(f"{partnet_data_path}/{obj_name}/mobility.urdf").expanduser().resolve()
    urdf_dir = urdf_path.parent
    joint_id = int(joint_name.split('_')[1])
    full_mesh, moving_mesh, static_mesh = get_gt_mesh(urdf_path, urdf_dir, joint_id, joint_data_dir)
    # remove vertices below ground
    full_mesh = remove_vertices_below_ground(full_mesh, obj_pose)
    moving_mesh = remove_vertices_below_ground(moving_mesh, obj_pose)
    static_mesh = remove_vertices_below_ground(static_mesh, obj_pose)
    full_mesh.export(f'{output_scene_path}/gt/whole_mesh.ply')
    moving_mesh.export(f'{output_scene_path}/gt/part_1.ply')
    static_mesh.export(f'{output_scene_path}/gt/part_0.ply')

    # gt joint info
    meta_data = json.load(open(f"{partnet_data_path}/{obj_name}/mobility_v2.json"))
    gt_joint_info = []
    for entry in meta_data:
        if entry['id'] == joint_id:
            gt_joint_info.append(entry)
            break
    gt_file_path = f'{output_scene_path}/gt/mobility_v2.json'
    with open(gt_file_path, 'w') as f:
        json.dump(gt_joint_info, f, indent=4)

    # other gt data
    files = ['actor_pose.pkl', 'gt_joint_value.npy', 'joint_id_list.txt', 'meta.json', 'qpos.npy']
    for file in files:
        os.system(f'cp {joint_data_dir}/{file} {output_scene_path}/gt/{file}')


def process_camera_data(input_scene_path, output_scene_path, obj_pose):
    intrinsics = np.load(f'{input_scene_path}/intrinsics.npy')
    focal_x, focal_y = intrinsics[0, 0], intrinsics[1, 1]
    fov_x = focal2fov(focal_x, W)
    fov_y = focal2fov(focal_y, H)
    pose_data = np.load(f'{input_scene_path}/camera_pose.npy')
    poses = []
    for p in pose_data:
        pose = cal_pose(p, obj_pose)
        poses.append(pose)
    poses = np.array(poses)
    intrinsic_info = {'focal_x': focal_x.item(), 'focal_y': focal_y.item(), 'fov_x': fov_x, 'fov_y': fov_y, 
                        'cx': intrinsics[0, 2].item(), 'cy': intrinsics[1, 2].item(), 'w': W, 'h': H}
    return poses, intrinsic_info


def process_scene(class_name, obj_name, joint_name, view_name, v2a_data_path, output_data_path):
    scene_path = f'{v2a_data_path}/{class_name}/{obj_name}'
    joint_data_dir = f'{scene_path}/{joint_name}'
    input_scene_path = f'{scene_path}/{joint_name}/{view_name}'
    output_obj_name = f'{obj_name}_{joint_name}_{view_name}'
    output_scene_path = f'{output_data_path}/{output_obj_name}'
    os.makedirs(output_scene_path, exist_ok=True)

    # obj pose
    obj_pose_dict = pickle.load(open(f'{joint_data_dir}/actor_pose.pkl', 'rb'))
    init_base_pose = obj_pose_dict["actor_6"][0]
    obj_pose = cal_obj_pose(init_base_pose)

    # cano data
    cano_path = f'{joint_data_dir}/view_init'
    cano_pcd, cano_colors, cano_masks, cano_depths, cano_poses = process_cano_data(cano_path, obj_pose)
    o3d.io.write_point_cloud(f'{output_scene_path}/point_cloud.ply', cano_pcd)
    
    # gt data
    if not os.path.exists(f'{output_scene_path}/gt'):
        process_gt_data(obj_name, joint_name, joint_data_dir, output_scene_path, obj_pose)
        
    # rgb, depth, mask, fg_img
    save_data(input_scene_path, output_scene_path, 
              cano_colors, cano_masks, cano_depths, visualize_depth=True)

    # camera data
    poses, intrinsic_info = process_camera_data(input_scene_path, output_scene_path, obj_pose)
    saveTransformFilesCanoMono1(cano_poses, poses, intrinsic_info, output_scene_path)

            

if __name__ == '__main__':
    v2a_data_path = "/mnt/fillipo/yuliu/video2articulation"
    with open(f'{v2a_data_path}/partnet_mobility_data_split.yaml', 'r') as f:
        split_dict = yaml.safe_load(f)
    scenes = split_dict['test']
    data_path = f'./data/v2a/sapien'
    os.makedirs(data_path, exist_ok=True)
    v2a_data_path = f'{v2a_data_path}/sim_data/partnet_mobility'
    
    for scene in tqdm(scenes):
        info = scene.split('/')
        class_name, obj_name, joint_name, view_name = info[2], info[3], info[4], info[5]
        process_scene(class_name, obj_name, joint_name, view_name, v2a_data_path, data_path)

            