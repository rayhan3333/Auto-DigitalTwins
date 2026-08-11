import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
import json
import numpy as np
from PIL import Image
from data_tools.process_utils import *
from utils.other_utils import vis_depth


if __name__ == '__main__':
    data_path = f'./data/artgs/sapien'
    scenes = sorted(os.listdir(data_path))
    scene_names = [os.path.basename(s) for s in scenes if os.path.isdir(os.path.join(data_path, s)) if 'new' in s]
    IMAGE_DIR = 'images'
    DEPTH_DIR = 'depth'
    num_cano = 150
    visualize_depth = True
    for scene_name in scene_names:
        scene = f'{data_path}/{scene_name}'
        file = json.load(open(f'{scene}/camera.json'))
        K = np.array(file['K'])
        focal_x = K[0][0]
        focal_y = K[1][1]
        w, h = Image.open(f'{scene}/{IMAGE_DIR}/000000.png').size
        fov_x = focal2fov(focal_x, w)
        fov_y = focal2fov(focal_y, h)
        poses = np.array(list(file.values())[1:])
        intrinsic_info = {'focal_x': focal_x, 'focal_y': focal_y, 'fov_x': fov_x, 'fov_y': fov_y, 
                          'cx': K[0, 2], 'cy': K[1, 2], 'w': w, 'h': h}
        saveTransformFilesCanoMono1(poses[:num_cano], poses[num_cano:], intrinsic_info, scene)

        rgbs = [np.array(Image.open(f'{scene}/{IMAGE_DIR}/{i:06d}.png')) for i in range(len(poses))]
        rgbs = np.array([rgb[..., :3] for rgb in rgbs])
        depths = np.array([np.array(Image.open(f'{scene}/{DEPTH_DIR}/{i:06d}.png')) / 1000 for i in range(len(poses))])
        gen_pcd_cano(f'{scene}/point_cloud.ply', K, poses[:num_cano], rgbs[:num_cano], depths[:num_cano], reprocess=True, cluster=False, visualize=False)
        if visualize_depth and not os.path.exists(f'{scene}/vis_depth'):
            os.makedirs(f'{scene}/vis_depth', exist_ok=True)
            for i in range(len(depths)):
                vis_depth(depths[i], f'{scene}/vis_depth/{i:06d}.png')