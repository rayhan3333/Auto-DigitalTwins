import os
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


def export_urdf(input_dir, output_dir, scene):
    meta_info = json.load(open(os.path.join(input_dir, 'joint_info.json')))
    urdf = f'<?xml version="1.0"?>\n<robot name="{scene}">\n'
    urdf += '\t<link name="base"/>\n'
    for i, item in enumerate(meta_info):
        item["visuals"][0] = item["visuals"][0]
        link_name = f'link_{i}'
        urdf += f'\t<link name="{link_name}">\n'
        if item['parent'] != -1:
            joint_limit = item['jointData']['limit']
            urdf += f'\t\t<visual>\n'
            urdf += f'\t\t\t<origin xyz="{-item["jointData"]["axis"]["origin"][0]} {-item["jointData"]["axis"]["origin"][1]} {-item["jointData"]["axis"]["origin"][2]}"/>\n'
            urdf += f'\t\t\t<geometry>\n'
            urdf += f'\t\t\t\t<mesh filename="{item["visuals"][0]}" />\n'
            urdf += f'\t\t\t</geometry>\n'
            urdf += f'\t\t</visual>\n'
            urdf += f'\t\t<collision>\n'
            urdf += f'\t\t\t<origin xyz="{-item["jointData"]["axis"]["origin"][0]} {-item["jointData"]["axis"]["origin"][1]} {-item["jointData"]["axis"]["origin"][2]}"/>\n'
            urdf += f'\t\t\t<geometry>\n'
            urdf += f'\t\t\t\t<mesh filename="{item["visuals"][0]}" />\n'
            urdf += f'\t\t\t</geometry>\n'
            urdf += f'\t\t</collision>\n'
            urdf += f'\t</link>\n'
            joint_type = 'revolute' if item["joint"] == 'hinge' else 'prismatic'
            urdf += f'\t<joint name="{item["name"]}" type="{joint_type}" >\n'
            urdf += f'\t\t<origin xyz="{item["jointData"]["axis"]["origin"][0]} {item["jointData"]["axis"]["origin"][1]} {item["jointData"]["axis"]["origin"][2]}" rpy="0 0 0"/>\n'
            urdf += f'\t\t<axis xyz="{item["jointData"]["axis"]["direction"][0]} {item["jointData"]["axis"]["direction"][1]} {item["jointData"]["axis"]["direction"][2]}"/>\n'
            urdf += f'\t\t<parent link="link_{item["parent"]}"/>\n'
            urdf += f'\t\t<child link="{link_name}"/>\n'
            urdf += f'\t\t<limit lower="{joint_limit["lower"]}" upper="{joint_limit["upper"]}"/>\n'
            urdf += f'\t</joint>\n'
        else:
            urdf += f'\t\t<visual>\n'
            urdf += f'\t\t\t<origin xyz="0 0 0"/>\n'
            urdf += f'\t\t\t<geometry>\n'
            urdf += f'\t\t\t\t<mesh filename="{item["visuals"][0]}" />\n'
            urdf += f'\t\t\t</geometry>\n'
            urdf += f'\t\t</visual>\n'
            urdf += f'\t\t<collision>\n'
            urdf += f'\t\t\t<origin xyz="0 0 0"/>\n'
            urdf += f'\t\t\t<geometry>\n'
            urdf += f'\t\t\t\t<mesh filename="{item["visuals"][0]}" />\n'
            urdf += f'\t\t\t</geometry>\n'
            urdf += f'\t\t</collision>\n'
            urdf += f'\t</link>\n'
            urdf += f'\t<joint name="joint_{i}" type="fixed" >\n'
            urdf += f'\t\t<origin rpy="1.570796326794897 0 -1.570796326794897" xyz="0 0 0"/>\n'
            urdf += f'\t\t<parent link="base"/>\n'
            urdf += f'\t\t<child link="{link_name}"/>\n'
            urdf += f'\t</joint>\n'
    urdf += '</robot>'
    with open(os.path.join(output_dir, f'{scene}.urdf'), 'w') as f:
        f.write(urdf)


def export_urdf_glb(input_dir, output_dir, scene):
    meta_info = json.load(open(os.path.join(input_dir, 'joint_info.json')))
    urdf = f'<?xml version="1.0"?>\n<robot name="{scene}">\n'
    urdf += '\t<link name="base"/>\n'
    rot = R.from_euler('xyz', [-90, 0, 0], degrees=True).as_matrix()
    for i, item in enumerate(meta_info):
        link_name = f'link_{i}'
        urdf += f'\t<link name="{link_name}">\n'
        if item['parent'] != -1:
            joint_limit = item['jointData']['limit']
            origin = np.array(item["jointData"]["axis"]["origin"])
            origin = rot @ origin
            origin = origin.tolist()
            direction = np.array(item["jointData"]["axis"]["direction"])
            direction = rot @ direction
            direction = direction.tolist()
            urdf += f'\t\t<visual>\n'
            urdf += f'\t\t\t<origin xyz="{-origin[0]} {-origin[1]} {-origin[2]}"/>\n'
            urdf += f'\t\t\t<geometry>\n'
            urdf += f'\t\t\t\t<mesh filename="{item["visuals"][0]}" />\n'
            urdf += f'\t\t\t</geometry>\n'
            urdf += f'\t\t</visual>\n'
            urdf += f'\t\t<collision>\n'
            urdf += f'\t\t\t<origin xyz="{-origin[0]} {-origin[1]} {-origin[2]}"/>\n'
            urdf += f'\t\t\t<geometry>\n'
            urdf += f'\t\t\t\t<mesh filename="{item["visuals"][0]}" />\n'
            urdf += f'\t\t\t</geometry>\n'
            urdf += f'\t\t</collision>\n'
            urdf += f'\t</link>\n'
            joint_type = 'revolute' if item["joint"] == 'hinge' else 'prismatic'
            urdf += f'\t<joint name="{item["name"]}" type="{joint_type}" >\n'
            urdf += f'\t\t<origin xyz="{origin[0]} {origin[1]} {origin[2]}" rpy="0 0 0"/>\n'
            urdf += f'\t\t<axis xyz="{direction[0]} {direction[1]} {direction[2]}"/>\n'
            urdf += f'\t\t<parent link="link_{item["parent"]}"/>\n'
            urdf += f'\t\t<child link="{link_name}"/>\n'
            urdf += f'\t\t<limit lower="{joint_limit["lower"]}" upper="{joint_limit["upper"]}"/>\n'
            urdf += f'\t</joint>\n'
        else:
            urdf += f'\t\t<visual>\n'
            urdf += f'\t\t\t<origin xyz="0 0 0"/>\n'
            urdf += f'\t\t\t<geometry>\n'
            urdf += f'\t\t\t\t<mesh filename="{item["visuals"][0]}" />\n'
            urdf += f'\t\t\t</geometry>\n'
            urdf += f'\t\t</visual>\n'
            urdf += f'\t\t<collision>\n'
            urdf += f'\t\t\t<origin xyz="0 0 0"/>\n'
            urdf += f'\t\t\t<geometry>\n'
            urdf += f'\t\t\t\t<mesh filename="{item["visuals"][0]}" />\n'
            urdf += f'\t\t\t</geometry>\n'
            urdf += f'\t\t</collision>\n'
            urdf += f'\t</link>\n'
            urdf += f'\t<joint name="joint_{i}" type="fixed" >\n'
            urdf += f'\t\t<origin rpy="1.570796326794897 0 -1.570796326794897" xyz="0 0 0"/>\n'
            urdf += f'\t\t<parent link="base"/>\n'
            urdf += f'\t\t<child link="{link_name}"/>\n'
            urdf += f'\t</joint>\n'
    urdf += '</robot>'
    urdf = urdf.replace('.ply', '.glb')
    with open(os.path.join(output_dir, f'{scene}_glb.urdf'), 'w') as f:
        f.write(urdf)


dataset = 'artgs'
output_dir = f'outputs/demo'

subset = 'realscan'
model_name = 'demo'
scenes = ['cab1', 'chair_1r', 'coffeemachine_2r', 'microwave_1r', 'cab_1r_1p', 'mac_1r']
scenes = ['microwave1']

for s in tqdm(scenes):
    output_scene = os.path.join(output_dir, 'urdf', s)
    os.makedirs(output_scene, exist_ok=True)
    input_dir = f'outputs/{dataset}/{subset}/{s}/{model_name}/train/ours_20000'
    os.system(f'cp -r {input_dir}/meshes {output_scene}/')
    os.system(f'cp {input_dir}/joint_info.json {output_scene}/')
    os.system(f'cp {input_dir}/joint_value.npy {output_scene}/')
    export_urdf(output_scene, output_scene, s)
    export_urdf_glb(output_scene, output_scene, s)

# subset = 'sapien'
# model_name = 'base_mi_nt'
# scenes = [
#         '100481_new', '101284_new', '101287_new', '101808_new', '101908_new',
#         '103015_new', '103811_new', '10489_new', '10655_new', '1280_new',
#         '168_new', '25493_new', '30666_new', '31249_new', '45194_new',
#         '45503_new', '45612_new', '47648_new', '8961_new', '9016_new'
#     ]

# for s in tqdm(scenes):
#     output_scene = os.path.join(output_dir, 'urdf', s)
#     os.makedirs(output_scene, exist_ok=True)
#     input_dir = f'outputs/{dataset}/{subset}/{s}/{model_name}/train/ours_20000'
#     # os.system(f'cp -r {input_dir}/meshes {output_scene}/')
#     # os.system(f'cp {input_dir}/joint_info.json {output_scene}/')
#     # os.system(f'cp {input_dir}/joint_value.npy {output_scene}/')
#     export_urdf(output_scene, output_scene, s)
#     export_urdf_glb(output_scene, output_scene, s)