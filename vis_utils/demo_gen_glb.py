import os
import json
import numpy as np
import bpy
import random


def clear_objects():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def reset(obj, rotation_euler=(0, 0, 0), location=(0, 0, 0)):
    obj.location = location
    obj.rotation_euler = rotation_euler
    obj.scale = (1, 1, 1)


def optimize_mesh(obj, decimate_ratio=0.1):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    modifier = obj.modifiers.new(name="Decimate", type='DECIMATE')
    modifier.ratio = decimate_ratio
    modifier.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier="Decimate")
    

def load_mesh_with_material(src_path, meta, optimize=False):
    for entry in meta:
        mesh_path = os.path.join(src_path, entry['visuals'][0])
        bpy.ops.wm.ply_import(filepath=mesh_path)
        
        # Get the imported object (assumes it's the active object)
        obj = bpy.context.active_object
        if optimize:
            optimize_mesh(obj)
        mesh = obj.data
        
        # Create a new material
        mat = bpy.data.materials.new(name=f"Material_{obj.name}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        
        # Clear default nodes
        nodes.clear()
        
        # Create necessary nodes
        node_vertexcolor = nodes.new(type='ShaderNodeVertexColor')
        node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        node_output = nodes.new(type='ShaderNodeOutputMaterial')
        
        # Connect nodes
        links = mat.node_tree.links
        links.new(node_vertexcolor.outputs['Color'], node_bsdf.inputs['Base Color'])
        links.new(node_bsdf.outputs['BSDF'], node_output.inputs['Surface'])
        
        # Assign material to object
        if len(obj.data.materials) == 0:
            obj.data.materials.append(mat)
        else:
            obj.data.materials[0] = mat
        
        # Enable vertex color display
        if len(mesh.vertex_colors) == 0:
            mesh.vertex_colors.new()


def load_joint_info(meta):
    joint_info = []
    for i, entry in enumerate(meta):
        if entry['joint'] == 'hinge':
            info = {
                "joint_type": 'revolute',
                'origin': np.array(entry['jointData']['axis']['origin']),
                'direction': np.array(entry['jointData']['axis']['direction'])
            }
        elif entry['joint'] == 'slider':
            info = {
                "joint_type": 'prismatic',
                'origin': np.array(entry['jointData']['axis']['origin']),
                'direction': np.array(entry['jointData']['axis']['direction'])
            }
        elif entry['joint'] == 'heavy':
            info = {} # root part
        else:
            raise ValueError(f"Unknown joint type: {entry['joint']}")
        joint_info.append(info)
    return joint_info


def R_from_direction_angle(direction, theta):
    # Normalize direction vector
    direction = np.array(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    # Rodriguez rotation formula
    K = np.array([[0, -direction[2], direction[1]],
                  [direction[2], 0, -direction[0]], 
                  [-direction[1], direction[0], 0]])
    
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R


def animate_joint(joint_info, obj_name, r_range=[-0.3, 0.3], p_range=[-0.05, 0.10], n_frames=100):
    # Set scene frame range
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = n_frames - 1
    
    # Get all parts except camera and light
    objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    
    # Animate other objects
    for i, (obj, info) in enumerate(zip(objects, joint_info)):
        print(i, obj, info)
        if not info:  # root part
            translation = obj.matrix_world.to_translation()
            rotation_euler = obj.rotation_euler
            rotation = np.array(rotation_euler.to_matrix())[:3, :3]
            print(translation, rotation)
            continue
        
        reset(obj)
        loc = np.array(obj.location)
        if info["joint_type"] == 'revolute':
            r1, r2 = random.random() / 2 + 0.5, random.random() / 2 + 0.5
            revolute_range = np.concatenate([
                np.linspace(np.pi * r_range[0] * r1, np.pi * r_range[1] * r2, n_frames // 2),
                np.linspace(np.pi * r_range[0] * r2, np.pi * r_range[1] * r1, n_frames // 2)[::-1],
            ])

            direction = info['direction']
            origin = info['origin']
#            direction = rotation @ direction
#            origin = rotation @ origin
            # Animate rotation
            if random.random() < 0.5:
                direction = -direction
            for frame in range(n_frames):
                bpy.context.scene.frame_set(frame)
                theta = revolute_range[frame]
                if 'storage_45503_start_1' in obj.name:
                    theta = -0.5 + theta / 2
                elif 'cabinet_3r_white_start_1' in obj.name:
                    theta = 0.7 + theta
                R = R_from_direction_angle(direction, theta)
                R1 = rotation @ R
                new_rotation = bpy.data.objects.new("", None).rotation_euler.to_matrix()
                for i in range(3):
                    for j in range(3):
                        new_rotation[i][j] = R1[i,j]
                obj.rotation_euler = new_rotation.to_euler()
                new_loc = rotation @ (-R @ origin + origin) + translation
                obj.location = new_loc
                obj.keyframe_insert(data_path='location')
                obj.keyframe_insert(data_path='rotation_euler')
                
        elif info["joint_type"] == 'prismatic':
            direction = info['direction']
            direction = rotation @ direction
            # Animate translation
            p1, p2 = random.random() / 2 + 0.5, random.random() / 2 + 0.5
            prismatic_range = np.concatenate([
                np.linspace(p_range[0] * p1, p_range[1] * p2, n_frames // 4),
                np.linspace(p_range[1] * p2, p_range[0], n_frames // 4),
                np.linspace(p_range[0], p_range[1] * p2, n_frames // 4),
                np.linspace(p_range[1] * p2, p_range[0] * p1, n_frames // 4),
            ])
            if random.random() < 0.5:
                direction = -direction
            for frame in range(n_frames):
                bpy.context.scene.frame_set(frame)
                theta = prismatic_range[frame]
                if 'table_25493_start_2' in obj.name:
                    theta = -0.1 + theta
                elif 'table_31249_start_1' in obj.name:
                    theta = -theta * 1.4 - 0.16
                elif 'storage_47648_start_2' in obj.name:
                    theta = theta - 0.05
                elif 'cabinet_1r2p_transparent_start_2' in obj.name:
                    theta = theta - 0.05
                t = direction * theta
                obj.location = loc + t + translation
                obj.keyframe_insert(data_path='location')

def export_glb(export_path):
    bpy.ops.export_scene.gltf(
        filepath=export_path,
        export_format='GLB',
        use_selection=False,
        export_lights=True,
        export_apply=True,
        export_draco_mesh_compression_enable=True,
        export_draco_mesh_compression_level=10
    )


root = '/mnt/fillipo/yuliu/wallbreaker/Projects/VideoArtGS/outputs/artgs'
# subset = 'sapien'
# src_dir = f'{root}/{subset}'
# scenes = ['100481_new', '101284_new', '101287_new', '101808_new', '101908_new', 
#           '103015_new', '103811_new', '10489_new', '10655_new', '1280_new', 
#           '168_new', '25493_new', '30666_new', '31249_new', '45194_new', 
#           '45503_new', '45612_new', '47648_new', '8961_new', '9016_new']
# scenes = ['100481']
# model_name = 'base_mi_nt'

subset = 'realscan'
src_dir = f'{root}/{subset}'
scenes = ['cab_1r_1p', 'coffeemachine_2r', 'mac_1r', 'chair_1r', 'microwave_1r', 'cab1']
scenes = ['chair_1r']
model_name = 'demo'

dst_root =  f'/mnt/fillipo/yuliu/wallbreaker/Projects/VideoArtGS/outputs/demo/{subset}'
glb_path = os.path.join(dst_root, 'glbs')
os.makedirs(glb_path, exist_ok=True)
# set seed
random.seed(0)
np.random.seed(0)

for scene in scenes:
    print(f'Processing {scene}')
#    clear_objects()
    src_path = os.path.join(src_dir, scene, model_name, 'train', 'ours_20000')
    meta = json.load(open(os.path.join(src_path, 'joint_info_aligned.json'), 'r'))

    # Load mesh with material
#    loaded = False
#    for obj in bpy.context.scene.objects:
#        if scene in obj.name:
#            loaded = True
#            break
#    if not loaded:
#        load_mesh_with_material(src_path, meta, optimize=True)
    # Load axis info
    joint_info = load_joint_info(meta)
    # Animate joint
    visualize_config = json.load(open(f'/mnt/fillipo/yuliu/wallbreaker/Projects/VideoArtGS/arguments/visualize_config.json'))[subset][scene]['joint_value']
    r_range = visualize_config['r_range']
    p_range = visualize_config['p_range']
    animate_joint(joint_info, scene, n_frames=100, r_range=r_range, p_range=p_range)
#    animate_joint(joint_info, scene, n_frames=100)
    # Export GLB
    export_path = os.path.join(glb_path, f'{scene}.glb')
    export_glb(export_path)
