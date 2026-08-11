import os
import bpy
from glob import glob


def clear_objects():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def export_glb(export_path):
    bpy.ops.export_scene.gltf(
        filepath=export_path,
        export_format='GLB',
        use_selection=False,
    )


def optimize_mesh(obj, decimate_ratio=0.1):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    modifier = obj.modifiers.new(name="Decimate", type='DECIMATE')
    modifier.ratio = decimate_ratio
    modifier.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier="Decimate")
    

def load_mesh_with_material(ply_path, optimize=True):
    bpy.ops.wm.ply_import(filepath=ply_path)
    
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


def convert_ply_to_glb(ply_path):
    clear_objects()
    load_mesh_with_material(ply_path)
    export_glb(ply_path.replace(".ply", ".glb"))


if __name__ == "__main__":
    data_dir = '/mnt/fillipo/yuliu/wallbreaker/Projects/VideoArtGS/outputs/demo/urdf'
    scenes = sorted(os.listdir(data_dir))

    for s in scenes:
        mesh_dir = os.path.join(data_dir, s, 'meshes')
        ply_paths = sorted(glob(os.path.join(mesh_dir, 'part_*.ply')))
        print(f"Processing {s}, {len(ply_paths)} meshes")
        for ply_path in ply_paths:
            convert_ply_to_glb(ply_path)