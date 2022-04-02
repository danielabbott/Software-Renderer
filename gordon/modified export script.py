# modified version of blender export script from game engine project
# outputs array of f32 positions and array of f32 normals & UVs

import bpy
import io
import os
import subprocess
import json
import struct
import mathutils
from mathutils import Vector,Matrix
from math import floor,sqrt
import bmesh
import sys

def clamp(i, small, big):
    return max(min(i, big), small)

class ByteArrayWriter():
    def __init__(self):
        self.data = io.BytesIO()

    def writeByte(self, i, signed=False):
        self.data.write(i.to_bytes(1, byteorder='little', signed=signed))
    
    def writeWord(self, i, signed=False):
        self.data.write(i.to_bytes(2, byteorder='little', signed=signed))
    
    def writeDWord(self, i, signed=False):
        self.data.write(i.to_bytes(4, byteorder='little', signed=signed))
        
    def writeFloat(self, f):
        self.data.write(bytearray(struct.pack("f", f)))

    def pad(self, padding):
        sz = self.size()
        if (sz % padding) != 0:
            bytes_needed = padding - (sz % padding)
            zero = 0
            self.data.write(zero.to_bytes(bytes_needed, byteorder='little', signed=False))

    def write(self, s):
        self.file.write(s)

    def size(self):
        return self.data.tell()

    def get(self):
        self.data.seek(0)
        return self.data.read()

# Collection of attributes
class Vertex():
    def __init__(self, position):
        self.position = position

    def str(self):
        s = 'position: ' + str(self.position)

        if hasattr(self, 'normal'):
            s += '\nnormal: ' + str(self.normal)

        if hasattr(self, 'uv'):
            s += '\nuv: ' + str(self.uv)

        if hasattr(self, 'tangent'):
            s += '\ntangent: ' + str(self.tangent)

        if hasattr(self, 'bitangent_sign'):
            s += '\nbitangent sign: ' + str(self.bitangent_sign)

        if hasattr(self, 'bone_indices'):
            s += '\nbone indices: ' + str(self.bone_indices)

        if hasattr(self, 'bone_weight'):
            s += '\nbone weight: ' + str(self.bone_weight)
        return s
      
# class VertexData:
#     def __init__(self, index):
#         self.index = index
#         self.occurrences = 1 # number of times this vertex appears


class Material:
    def __init__(self, name, colour, flat_colour):
        self.indices = []
        self.name = name
        self.colour = colour
        self.flat_colour = flat_colour
        self.texture = ''
        self.normal_texture = ''
        self.specular = 0

    def is_equiv(self, mat2):
        return self.colour[0] == mat2.colour[0] and self.colour[1] == mat2.colour[1]\
            and self.colour[2] == mat2.colour[2] and self.texture == mat2.texture\
            and self.flat_colour[0] == mat2.flat_colour[0] and self.flat_colour[1] == mat2.flat_colour[1]\
            and self.flat_colour[2] == mat2.flat_colour[2]\
            and self.normal_texture == mat2.normal_texture\
            and self.specular == mat2.specular

def write_files(asset_text, data, resource_names, filepath, use_zstd, zstd_path_override):
    dir = os.path.dirname(filepath)
   # if not os.path.exists(dir):
    #    os.makedirs(dir)

    data_file_path = filepath.replace('.asset', '.data')
    print(data_file_path)
    data_file = open(data_file_path, 'wb')


    asset_text += 'resource_file ' + os.path.basename(data_file_path) + '\n'

    total_uncompressed_size = 0
    total_compressed_size = 0

    for data_to_write, resource_name in zip(data, resource_names):
        using_zstd = False
        uncompressed_length = len(data_to_write)

        if uncompressed_length == 0:
            continue

        total_uncompressed_size += uncompressed_length

        if use_zstd and uncompressed_length > 512:
            zstd_cmd = 'zstd' if zstd_path_override=='' else zstd_path_override

            try:
                result = subprocess.run([zstd_cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, input=data_to_write)
                if result.returncode != 0:
                    raise RuntimeError('zstd error ' + result.stderr.decode())
                compressed = result.stdout

                compression_ratio = len(compressed) / uncompressed_length

                if compression_ratio <= 0.8:
                    asset_text += 'resource ' + resource_name + ' zstd:' + str(uncompressed_length) + '>' + str(len(compressed)) + '\n'
                    data_to_write = compressed
                    using_zstd = True
                    total_compressed_size += len(compressed)
            except FileNotFoundError:
                print('ZSTD NOT FOUND')


        if not using_zstd:
            asset_text += 'resource ' + resource_name + ' raw:' + str(uncompressed_length) + '\n'
            total_compressed_size += uncompressed_length
        
        data_file.write(data_to_write)
    
    asset_text += "# ratio {:.1f}%".format((float(total_compressed_size) / float(total_uncompressed_size)) * 100.0)

    print('1')
    with open(filepath, 'w') as f:
        f.write(asset_text)

    print('2')
    data_file.close()
    print('3')

# Blender <--> Game Engine
def switch_coord_system(coords):
    # Objects face +Z in game engine's coordinate system
    return [coords[0], coords[2], -coords[1]]

class VertexPrepassData:
    pass

class Bone:
    pass

def get_bones():
    bones = []
    bone_name_to_index = {}
    blender_bone_to_index = {}

    for obj in bpy.data.objects:
        if not hasattr(obj.data, 'bones'):
            continue
        for blender_bone in obj.data.bones:
            bone = Bone()
            bone.name = blender_bone.name
            bone.head = switch_coord_system(obj.matrix_world @ blender_bone.head_local)
            bone.tail = switch_coord_system(obj.matrix_world @ blender_bone.tail_local)
            bone.blender_bone = blender_bone
            bone.blender_object = obj

            bone_name_to_index[bone.name] = len(bones)
            blender_bone_to_index[bone.blender_bone] = len(bones)
            bones.append(bone)

    for b in bones:
        b.m = None
        b.parent_index = None
        if b.blender_bone.parent != None and b.blender_bone != b.blender_bone.parent:
            b.parent_index = blender_bone_to_index[b.blender_bone.parent]
        if b.parent_index == None:
            b.parent_index = -1

    return bones, bone_name_to_index

def get_materials():
    materials = []
    blender_material_ref_to_materials_index = {}
    
    for mat in bpy.data.materials:
        if mat.name == 'Dots Stroke':
            continue

        m = Material(mat.name, [0.8, 0.8, 0.8], [0.8, 0.8, 0.8])
        m.name = mat.name

        nodes = mat.node_tree.nodes
        bsdfnodes = [n for n in nodes if isinstance(n, bpy.types.ShaderNodeBsdfPrincipled)]
        if len(bsdfnodes) > 0:
            bsdf = bsdfnodes[0]
            colour_input = bsdf.inputs['Base Color']
            m.flat_colour[0] = colour_input.default_value[0]
            m.flat_colour[1] = colour_input.default_value[1]
            m.flat_colour[2] = colour_input.default_value[2]
            m.colour = m.flat_colour

            m.specular = bsdf.inputs['Specular'].default_value

            if len(colour_input.links) > 0:
                texture_node = colour_input.links[0].from_node
                if isinstance(texture_node, bpy.types.ShaderNodeMixRGB):
                    mix_node = texture_node.inputs['Color1']
                    if len(mix_node.links) > 0:
                        texture_node = mix_node.links[0].from_node
                if isinstance(texture_node, bpy.types.ShaderNodeTexImage):
                    m.texture = texture_node.image.filepath.split('/')[-1]
                    m.colour = [0.8, 0.8, 0.8]

            normal_map_input = bsdf.inputs['Normal']
            if len(normal_map_input.links) > 0:
                normal_map_node = normal_map_input.links[0].from_node
                if normal_map_node.space == 'TANGENT':
                    normal_texture_input = normal_map_node.inputs['Color']
                    if len(normal_texture_input.links) > 0:
                        normal_texture_node = normal_texture_input.links[0].from_node
                        if isinstance(normal_texture_node, bpy.types.ShaderNodeTexImage):
                            m.normal_texture = normal_texture_node.image.filepath.split('/')[-1]
                else:
                    print('Normals must be in tangent space')



        materials.append(m)
        blender_material_ref_to_materials_index[mat] = materials[-1]

    
    default_mat = Material('Default Material', [0.8, 0.8, 0.8], [0.8, 0.8, 0.8])
    materials.append(default_mat)

    return materials, blender_material_ref_to_materials_index


def vertices_prepass(flat_shading, export_tangents, bones, bone_name_to_index):
    vertices = []
    new_blender_objects = []
    max_position = Vector ([-9999.0,-9999.0,-9999.0])
    min_position = Vector ([9999.0,9999.0,9999.0])

    objects_with_polygons = []
    for o in bpy.data.objects:
        if not hasattr(o.data, 'polygons') or \
            len(o.data.vertices) < 3 or \
            len(o.data.polygons) < 1:
            continue
        objects_with_polygons.append(o)


    for original_blender_object in objects_with_polygons:
        blender_object = original_blender_object.copy()
        blender_object.name = 'copy'
        new_blender_objects.append(blender_object)

        # Triangulate

        m = bmesh.new()
        m.from_mesh(original_blender_object.data)
        bmesh.ops.triangulate(m, faces=m.faces[:])
        m.to_mesh(blender_object.data)
        m.free()

        # Calculate data

        if flat_shading:
            blender_object.data.calc_normals()

        if export_tangents:
            try:
                blender_object.data.calc_tangents()
            except:
                raise RuntimeError('calc_tangents error. Problematic object: ' + blender_object.name+
                    '. Is the object UV mapped?')
            
        normal_matrix = blender_object.matrix_world.to_3x3()
        normal_matrix.invert()
        normal_matrix = normal_matrix.transposed().to_4x4()


        for blender_vertex in blender_object.data.vertices:
            vertex = VertexPrepassData()

            # Position
            vertex.position = switch_coord_system(blender_object.matrix_world @ blender_vertex.co)

            # Normal
            vertex.normal = normal_matrix @ blender_vertex.normal
            vertex.normal.normalize()
            vertex.normal = switch_coord_system(vertex.normal)

            if len(bones) > 0:
                # bone weights
                bone_weights = [0.0,0.0]
                bone_indices = [-1,-1]
                for vgroup in blender_vertex.groups:
                    w = vgroup.weight
                    bone_name = blender_object.vertex_groups[vgroup.group].name
                    if bone_name in bone_name_to_index:
                        bone_index = bone_name_to_index[bone_name]

                        if w >= bone_weights[0]:
                            bone_weights[1] = bone_weights[0]
                            bone_indices[1] = bone_indices[0]
                            bone_weights[0] = w
                            bone_indices[0] = bone_index

                        elif w > bone_weights[1]:
                            bone_weights[1] = w
                            bone_indices[1] = bone_index


                weight_sum = bone_weights[0] + bone_weights[1]
                if weight_sum <= 0.0:
                    weight_sum = 1.0
                weight_mul = 1.0 / weight_sum
                vertex.bone_weight = bone_weights[0]*weight_mul
                vertex.bone_indices = bone_indices

            vertices.append(vertex)

            # Update min/max position
            co = vertex.position
            if co[0] > max_position[0]:
                max_position[0] = co[0]
            if co[1] > max_position[1]:
                max_position[1] = co[1]
            if co[2] > max_position[2]:
                max_position[2] = co[2]
            if co[0] < min_position[0]:
                min_position[0] = co[0]
            if co[1] < min_position[1]:
                min_position[1] = co[1]
            if co[2] < min_position[2]:
                min_position[2] = co[2]



    return vertices,new_blender_objects,max_position,min_position
        


def get_mesh_data(flat_shading, export_tangents):
    bones, bone_name_to_index = get_bones()

    vertices_constant_data,blender_objects,max_position,min_position = \
    vertices_prepass(flat_shading, export_tangents, bones, bone_name_to_index)
    
    materials, blender_material_ref_to_materials_index = get_materials()
    

    # Vertex.str() -> index
    vertices_dict = {}
    vertices_list = []

    object_vertex_index = 0
    for blender_object in blender_objects:
        normal_matrix = blender_object.matrix_world.to_3x3()
        normal_matrix.invert()
        normal_matrix = normal_matrix.transposed().to_4x4()

        for polygon in blender_object.data.polygons:
            face_indices = polygon.vertices
            loop_indices = polygon.loop_indices

            if len(face_indices) != len(loop_indices):
                raise RuntimeError('Invalid face data')

            if len(face_indices) > 3:
                raise RuntimeError('Triangulation error')
            elif len(polygon.vertices) < 3:
                continue
             
            

            # Get material of triangle
            if polygon.material_index >= 0 and len(blender_object.data.materials) > 0 \
                    and len(bpy.data.materials) > 0:
                mat = blender_material_ref_to_materials_index[blender_object.data.materials[polygon.material_index]]
            else:
                mat = materials[-1]


            # Create vertices & indices
            for j in range(len(face_indices)):
                face_index = face_indices[j]
                loop_index = loop_indices[j]
                const = vertices_constant_data[object_vertex_index + face_index]

                vertex = Vertex(const.position)

                if len(bones) > 0:
                    vertex.bone_weight = const.bone_weight
                    vertex.bone_indices = const.bone_indices

                if flat_shading:
                    vertex.normal = normal_matrix @ polygon.normal
                else:
                    vertex.normal = const.normal


                vertex.uv = Vector((0.0,0.0))
                vertex.tangent = Vector((0.0,0.0,0.0))
                vertex.bitangent_sign = 0.0
                if blender_object.data.uv_layers.active is not None:
                    vertex.uv = blender_object.data.uv_layers.active.data[loop_index].uv
                
                    vertex.tangent = normal_matrix @ blender_object.data.loops[loop_index].tangent
                    vertex.tangent.normalize()
                    vertex.tangent = switch_coord_system(vertex.tangent)
                    vertex.bitangent_sign = blender_object.data.loops[loop_index].bitangent_sign

                
                if vertex.str() in vertices_dict:
                    mat.indices.append(vertices_dict[vertex.str()])
                else:
                    idx = len(vertices_list)
                    vertices_dict[vertex.str()] = idx
                    vertices_list.append(vertex)
                    mat.indices.append(idx)


        object_vertex_index += len(blender_object.data.vertices)
    

    optimised_materials = []

    i = 0
    while i < len(materials):
        mat = materials[i]

        j = i+1
        while j < len(materials):
            mat2 = materials[j]
            if mat.is_equiv(mat2):
                mat.indices += mat2.indices
                materials.remove(mat2)
            else:
                j += 1

        if len(mat.indices) >= 3:
            optimised_materials.append(mat)
        
        i += 1

    materials = optimised_materials


    return vertices_list,min_position,max_position,materials,bones

class Animation:
    pass

class BoneData:
    pass

def get_animation_data(loop_animations, bones):
    animations = []

    # Matrices (bone matrices assume blender coord system but vertices will be in vulkan coord system)
    to_vulkan_coords = Matrix()
    to_vulkan_coords[1][1] = 0.0
    to_vulkan_coords[1][2] = 1.0
    to_vulkan_coords[2][1] = -1.0
    to_vulkan_coords[2][2] = 0.0

    to_blender_coords = to_vulkan_coords.inverted()


    for action in bpy.data.actions:
        anim = Animation()
        anim.name = action.name.replace(' ', '_').lower()

        anim.loops = False
        if anim.name in loop_animations:
            anim.loops = True
            loop_animations.remove(anim.name)

        # Make action current on all armatures
        for obj in bpy.data.objects:
            if hasattr(obj.data, 'bones') and hasattr(obj.data, 'animation_data'):
                obj.animation_data.action = action

        first_frame = int(action.frame_range[0])
        total_frames = int(action.frame_range[1]) - first_frame
        if anim.loops or total_frames == 2:
            # If looped, last frame should be same as first (so interpolation is correct)
            # If there are only 2 frames then this is a single pose (not animated)
            # ^ first_frames is (1,2) for both 1 and 2 frames of animations for some reason
            total_frames -= 1

        anim.frames = []

        for frame in range(total_frames):
            frame_data = []
            bpy.context.scene.frame_set(first_frame + frame)

            for bone_index,b in enumerate(bones):
                                
                obj = b.blender_object
                pose_bone = obj.pose.bones[b.name]
                
                edit_mode_transform = b.blender_bone.matrix_local
                pose_mode_transform = pose_bone.matrix
                
                m = to_vulkan_coords @ b.blender_object.matrix_world @\
                 pose_mode_transform @ edit_mode_transform.inverted() @\
                  b.blender_object.matrix_world.inverted() @ to_blender_coords

                
                bone_data = BoneData()
                bone_data.translation, bone_data.rotation, scale = m.decompose()
                bone_data.scale = (scale.x + scale.y + scale.z) / 3.0
               

                frame_data.append(bone_data)
                

            anim.frames.append(frame_data)
        animations.append(anim)

    if len(loop_animations) > 0:
        print('The following animations were referenced in the import file but were not found:', loop_animations)

    
    return bpy.context.scene.render.fps, animations

def create_model_file(context, asset_name, filepath, use_zstd, zstd_path_override, flat_shading, \
export_uv, export_tangents, loop_animations):
    if filepath[-6:] != '.asset':
        raise ValueError('Output file should be a .asset file')


    vertices,min_position,max_position,materials,bones = \
    get_mesh_data(flat_shading, export_tangents)
    
    position_value_range = max_position - min_position

    if position_value_range[0] == 0.0:
        position_value_range[0] = 1.0
    if position_value_range[1] == 0.0:
        position_value_range[1] = 1.0
    if position_value_range[2] == 0.0:
        position_value_range[2] = 1.0


    asset_text_file = 'asset ' + asset_name + '\n'
    asset_text_file += 'asset_type model\n'
    asset_text_file += 'vertex_count ' + str(len(vertices)) + '\n'
    
    indices_count = 0

    if len(vertices) > 0:
        asset_text_file += 'bounds_minimum ' + str(min_position[0]) + ' ' + str(min_position[1]) + \
            ' ' + str(min_position[2]) + '\n'
        asset_text_file += 'bounds_range ' + str(position_value_range[0]) + ' ' + str(position_value_range[1]) + \
            ' ' + str(position_value_range[2]) + '\n'


        for m in materials:
            indices_count += len(m.indices)

        asset_text_file += 'indices_count ' + str(indices_count) + '\n'


        indices_offset = 0
        for m in materials:
            asset_text_file += 'material ' + str(m.name).replace(' ', '_').lower() + '\n'
            asset_text_file += "colour {:.3f} {:.3f} {:.3f}\n".format(m.colour[0], m.colour[1], m.colour[2])
            asset_text_file += "flat_colour {:.3f} {:.3f} {:.3f}\n".format(m.flat_colour[0], m.flat_colour[1], m.flat_colour[2])
            asset_text_file += 'first ' + str(indices_offset) + '\n'
            material_index_count = len(m.indices)
            indices_offset += material_index_count
            asset_text_file += 'count ' + str(material_index_count) + '\n'
            if m.texture != '':
                asset_text_file += 'texture ' + m.texture + '\n'
            if m.normal_texture != '':
                asset_text_file += 'normal_map ' + m.normal_texture + '\n'
            asset_text_file += 'specular ' + str(m.specular) + '\n'
            

    if len(bones) > 0:
        for bone in bones:
            asset_text_file += 'bone ' + bone.name.replace(' ', '_').lower() + '\n'
            asset_text_file += 'head ' + str(bone.head[0]) + ' ' + str(bone.head[1]) + ' ' + str(bone.head[2]) + '\n'
            # asset_text_file += 'tail ' + str(bone.tail[0]) + ' ' + str(bone.tail[1]) + ' ' + str(bone.tail[2]) + '\n'
            if bone.parent_index >= 0:
                asset_text_file += 'parent ' + str(bone.parent_index) + '\n'

        fps,animations = get_animation_data(loop_animations, bones)
        asset_text_file += 'frame_rate ' + str(fps) + '\n'

        for anim in animations:
            asset_text_file += 'animation ' + anim.name + '\n'
            asset_text_file += 'loops ' + str(anim.loops).lower() + '\n'
            asset_text_file += 'frames ' + str(len(anim.frames)) + '\n'


    resource_names = ['vattr-position-f32', 'vattr-position-normal-uv-f32']
    writers = [ByteArrayWriter(),ByteArrayWriter()]

    w = writers[0]
    for v in vertices:
        w.writeFloat(v.position[0])
        w.writeFloat(v.position[1])
        w.writeFloat(v.position[2])

        
    w = writers[1]
    for v in vertices:
        w.writeFloat(v.normal[0])
        w.writeFloat(v.normal[1])
        w.writeFloat(v.normal[2])

        if v.uv is None:
            w.writeFloat(0)
            w.writeFloat(0)
        else:
            w.writeFloat(v.uv[0])
            w.writeFloat(1-v.uv[1])


    

    if indices_count > 0:
        w = ByteArrayWriter()
        writers.append(w)
        resource_names.append('indices_u16')
        for m in materials:
            for i in m.indices:
                w.writeWord(i)


    data = [x.get() for x in writers]

    write_files(asset_text_file, data, resource_names, filepath, use_zstd, zstd_path_override)


    return {'FINISHED'}


if __name__ == "__main__":
    # https://blender.stackexchange.com/a/8405/74215
    #output_file = sys.argv[sys.argv.index("--") + 1]  # get all args after "--"
    #asset_name = sys.argv[sys.argv.index("--") + 2]  # get args after "--"
    
    output_file = 'gordon.asset'
    asset_name = 'gordon'

    use_flat_shading = False

    loop_animations = []

    try:
        with open(sys.argv[1] + '.import', 'r') as f:
            lines = f.readlines()
            for l in lines:
                x = l.split()
                if len(x) >= 1: 
                    if x[0].lower() == 'flat_shading':
                        use_flat_shading = True
                    elif x[0].lower() == 'loop':
                        loop_animations.append(x[1])
                    else:
                        print('Unrecognised import option: ' + x)
    except:
        pass

    print ('Outputting to ' + output_file)
    create_model_file(None, asset_name, output_file, use_zstd=False, zstd_path_override='', \
    flat_shading=False, export_uv=True, export_tangents=False, loop_animations=loop_animations)
