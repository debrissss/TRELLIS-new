import argparse, sys, os, math, re, glob, time
from typing import *
import bpy
from mathutils import Vector, Matrix
import numpy as np
import json
import glob


"""=============== BLENDER ==============="""

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.import_scene.obj,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.import_mesh.ply,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
}

EXT = {
    'PNG': 'png',
    'JPEG': 'jpg',
    'OPEN_EXR': 'exr',
    'TIFF': 'tiff',
    'BMP': 'bmp',
    'HDR': 'hdr',
    'TARGA': 'tga'
}

def init_render(engine='CYCLES', resolution=512, geo_mode=False, disable_denoise=False):
    bpy.context.scene.render.engine = engine
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.film_transparent = True
    
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.samples = 128 if not geo_mode else 1
    bpy.context.scene.cycles.filter_type = 'BOX'
    bpy.context.scene.cycles.filter_width = 1
    bpy.context.scene.cycles.diffuse_bounces = 1
    bpy.context.scene.cycles.glossy_bounces = 1
    bpy.context.scene.cycles.transparent_max_bounces = 3 if not geo_mode else 0
    bpy.context.scene.cycles.transmission_bounces = 3 if not geo_mode else 1
    bpy.context.scene.cycles.use_denoising = not disable_denoise
    bpy.context.scene.render.use_persistent_data = True

    cycles_preferences = bpy.context.preferences.addons['cycles'].preferences
    for device_type in ('OPTIX', 'CUDA'):
        try:
            cycles_preferences.compute_device_type = device_type
            cycles_preferences.get_devices()
            enabled_devices = []
            for device in cycles_preferences.devices:
                device.use = device.type == device_type
                if device.use:
                    enabled_devices.append(f'{device.name} ({device.type})')
            if enabled_devices:
                bpy.context.scene.cycles.device = 'GPU'
                print(f'[INFO] Cycles GPU device type: {device_type}')
                print(f'[INFO] Enabled Cycles devices: {", ".join(enabled_devices)}')
                break
        except TypeError:
            continue
        except Exception as e:
            print(f'[WARN] Failed to enable Cycles {device_type}: {e}')
    else:
        bpy.context.scene.cycles.device = 'CPU'
        print('[WARN] No OPTIX/CUDA Cycles GPU device enabled; falling back to CPU.')
    
def init_nodes(save_depth=False, save_normal=False, save_albedo=False, save_mist=False):
    if not any([save_depth, save_normal, save_albedo, save_mist]):
        return {}, {}
    outputs = {}
    spec_nodes = {}
    
    bpy.context.scene.use_nodes = True
    bpy.context.scene.view_layers['View Layer'].use_pass_z = save_depth
    bpy.context.scene.view_layers['View Layer'].use_pass_normal = save_normal
    bpy.context.scene.view_layers['View Layer'].use_pass_diffuse_color = save_albedo
    bpy.context.scene.view_layers['View Layer'].use_pass_mist = save_mist
    
    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links
    for n in nodes:
        nodes.remove(n)
    
    render_layers = nodes.new('CompositorNodeRLayers')
    
    if save_depth:
        depth_file_output = nodes.new('CompositorNodeOutputFile')
        depth_file_output.base_path = ''
        depth_file_output.file_slots[0].use_node_format = True
        depth_file_output.format.file_format = 'PNG'
        depth_file_output.format.color_depth = '16'
        depth_file_output.format.color_mode = 'BW'
        # Remap to 0-1
        map = nodes.new(type="CompositorNodeMapRange")
        map.inputs[1].default_value = 0  # (min value you will be getting)
        map.inputs[2].default_value = 10 # (max value you will be getting)
        map.inputs[3].default_value = 0  # (min value you will map to)
        map.inputs[4].default_value = 1  # (max value you will map to)
        
        links.new(render_layers.outputs['Depth'], map.inputs[0])
        links.new(map.outputs[0], depth_file_output.inputs[0])
        
        outputs['depth'] = depth_file_output
        spec_nodes['depth_map'] = map
    
    if save_normal:
        normal_file_output = nodes.new('CompositorNodeOutputFile')
        normal_file_output.base_path = ''
        normal_file_output.file_slots[0].use_node_format = True
        normal_file_output.format.file_format = 'OPEN_EXR'
        normal_file_output.format.color_mode = 'RGB'
        normal_file_output.format.color_depth = '16'
        
        links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])
        
        outputs['normal'] = normal_file_output
    
    if save_albedo:
        albedo_file_output = nodes.new('CompositorNodeOutputFile')
        albedo_file_output.base_path = ''
        albedo_file_output.file_slots[0].use_node_format = True
        albedo_file_output.format.file_format = 'PNG'
        albedo_file_output.format.color_mode = 'RGBA'
        albedo_file_output.format.color_depth = '8'
        
        alpha_albedo = nodes.new('CompositorNodeSetAlpha')
        
        links.new(render_layers.outputs['DiffCol'], alpha_albedo.inputs['Image'])
        links.new(render_layers.outputs['Alpha'], alpha_albedo.inputs['Alpha'])
        links.new(alpha_albedo.outputs['Image'], albedo_file_output.inputs[0])
        
        outputs['albedo'] = albedo_file_output
        
    if save_mist:
        bpy.data.worlds['World'].mist_settings.start = 0
        bpy.data.worlds['World'].mist_settings.depth = 10
        
        mist_file_output = nodes.new('CompositorNodeOutputFile')
        mist_file_output.base_path = ''
        mist_file_output.file_slots[0].use_node_format = True
        mist_file_output.format.file_format = 'PNG'
        mist_file_output.format.color_mode = 'BW'
        mist_file_output.format.color_depth = '16'
        
        links.new(render_layers.outputs['Mist'], mist_file_output.inputs[0])
        
        outputs['mist'] = mist_file_output
        
    return outputs, spec_nodes

def init_scene() -> None:
    """Resets the scene to a clean state.

    Returns:
        None
    """
    # delete everything
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

def init_camera():
    cam = bpy.data.objects.new('Camera', bpy.data.cameras.new('Camera'))
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.data.sensor_height = cam.data.sensor_width = 32
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    cam_empty = bpy.data.objects.new("Empty", None)
    cam_empty.location = (0, 0, 0)
    bpy.context.scene.collection.objects.link(cam_empty)
    cam_constraint.target = cam_empty
    return cam

def init_lighting():
    # Clear existing lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()
    
    # Create key light
    default_light = bpy.data.objects.new("Default_Light", bpy.data.lights.new("Default_Light", type="POINT"))
    bpy.context.collection.objects.link(default_light)
    default_light.data.energy = 1000
    default_light.location = (4, 1, 6)
    default_light.rotation_euler = (0, 0, 0)
    
    # create top light
    top_light = bpy.data.objects.new("Top_Light", bpy.data.lights.new("Top_Light", type="AREA"))
    bpy.context.collection.objects.link(top_light)
    top_light.data.energy = 10000
    top_light.location = (0, 0, 10)
    top_light.scale = (100, 100, 100)
    
    # create bottom light
    bottom_light = bpy.data.objects.new("Bottom_Light", bpy.data.lights.new("Bottom_Light", type="AREA"))
    bpy.context.collection.objects.link(bottom_light)
    bottom_light.data.energy = 1000
    bottom_light.location = (0, 0, -10)
    bottom_light.rotation_euler = (0, 0, 0)
    
    return {
        "default_light": default_light,
        "top_light": top_light,
        "bottom_light": bottom_light
    }


def profile_log(enabled: bool, event: str, **kwargs) -> None:
    if not enabled:
        return
    payload = " ".join(f"{key}={value}" for key, value in kwargs.items())
    print(f"[PROFILE] {event}" + (f" {payload}" if payload else ""), flush=True)


def load_object(object_path: str) -> None:
    """Loads a model with a supported file extension into the scene.

    Args:
        object_path (str): Path to the model file.

    Raises:
        ValueError: If the file extension is not supported.

    Returns:
        None
    """
    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")

    if file_extension == "ply" and load_binary_ply_fast(object_path):
        return None

    if file_extension == "usdz":
        # install usdz io package
        dirname = os.path.dirname(os.path.realpath(__file__))
        usdz_package = os.path.join(dirname, "io_scene_usdz.zip")
        bpy.ops.preferences.addon_install(filepath=usdz_package)
        # enable it
        addon_name = "io_scene_usdz"
        bpy.ops.preferences.addon_enable(module=addon_name)
        # import the usdz
        from io_scene_usdz.import_usdz import import_usdz

        import_usdz(context, filepath=object_path, materials=True, animations=True)
        return None

    # load from existing import functions
    import_function = IMPORT_FUNCTIONS[file_extension]

    print(f"Loading object from {object_path}")
    if file_extension == "blend":
        import_function(directory=object_path, link=False)
    elif file_extension in {"glb", "gltf"}:
        import_function(filepath=object_path, merge_vertices=True, import_shading='NORMALS')
    else:
        import_function(filepath=object_path)


PLY_NUMPY_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def _parse_ply_header(file) -> Tuple[Dict[str, Any], int]:
    header_lines = []
    header_size = 0
    while True:
        line = file.readline()
        if not line:
            raise ValueError("PLY header is incomplete")
        header_size += len(line)
        decoded = line.decode("ascii").strip()
        header_lines.append(decoded)
        if decoded == "end_header":
            break

    elements = []
    current = None
    ply_format = None
    for line in header_lines:
        if line.startswith("format "):
            ply_format = line.split()[1]
        elif line.startswith("element "):
            _, name, count = line.split()[:3]
            current = {"name": name, "count": int(count), "properties": []}
            elements.append(current)
        elif line.startswith("property ") and current is not None:
            parts = line.split()
            if parts[1] == "list":
                current["properties"].append({
                    "kind": "list",
                    "count_type": parts[2],
                    "item_type": parts[3],
                    "name": parts[4],
                })
            else:
                current["properties"].append({
                    "kind": "scalar",
                    "type": parts[1],
                    "name": parts[2],
                })

    return {"format": ply_format, "elements": elements}, header_size


def load_binary_ply_fast(object_path: str) -> bool:
    """Fast path for standard binary little-endian triangular PLY meshes."""
    try:
        with open(object_path, "rb") as f:
            header, header_size = _parse_ply_header(f)
            if header["format"] != "binary_little_endian":
                return False

            elements = {element["name"]: element for element in header["elements"]}
            vertex_element = elements.get("vertex")
            face_element = elements.get("face")
            if vertex_element is None or face_element is None:
                return False

            vertex_dtype_fields = []
            for prop in vertex_element["properties"]:
                if prop["kind"] != "scalar" or prop["type"] not in PLY_NUMPY_DTYPES:
                    return False
                vertex_dtype_fields.append((prop["name"], PLY_NUMPY_DTYPES[prop["type"]]))

            vertex_dtype = np.dtype(vertex_dtype_fields)
            vertex_count = vertex_element["count"]
            vertices_raw = np.frombuffer(f.read(vertex_count * vertex_dtype.itemsize), dtype=vertex_dtype, count=vertex_count)
            if not {"x", "y", "z"}.issubset(vertices_raw.dtype.names):
                return False

            vertices = np.column_stack((
                vertices_raw["x"],
                vertices_raw["y"],
                vertices_raw["z"],
            )).astype(np.float32, copy=False)

            face_props = face_element["properties"]
            if len(face_props) != 1 or face_props[0]["kind"] != "list":
                return False
            face_prop = face_props[0]
            if face_prop["count_type"] not in {"uchar", "uint8"} or face_prop["item_type"] not in {"int", "int32"}:
                return False

            face_count = face_element["count"]
            face_dtype = np.dtype([
                ("count", "u1"),
                ("v0", "<i4"),
                ("v1", "<i4"),
                ("v2", "<i4"),
            ])
            faces_raw = np.frombuffer(f.read(face_count * face_dtype.itemsize), dtype=face_dtype, count=face_count)
            if len(faces_raw) != face_count or not np.all(faces_raw["count"] == 3):
                return False

            faces = np.column_stack((
                faces_raw["v0"],
                faces_raw["v1"],
                faces_raw["v2"],
            )).astype(np.int32, copy=False)

        print(f"Loading binary PLY object from {object_path}")
        mesh = bpy.data.meshes.new("mesh")
        mesh.vertices.add(len(vertices))
        mesh.loops.add(len(faces) * 3)
        mesh.polygons.add(len(faces))

        mesh.vertices.foreach_set("co", vertices.reshape(-1))
        mesh.loops.foreach_set("vertex_index", faces.reshape(-1))
        mesh.polygons.foreach_set("loop_start", np.arange(0, len(faces) * 3, 3, dtype=np.int32))
        mesh.polygons.foreach_set("loop_total", np.full(len(faces), 3, dtype=np.int32))
        mesh.update()

        obj = bpy.data.objects.new("mesh", mesh)
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        print(f"[INFO] Fast binary PLY import succeeded: {len(vertices)} vertices, {len(faces)} faces")
        return True
    except Exception as e:
        print(f"[WARN] Fast binary PLY import failed, falling back to Blender importer: {e}")
        return False
        
def delete_invisible_objects() -> None:
    """Deletes all invisible objects in the scene.

    Returns:
        None
    """
    # bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.hide_viewport or obj.hide_render:
            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_select = False
            obj.select_set(True)
    bpy.ops.object.delete()

    # Delete invisible collections
    invisible_collections = [col for col in bpy.data.collections if col.hide_viewport]
    for col in invisible_collections:
        bpy.data.collections.remove(col)
        
def split_mesh_normal():
    bpy.ops.object.select_all(action="DESELECT")
    objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    bpy.context.view_layer.objects.active = objs[0]
    for obj in objs:
        obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.split_normals()
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action="DESELECT")
            
def delete_custom_normals():
     for this_obj in bpy.data.objects:
        if this_obj.type == "MESH":
            bpy.context.view_layer.objects.active = this_obj
            bpy.ops.mesh.customdata_custom_splitnormals_clear()

def override_material():
    new_mat = bpy.data.materials.new(name="Override0123456789")
    new_mat.use_nodes = True
    new_mat.node_tree.nodes.clear()
    bsdf = new_mat.node_tree.nodes.new('ShaderNodeBsdfDiffuse')
    bsdf.inputs[0].default_value = (0.5, 0.5, 0.5, 1)
    bsdf.inputs[1].default_value = 1
    output = new_mat.node_tree.nodes.new('ShaderNodeOutputMaterial')
    new_mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    bpy.context.scene.view_layers['View Layer'].material_override = new_mat

def unhide_all_objects() -> None:
    """Unhides all objects in the scene.

    Returns:
        None
    """
    for obj in bpy.context.scene.objects:
        obj.hide_set(False)
        
def convert_to_meshes() -> None:
    """Converts all objects in the scene to meshes.

    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"][0]
    for obj in bpy.context.scene.objects:
        obj.select_set(True)
    bpy.ops.object.convert(target="MESH")
        
def triangulate_meshes() -> None:
    """Triangulates all meshes in the scene.

    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")
    objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    bpy.context.view_layer.objects.active = objs[0]
    for obj in objs:
        obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.reveal()
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")

def scene_bbox() -> Tuple[Vector, Vector]:
    """Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    scene_meshes = [obj for obj in bpy.context.scene.objects.values() if isinstance(obj.data, bpy.types.Mesh)]
    for obj in scene_meshes:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

def normalize_scene() -> Tuple[float, Vector]:
    """Normalizes the scene by scaling and translating it to fit in a unit cube centered
    at the origin.

    Mostly taken from the Point-E / Shap-E rendering script
    (https://github.com/openai/point-e/blob/main/point_e/evals/scripts/blender_script.py#L97-L112),
    but fix for multiple root objects: (see bug report here:
    https://github.com/openai/shap-e/pull/60).

    Returns:
        Tuple[float, Vector]: The scale factor and the offset applied to the scene.
    """
    scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
    if len(scene_root_objects) > 1:
        # create an empty object to be used as a parent for all root objects
        scene = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(scene)

        # parent all root objects to the empty object
        for obj in scene_root_objects:
            obj.parent = scene
    else:
        scene = scene_root_objects[0]

    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    scene.scale = scene.scale * scale

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    scene.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")
    
    return scale, offset

def get_transform_matrix(obj: bpy.types.Object) -> list:
    pos, rt, _ = obj.matrix_world.decompose()
    rt = rt.to_matrix()
    matrix = []
    for ii in range(3):
        a = []
        for jj in range(3):
            a.append(rt[ii][jj])
        a.append(pos[ii])
        matrix.append(a)
    matrix.append([0, 0, 0, 1])
    return matrix

def profile_scene_bbox(enabled: bool, event: str) -> None:
    if not enabled:
        return
    bbox_min, bbox_max = scene_bbox()
    extent = bbox_max - bbox_min
    center = (bbox_min + bbox_max) / 2
    profile_log(
        True,
        event,
        min=f"[{bbox_min.x:.6f},{bbox_min.y:.6f},{bbox_min.z:.6f}]",
        max=f"[{bbox_max.x:.6f},{bbox_max.y:.6f},{bbox_max.z:.6f}]",
        extent=f"[{extent.x:.6f},{extent.y:.6f},{extent.z:.6f}]",
        max_extent=f"{max(extent):.6f}",
        center=f"[{center.x:.6f},{center.y:.6f},{center.z:.6f}]",
    )

def main(arg):
    total_start = time.perf_counter()
    os.makedirs(arg.output_folder, exist_ok=True)
    
    # Initialize context
    init_start = time.perf_counter()
    init_render(
        engine=arg.engine,
        resolution=arg.resolution,
        geo_mode=arg.geo_mode,
        disable_denoise=arg.profile_disable_denoise,
    )
    outputs, spec_nodes = init_nodes(
        save_depth=arg.save_depth,
        save_normal=arg.save_normal,
        save_albedo=arg.save_albedo,
        save_mist=arg.save_mist
    )
    profile_log(arg.profile, "init_render", seconds=f"{time.perf_counter() - init_start:.6f}")

    load_start = time.perf_counter()
    if arg.object.endswith(".blend"):
        delete_invisible_objects()
    else:
        init_scene()
        load_object(arg.object)
        if arg.split_normal:
            split_mesh_normal()
        # delete_custom_normals()
    print('[INFO] Scene initialized.')
    profile_log(arg.profile, "load_object", seconds=f"{time.perf_counter() - load_start:.6f}")
    profile_scene_bbox(arg.profile, "bbox_after_load")
    
    # normalize scene
    normalize_start = time.perf_counter()
    if arg.skip_normalize:
        scale, offset = 1.0, Vector((0.0, 0.0, 0.0))
        print('[INFO] Scene normalization skipped.')
    else:
        scale, offset = normalize_scene()
        print('[INFO] Scene normalized.')
    profile_log(arg.profile, "normalize_scene", seconds=f"{time.perf_counter() - normalize_start:.6f}")
    profile_scene_bbox(arg.profile, "bbox_after_normalize")
    
    # Initialize camera and lighting
    camera_start = time.perf_counter()
    cam = init_camera()
    init_lighting()
    print('[INFO] Camera and lighting initialized.')
    profile_log(arg.profile, "init_camera_lighting", seconds=f"{time.perf_counter() - camera_start:.6f}")

    # Override material
    if arg.geo_mode:
        override_material()
    
    # Create a list of views
    to_export = {
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "scale": scale,
        "offset": [offset.x, offset.y, offset.z],
        "frames": []
    }
    views = json.loads(arg.views)
    frame_times = []
    render_times = []
    output_times = []
    for i, view in enumerate(views):
        frame_start = time.perf_counter()
        cam.location = (
            view['radius'] * np.cos(view['yaw']) * np.cos(view['pitch']),
            view['radius'] * np.sin(view['yaw']) * np.cos(view['pitch']),
            view['radius'] * np.sin(view['pitch'])
        )
        cam.data.lens = 16 / np.tan(view['fov'] / 2)
        
        if arg.save_depth:
            spec_nodes['depth_map'].inputs[1].default_value = view['radius'] - 0.5 * np.sqrt(3)
            spec_nodes['depth_map'].inputs[2].default_value = view['radius'] + 0.5 * np.sqrt(3)
        
        bpy.context.scene.render.filepath = os.path.join(arg.output_folder, f'{i:03d}.png')
        for name, output in outputs.items():
            output.file_slots[0].path = os.path.join(arg.output_folder, f'{i:03d}_{name}')
            
        # Render the scene
        render_start = time.perf_counter()
        bpy.ops.render.render(write_still=not arg.profile_no_write)
        render_seconds = time.perf_counter() - render_start
        output_start = time.perf_counter()
        if arg.profile_no_write:
            bpy.data.images['Render Result'].save_render(filepath=os.path.join(arg.output_folder, f'{i:03d}.png'))
        bpy.context.view_layer.update()
        for name, output in outputs.items():
            ext = EXT[output.format.file_format]
            path = glob.glob(f'{output.file_slots[0].path}*.{ext}')[0]
            os.rename(path, f'{output.file_slots[0].path}.{ext}')
            
        # Save camera parameters
        metadata = {
            "file_path": f'{i:03d}.png',
            "camera_angle_x": view['fov'],
            "transform_matrix": get_transform_matrix(cam)
        }
        if arg.save_depth:
            metadata['depth'] = {
                'min': view['radius'] - 0.5 * np.sqrt(3),
                'max': view['radius'] + 0.5 * np.sqrt(3)
            }
        to_export["frames"].append(metadata)
        output_seconds = time.perf_counter() - output_start
        frame_seconds = time.perf_counter() - frame_start
        render_times.append(render_seconds)
        output_times.append(output_seconds)
        frame_times.append(frame_seconds)
        profile_log(
            arg.profile,
            "frame",
            index=i,
            total=f"{frame_seconds:.6f}",
            render=f"{render_seconds:.6f}",
            post=f"{output_seconds:.6f}",
        )
    
    # Save the camera parameters
    transforms_start = time.perf_counter()
    with open(os.path.join(arg.output_folder, 'transforms.json'), 'w') as f:
        json.dump(to_export, f, indent=4)
    profile_log(arg.profile, "write_transforms", seconds=f"{time.perf_counter() - transforms_start:.6f}")
    if arg.profile and frame_times:
        profile_log(
            True,
            "summary",
            frames=len(frame_times),
            total=f"{time.perf_counter() - total_start:.6f}",
            frame_avg=f"{np.mean(frame_times):.6f}",
            frame_min=f"{np.min(frame_times):.6f}",
            frame_max=f"{np.max(frame_times):.6f}",
            render_avg=f"{np.mean(render_times):.6f}",
            post_avg=f"{np.mean(output_times):.6f}",
        )
        
    if arg.save_mesh:
        # triangulate meshes
        unhide_all_objects()
        convert_to_meshes()
        triangulate_meshes()
        print('[INFO] Meshes triangulated.')
        
        # export ply mesh
        bpy.ops.export_mesh.ply(filepath=os.path.join(arg.output_folder, 'mesh.ply'))

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
    parser.add_argument('--views', type=str, help='JSON string of views. Contains a list of {yaw, pitch, radius, fov} object.')
    parser.add_argument('--object', type=str, help='Path to the 3D model file to be rendered.')
    parser.add_argument('--output_folder', type=str, default='/tmp', help='The path the output will be dumped to.')
    parser.add_argument('--resolution', type=int, default=512, help='Resolution of the images.')
    parser.add_argument('--engine', type=str, default='CYCLES', help='Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...')
    parser.add_argument('--geo_mode', action='store_true', help='Geometry mode for rendering.')
    parser.add_argument('--save_depth', action='store_true', help='Save the depth maps.')
    parser.add_argument('--save_normal', action='store_true', help='Save the normal maps.')
    parser.add_argument('--save_albedo', action='store_true', help='Save the albedo maps.')
    parser.add_argument('--save_mist', action='store_true', help='Save the mist distance maps.')
    parser.add_argument('--split_normal', action='store_true', help='Split the normals of the mesh.')
    parser.add_argument('--save_mesh', action='store_true', help='Save the mesh as a .ply file.')
    parser.add_argument('--skip_normalize', action='store_true', help='Skip scene normalization for already-normalized meshes.')
    parser.add_argument('--profile', action='store_true', help='Print timing information for render pipeline profiling.')
    parser.add_argument('--profile_disable_denoise', action='store_true', help='Profiling only: disable Cycles denoising.')
    parser.add_argument('--profile_no_write', action='store_true', help='Profiling only: render without write_still, then time image saving in post.')
    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    main(args)
    
