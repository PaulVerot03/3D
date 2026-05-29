import os
import sys
# pyrefly: ignore [missing-import]
import bpy
# pyrefly: ignore [missing-import]
import addon_utils
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import mathutils
import datetime
import glob
import subprocess

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, **kwargs):
        total = len(iterable) if hasattr(iterable, '__len__') else None
        for i, item in enumerate(iterable):
            if total is not None:
                print(f"{desc or 'Progress'}: {i+1}/{total}", flush=True)
            else:
                print(f"{desc or 'Progress'}: {i+1}", flush=True)
            yield item

import argparse

# Extract arguments after '--' if present
if "--" in sys.argv:
    args_list = sys.argv[sys.argv.index('--') + 1:]
else:
    args_list = []

parser = argparse.ArgumentParser(description="Render PDB trajectories in Blender.")
parser.add_argument('--blend', '-b', type=str, default='rna2.blend', help='Blender template file to load')
parser.add_argument('--trajectories', '-t', type=str, default='trajectories', help='Directory containing PDB files')
parser.add_argument('--start', '-s', type=int, default=None, help='Start frame for rendering')
parser.add_argument('--end', '-e', type=int, default=None, help='End frame for rendering')

parsed_args = parser.parse_args(args_list)
blender_file = parsed_args.blend


def enable_gpu_if_available(scene):
    if scene.render.engine == 'CYCLES':
        try:
            cycles_pref = bpy.context.preferences.addons['cycles'].preferences
            cycles_pref.refresh_devices()
            
            gpu_types = ('OPTIX', 'CUDA', 'HIP', 'ONEAPI', 'METAL')
            activated_type = None
            
            for gtype in gpu_types:
                devices = cycles_pref.get_devices_for_type(gtype)
                if devices:
                    for dev in devices:
                        dev.use = True
                        print(f"Cycles: Enabled GPU device: {dev.name} ({gtype})")
                    activated_type = gtype
                    break
            
            if activated_type:
                cycles_pref.compute_device_type = activated_type
                scene.cycles.device = 'GPU'
                print(f"Cycles: Successfully set rendering device to GPU ({activated_type})")
            else:
                scene.cycles.device = 'CPU'
                print("Cycles: No compatible GPU devices found. Using CPU.")
        except Exception as e:
            print(f"Warning: Failed to configure Cycles GPU settings: {e}")

addon_utils.enable('bl_ext.user_default.molecularnodes')

if not bpy.data.filepath and os.path.exists(blender_file):
    print("Loading " + blender_file + " template...")
    bpy.ops.wm.open_mainfile(filepath=blender_file)

script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.isabs(parsed_args.trajectories):
    combined_dir = parsed_args.trajectories
else:
    combined_dir = os.path.join(script_dir, parsed_args.trajectories)

print(f"Looking for PDB trajectories in: {combined_dir}")
pdb_files = [
    f for f in sorted(glob.glob(os.path.join(combined_dir, "*.pdb")))
    if not os.path.basename(f).startswith("combined_")
]

if not pdb_files:
    print(f"Error: No PDB files found in {combined_dir}")
    sys.exit(1)

scene = bpy.context.scene
enable_gpu_if_available(scene)

camera_obj = scene.camera
target_obj = None
if camera_obj:
    target_name = "CameraTarget"
    if target_name in bpy.data.objects:
        target_obj = bpy.data.objects[target_name]
    else:
        target_obj = bpy.data.objects.new(target_name, None)
        scene.collection.objects.link(target_obj)
    
    track_constraint = None
    for constraint in camera_obj.constraints:
        if constraint.type == 'TRACK_TO' and constraint.target == target_obj:
            track_constraint = constraint
            break
    if not track_constraint:
        track_constraint = camera_obj.constraints.new(type='TRACK_TO')
        track_constraint.target = target_obj
        track_constraint.track_axis = 'TRACK_NEGATIVE_Z'
        track_constraint.up_axis = 'UP_Y'
    camera_obj.data.lens = 36.0
    print(f"Camera '{camera_obj.name}' is tracking '{target_name}' empty.")
else:
    print("Warning: No active camera found in the scene to apply tracking.")

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
runs_parent_dir = os.path.join(script_dir, "render_runs", f"run_{timestamp}_{blender_file}")
os.makedirs(runs_parent_dir, exist_ok=True)
for pdb_path in pdb_files:
    pdb_name = os.path.splitext(os.path.basename(pdb_path))[0].replace("'", "")
    print(f"\n=========================================")
    print(f"Processing trajectory: {pdb_name}")
    print(f"=========================================")
    
    if "renders" in bpy.data.objects:
        obj_to_del = bpy.data.objects["renders"]
        mesh_to_del = obj_to_del.data
        bpy.data.objects.remove(obj_to_del, do_unlink=True)
        if mesh_to_del:
            bpy.data.meshes.remove(mesh_to_del, do_unlink=True)
            
    scene.mn.import_md_name = "renders"
    scene.mn.import_md_topology = pdb_path
    scene.mn.import_md_trajectory = pdb_path
    
    print(f"Importing PDB trajectory...")
    bpy.ops.mn.import_trajectory(
        topology=pdb_path,
        trajectory=pdb_path,
        name="renders"
    )
    
    imported_obj = bpy.context.active_object
    
    if imported_obj and imported_obj.type == 'MESH':
        MY_CUSTOM_TREE = "CustomProt"
        if MY_CUSTOM_TREE in bpy.data.node_groups:
            for mod in imported_obj.modifiers:
                if mod.type == 'NODES':
                    mod.node_group = bpy.data.node_groups[MY_CUSTOM_TREE]
                    print("Successfully applied CustomProt node tree style.")
                    break
        else:
            print(f"Warning: Node group '{MY_CUSTOM_TREE}' not found in the blend file.")
            
    output_dir = os.path.join(runs_parent_dir, pdb_name)
    os.makedirs(output_dir, exist_ok=True)
    
    scene.frame_start = 1
    
    if parsed_args.start is not None:
        scene.frame_start = parsed_args.start
    else:
        scene.frame_start = 1
        
    if parsed_args.end is not None:
        scene.frame_end = parsed_args.end
        
    total_frames = scene.frame_end - scene.frame_start + 1
    print(f"Rendering sequence for {pdb_name}: frames {scene.frame_start} to {scene.frame_end} ({total_frames} frames)...")
    halfway_frame = scene.frame_start + (scene.frame_end - scene.frame_start) // 2
    
    for frame in tqdm(range(scene.frame_start, scene.frame_end + 1)):
        scene.frame_set(frame)
        
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and obj.modifiers:
                obj.data.update()
                
        bpy.context.view_layer.update()
        
        if camera_obj and target_obj and imported_obj:
            try:
                dg = bpy.context.evaluated_depsgraph_get()
                eval_obj = imported_obj.evaluated_get(dg)
                eval_mesh = eval_obj.to_mesh()
                
                num_verts = len(eval_mesh.vertices)
                if num_verts > 0:
                    coords = np.empty(num_verts * 3, dtype=np.float32)
                    eval_mesh.vertices.foreach_get('co', coords)
                    coords = coords.reshape((-1, 3))
                    
                    R_world = np.array(imported_obj.matrix_world.to_3x3())
                    T_world = np.array(imported_obj.matrix_world.translation)
                    coords_world = coords @ R_world.T + T_world
                    
                    world_center = coords_world.mean(axis=0)
                    target_obj.location = mathutils.Vector(world_center)
                    
                    render = scene.render
                    aspect_ratio = (render.resolution_x * render.pixel_aspect_x) / (render.resolution_y * render.pixel_aspect_y)
                    
                    sensor_width = camera_obj.data.sensor_width
                    sensor_height = camera_obj.data.sensor_height
                    
                    if camera_obj.data.sensor_fit == 'AUTO':
                        if aspect_ratio >= 1.0:
                            w = sensor_width
                            h = sensor_width / aspect_ratio
                        else:
                            h = sensor_width
                            w = sensor_width * aspect_ratio
                    elif camera_obj.data.sensor_fit == 'HORIZONTAL':
                        w = sensor_width
                        h = sensor_width / aspect_ratio
                    elif camera_obj.data.sensor_fit == 'VERTICAL':
                        h = sensor_height
                        w = sensor_height * aspect_ratio
                    else:
                        w = sensor_width
                        h = sensor_height
                    
                    margin = 1.15
                    w_eff = w / margin
                    h_eff = h / margin
                    diff = coords_world - world_center
                    
                    x_cam = np.array(camera_obj.matrix_world.col[0].to_3d())
                    y_cam = np.array(camera_obj.matrix_world.col[1].to_3d())
                    z_cam = np.array(camera_obj.matrix_world.col[2].to_3d())
                    
                    x_local = diff @ x_cam
                    y_local = diff @ y_cam
                    z_local = diff @ z_cam
                    
                    f = camera_obj.data.lens
                    d_x = z_local + (2.0 * f / w_eff) * np.abs(x_local)
                    d_y = z_local + (2.0 * f / h_eff) * np.abs(y_local)
                    
                    d_required = np.max(np.maximum(d_x, d_y))
                    
                    d_required = max(float(d_required), 5.0)
                    
                    z_cam_vec = camera_obj.matrix_world.col[2].to_3d().normalized()
                    camera_obj.location = target_obj.location + d_required * z_cam_vec
                
                eval_obj.to_mesh_clear()
            except Exception as e:
                print(f"Warning: Failed to auto-frame camera: {e}")
            
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()
        
        scene.render.filepath = os.path.join(output_dir, f"frame_{frame:04d}.png")
        bpy.ops.render.render(write_still=True)
        
    print(f"Trajectory {pdb_name} rendering complete!")
    
    video_output_path = os.path.join(output_dir, f"{pdb_name}.mp4")
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(scene.render.fps),
        "-i", os.path.join(output_dir, "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        video_output_path
    ]
    
    print(f"Compiling video for '{pdb_name}' using ffmpeg...")
    try:
        res = subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Video successfully created: {video_output_path}")
    except subprocess.CalledError as e:
        print(f"Error compiling video with ffmpeg: {e.stderr.decode()}")

print("\nAll trajectories processed successfully!")