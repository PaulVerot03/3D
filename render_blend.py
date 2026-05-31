import os
import sys
# pyrefly: ignore [missing-import]
import bpy
# pyrefly: ignore [missing-import]
import addon_utils
import datetime
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

parser = argparse.ArgumentParser(description="Render a Blender template file directly.")
parser.add_argument('--blend', '-b', type=str, default='rna2.blend', help='Blender template file to load')
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

# Enable Molecular Nodes addon if installed/available
try:
    addon_utils.enable('bl_ext.user_default.molecularnodes')
except Exception as e:
    print(f"Warning: Could not enable Molecular Nodes: {e}")

# Load the template blend file if one is not already open
if not bpy.data.filepath and os.path.exists(blender_file):
    print(f"Loading template blend file '{blender_file}'...")
    bpy.ops.wm.open_mainfile(filepath=blender_file)

scene = bpy.context.scene
enable_gpu_if_available(scene)

# Determine the output folder
script_dir = os.path.dirname(os.path.abspath(__file__))
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

blend_filepath = bpy.data.filepath
if blend_filepath:
    blend_name = os.path.splitext(os.path.basename(blend_filepath))[0].replace("'", "")
else:
    blend_name = "default_render"

runs_parent_dir = os.path.join(script_dir, "render_runs", f"run_{timestamp}_{blend_name}")
output_dir = os.path.join(runs_parent_dir, blend_name)
os.makedirs(output_dir, exist_ok=True)

# Frame range
if parsed_args.start is not None:
    scene.frame_start = parsed_args.start

if parsed_args.end is not None:
    scene.frame_end = parsed_args.end

total_frames = scene.frame_end - scene.frame_start + 1
print(f"\n=========================================")
print(f"Rendering active scene: {blend_name}")
print(f"Frames: {scene.frame_start} to {scene.frame_end} ({total_frames} frames)")
print(f"=========================================")

for frame in tqdm(range(scene.frame_start, scene.frame_end + 1), desc=f"Rendering {blend_name}"):
    scene.frame_set(frame)
    
    # Force evaluation of modifier evaluations on meshes
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.modifiers:
            obj.data.update()
            
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    
    # Save frame image
    scene.render.filepath = os.path.join(output_dir, f"frame_{frame:04d}.png")
    bpy.ops.render.render(write_still=True)
    
print(f"Rendering complete!")

# Compile the rendered PNG sequence into an MP4 video using ffmpeg
video_output_path = os.path.join(output_dir, f"{blend_name}.mp4")
ffmpeg_cmd = [
    "ffmpeg",
    "-y",
    "-framerate", str(scene.render.fps),
    "-i", os.path.join(output_dir, "frame_%04d.png"),
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    video_output_path
]

print(f"Compiling video using ffmpeg...")
try:
    res = subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"Video successfully created: {video_output_path}")
except subprocess.CalledError as e:
    print(f"Error compiling video with ffmpeg: {e.stderr.decode()}")
