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

# 0. Load the template blend file if one is not already open
if not bpy.data.filepath and os.path.exists("rna2.blend"):
    print("Loading rna2.blend template...")
    bpy.ops.wm.open_mainfile(filepath="rna2.blend")

# 1. Enable Molecular Nodes addon/extension
addon_utils.enable('bl_ext.user_default.molecularnodes')

# 2. Get the directory containing combined PDBs
script_dir = os.path.dirname(os.path.abspath(__file__))
combined_dir = os.path.join(script_dir, "trajectories")

# Parse command line arguments if passed after '--'
if "--" in sys.argv:
    args = sys.argv[sys.argv.index('--') + 1:]
    if args:
        combined_dir = os.path.abspath(args[0])

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

# Get or create the CameraTarget Empty (we only need one in the scene)
camera_obj = scene.camera
target_obj = None
if camera_obj:
    target_name = "CameraTarget"
    if target_name in bpy.data.objects:
        target_obj = bpy.data.objects[target_name]
    else:
        target_obj = bpy.data.objects.new(target_name, None)
        scene.collection.objects.link(target_obj)
    
    # Check if a Track To constraint already exists
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

# Create the main render runs folder
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
runs_parent_dir = os.path.join(script_dir, "render_runs", f"run_{timestamp}")
os.makedirs(runs_parent_dir, exist_ok=True)

# Loop through all PDB files
for pdb_path in pdb_files:
    pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]
    print(f"\n=========================================")
    print(f"Processing trajectory: {pdb_name}")
    print(f"=========================================")
    
    # Clean up previous "renders" object and mesh to avoid accumulation
    if "renders" in bpy.data.objects:
        obj_to_del = bpy.data.objects["renders"]
        mesh_to_del = obj_to_del.data
        bpy.data.objects.remove(obj_to_del, do_unlink=True)
        if mesh_to_del:
            bpy.data.meshes.remove(mesh_to_del, do_unlink=True)
            
    # Set Molecular Nodes UI paths (for compatibility)
    scene.mn.import_md_name = "renders"
    scene.mn.import_md_topology = pdb_path
    scene.mn.import_md_trajectory = pdb_path
    
    # Import the trajectory
    print(f"Importing PDB trajectory...")
    bpy.ops.mn.import_trajectory(
        topology=pdb_path,
        trajectory=pdb_path,
        name="renders"
    )
    
    imported_obj = bpy.context.active_object
    
    # Apply custom node tree
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
            
    # Set up rendering directory for this specific trajectory
    output_dir = os.path.join(runs_parent_dir, pdb_name)
    os.makedirs(output_dir, exist_ok=True)
    
    scene.frame_start = 1
    # scene.frame_end is automatically set by the Molecular Nodes import operator
    
    print(f"Rendering sequence for {pdb_name}: frames {scene.frame_start} to {scene.frame_end}...")
    halfway_frame = scene.frame_start + (scene.frame_end - scene.frame_start) // 2
    
    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)
        
        if camera_obj:
            camera_obj.data.lens = 36.0 if frame < halfway_frame else 8.0
        
        # Force evaluate modifier evaluations on the object
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and obj.modifiers:
                obj.data.update()
                
        bpy.context.view_layer.update()
        
        # Update target empty position to track the molecule center
        if target_obj and imported_obj and len(imported_obj.data.vertices) > 0:
            coords = np.empty(len(imported_obj.data.vertices) * 3, dtype=np.float32)
            imported_obj.data.vertices.foreach_get('co', coords)
            coords = coords.reshape((-1, 3))
            center = coords.mean(axis=0)
            world_center = imported_obj.matrix_world @ mathutils.Vector(center)
            target_obj.location = world_center
            
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()
        
        # Save frame image
        scene.render.filepath = os.path.join(output_dir, f"frame_{frame:04d}.png")
        bpy.ops.render.render(write_still=True)
        
    print(f"Trajectory {pdb_name} rendering complete!")
    
    # Compile the rendered PNG sequence into an MP4 video using ffmpeg
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