import os
import sys
import glob


parent_dir = "/home/paul/Documents/Evry/Stage/Optimize_3D_ARNStructure/outputs"

if len(sys.argv) > 1:
    parent_dir = sys.argv[1]

parent_dir = os.path.abspath(parent_dir)

script_dir = os.path.dirname(os.path.abspath(__file__))
output_folder = os.path.join(script_dir, "combined_trajectories")
os.makedirs(output_folder, exist_ok=True)

subdirs = [
    os.path.join(parent_dir, d)
    for d in os.listdir(parent_dir)
    if os.path.isdir(os.path.join(parent_dir, d)) and os.path.abspath(os.path.join(parent_dir, d)) != os.path.abspath(output_folder)
]

print(f"Scanning subdirectories in: {parent_dir}")
print(f"Outputs will be saved to: {output_folder}\n")

for subdir in sorted(subdirs):
    subdir_name = os.path.basename(subdir)
    pdb_files = sorted(glob.glob(os.path.join(subdir, "*.pdb")))
    
    pdb_files = [f for f in pdb_files if not os.path.basename(f).startswith("combined_")]
    
    if not pdb_files:
        print(f"Skipping {subdir_name}: No PDB files found.")
        continue
        
    output_file = os.path.join(output_folder, f"{subdir_name}.pdb")
    print(f"Combining {len(pdb_files)} files from '{subdir_name}' -> '{subdir_name}.pdb'...")
    
    with open(output_file, "w") as outfile:
        for i, file_path in enumerate(pdb_files):
            outfile.write(f"MODEL        {i+1}\n")
            with open(file_path, "r") as infile:
                for line in infile:
                    if not (line.startswith("END") or line.startswith("MASTER")):
                        outfile.write(line)
            outfile.write("ENDMDL\n")
            
    print(f"Successfully created: {output_file}\n")

print("All directories processed successfully!")
