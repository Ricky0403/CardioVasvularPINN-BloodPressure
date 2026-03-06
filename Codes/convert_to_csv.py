# # import pyvista as pv
# # import pandas as pd
# # import numpy as np
# # import glob
# # import os

# # # --- Configuration ---
# # # Point this to your folder containing the 51 .vtu files
# # vtu_folder = r"../VelocityData3D" 
# # csv_output_name = "../cardiovascular_data.csv"
# # total_duration = 1.0 # The total physical time of the simulation

# # # --- Find and sort the files ---
# # search_path = os.path.join(vtu_folder, "*.vtu")
# # vtu_files = sorted(glob.glob(search_path))

# # if not vtu_files:
# #     print(f"Error: No .vtu files found in {vtu_folder}")
# #     exit()

# # print(f"Found {len(vtu_files)} VTU files. Starting conversion...")

# # dt = total_duration / len(vtu_files)
# # all_dataframes = []

# # # --- Extract Data Loop ---
# # for i, file_path in enumerate(vtu_files):
# #     file_name = os.path.basename(file_path)
# #     print(f"Extracting {file_name} (Time step {i+1}/{len(vtu_files)})...")
    
# #     # Load the mesh
# #     mesh = pv.read(file_path)
    
# #     # 1. Extract Coordinates
# #     points = mesh.points
# #     x = points[:, 0]
# #     y = points[:, 1]
# #     z = points[:, 2]
    
# #     # 2. Generate Time Array (same 't' for every point in this file)
# #     current_time = i * dt
# #     t = np.full(len(x), current_time)
    
# #     # 3. Extract Velocity (Handle uppercase/lowercase naming)
# #     if "Velocity" in mesh.point_data:
# #         velocity = mesh.point_data["Velocity"]
# #     elif "velocity" in mesh.point_data:
# #         velocity = mesh.point_data["velocity"]
# #     else:
# #         print(f"Warning: No velocity found in {file_name}")
# #         velocity = np.zeros((len(x), 3))
        
# #     u, v, w = velocity[:, 0], velocity[:, 1], velocity[:, 2]
    
# #     # 4. Extract Pressure
# #     if "Pressure" in mesh.point_data:
# #         p = mesh.point_data["Pressure"]
# #     elif "pressure" in mesh.point_data:
# #         p = mesh.point_data["pressure"]
# #     else:
# #         print(f"Warning: No pressure found in {file_name}")
# #         p = np.zeros(len(x))
        
# #     # 5. Build a temporary Pandas DataFrame for this specific file
# #     df_step = pd.DataFrame({
# #         'x': x, 'y': y, 'z': z, 't': t,
# #         'u': u, 'v': v, 'w': w, 'p': p
# #     })
    
# #     all_dataframes.append(df_step)

# # # --- Combine and Save ---
# # print("\nMerging all time steps into a single dataset...")
# # final_dataset = pd.concat(all_dataframes, ignore_index=True)

# # print(f"Total data points extracted: {len(final_dataset):,}")
# # print("Writing to CSV... (This might take a minute)")

# # final_dataset.to_csv(csv_output_name, index=False)
# # print(f"Success! Data saved to {csv_output_name}")


# import pyvista as pv
# import pandas as pd
# import numpy as np
# import os

# # --- Configuration ---
# # Point this directly to your wall file
# wall_file_path = r"../VelocityData3D/WallMesh/wall.vtp"
# csv_output_name = "../wall_boundary_data.csv"

# print(f"Loading wall mesh from: {wall_file_path}")

# if not os.path.exists(wall_file_path):
#     print("Error: Wall file not found. Please check the path.")
#     exit()

# # Load the PolyData mesh
# wall_mesh = pv.read(wall_file_path)

# # 1. Extract the physical coordinates
# points = wall_mesh.points
# x = points[:, 0]
# y = points[:, 1]
# z = points[:, 2]

# print(f"Successfully extracted {len(x):,} wall boundary coordinates.")

# # 2. Enforce the No-Slip Boundary Condition (Velocity = 0)
# u = np.zeros(len(x))
# v = np.zeros(len(x))
# w = np.zeros(len(x))

# # 3. Build the DataFrame
# # We leave out 't' and 'p' because the wall doesn't inherently have 
# # a single time step or a fixed pressure, just fixed velocity!
# df_wall = pd.DataFrame({
#     'x': x,
#     'y': y,
#     'z': z,
#     'u': u,
#     'v': v,
#     'w': w
# })

# # --- Save to CSV ---
# print("Writing wall data to CSV...")
# df_wall.to_csv(csv_output_name, index=False)
# print(f"Success! Wall boundary data saved to {csv_output_name}")


import pyvista as pv
import pandas as pd
import numpy as np
import glob
import os

# --- Configuration ---
vtu_folder = r"../VelocityData3D" 
wall_file_path = os.path.join(vtu_folder, "WallMesh", "wall.vtp")
csv_output_name = "../aikosh_unified_data.csv"
total_duration = 1.0 

# --- Find Files ---
vtu_files = sorted(glob.glob(os.path.join(vtu_folder, "*.vtu")))

if not vtu_files:
    print(f"Error: No .vtu files found in {vtu_folder}")
    exit()

if not os.path.exists(wall_file_path):
    print(f"Error: Wall file not found at {wall_file_path}")
    exit()

print(f"Found {len(vtu_files)} VTU files and 1 Wall VTP file.")

# --- Load the Wall Mesh Once ---
print("Loading static wall geometry...")
wall_mesh = pv.read(wall_file_path)
wall_x, wall_y, wall_z = wall_mesh.points[:, 0], wall_mesh.points[:, 1], wall_mesh.points[:, 2]

# The wall never moves (No-Slip Condition)
wall_u = np.zeros(len(wall_x))
wall_v = np.zeros(len(wall_y))
wall_w = np.zeros(len(wall_z))

# We leave pressure as NaN (Not a Number) because pressure is calculated 
# by the fluid dynamics, it is not a fixed boundary condition at the wall.
wall_p = np.full(len(wall_x), np.nan) 

dt = total_duration / len(vtu_files)
all_dataframes = []

# --- Extract Data Loop ---
for i, file_path in enumerate(vtu_files):
    file_name = os.path.basename(file_path)
    current_time = i * dt
    print(f"Processing time step {i+1}/{len(vtu_files)} (t={current_time:.3f})...")
    
    # 1. LOAD FLUID DATA
    mesh = pv.read(file_path)
    points = mesh.points
    f_x, f_y, f_z = points[:, 0], points[:, 1], points[:, 2]
    
    # Extract Velocity
    if "Velocity" in mesh.point_data:
        velocity = mesh.point_data["Velocity"]
    elif "velocity" in mesh.point_data:
        velocity = mesh.point_data["velocity"]
    else:
        velocity = np.zeros((len(f_x), 3))
    f_u, f_v, f_w = velocity[:, 0], velocity[:, 1], velocity[:, 2]
    
    # Extract Pressure
    if "Pressure" in mesh.point_data:
        f_p = mesh.point_data["Pressure"]
    elif "pressure" in mesh.point_data:
        f_p = mesh.point_data["pressure"]
    else:
        f_p = np.zeros(len(f_x))
        
    # Create Fluid DataFrame
    df_fluid = pd.DataFrame({
        'x': f_x, 'y': f_y, 'z': f_z, 't': current_time,
        'u': f_u, 'v': f_v, 'w': f_w, 'p': f_p,
        'is_wall': 0  # 0 means this is internal fluid
    })
    
    # 2. CREATE WALL DATA FOR THIS SPECIFIC TIME STEP
    df_wall = pd.DataFrame({
        'x': wall_x, 'y': wall_y, 'z': wall_z, 't': current_time,
        'u': wall_u, 'v': wall_v, 'w': wall_w, 'p': wall_p,
        'is_wall': 1  # 1 means this is the rigid boundary
    })
    
    # Append both to our master list
    all_dataframes.append(df_fluid)
    all_dataframes.append(df_wall)

# --- Combine and Save ---
print("\nMerging all fluid and boundary data into a single master dataset...")
final_dataset = pd.concat(all_dataframes, ignore_index=True)

print(f"Total rows generated: {len(final_dataset):,}")
print("Writing to CSV... (This will take a moment due to the file size)")

# Save to CSV (NaN values will automatically be left as empty cells, which is standard)
final_dataset.to_csv(csv_output_name, index=False)
print(f"Success! Master dataset saved to {csv_output_name}")