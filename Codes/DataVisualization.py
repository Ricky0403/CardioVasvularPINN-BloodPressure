import pyvista as pv
import pandas as pd

# 1. Load the mesh
mesh = pv.read("..\VelocityData3D\Velocity_10000.vtu")

print("--- FILE CONTENTS ---")
print(f"Point Arrays Found: {mesh.point_data.keys()}")
print(f"Cell Arrays Found:  {mesh.cell_data.keys()}\n")

# 2. Initialize a dictionary with the 3D spatial coordinates
data_dict = {
    'X_coord': mesh.points[:, 0],
    'Y_coord': mesh.points[:, 1],
    'Z_coord': mesh.points[:, 2]
}

# 3. Dynamically loop through EVERY array attached to the points
for array_name in mesh.point_data.keys():
    array = mesh.point_data[array_name]
    
    # Check if the array is a 1D scalar (like Pressure)
    if len(array.shape) == 1 or array.shape[1] == 1:
        data_dict[array_name] = array.flatten()
        
    # Check if the array is a 2D/3D vector (like Velocity)
    else:
        # Unpack each component into its own column
        for i in range(array.shape[1]):
            # Name them nicely (e.g., velocity_X, velocity_Y, velocity_Z)
            axis_suffix = ['X', 'Y', 'Z'][i] if i < 3 else str(i)
            col_name = f"{array_name}_{axis_suffix}"
            data_dict[col_name] = array[:, i]

# 4. Build the master DataFrame
df = pd.DataFrame(data_dict)

print("--- FULL DATAFRAME PREVIEW ---")
# Use to_string() so pandas doesn't hide middle columns if there are a lot of them
print(df.head().to_string()) 

print("\n" + "-" * 50)
print(f"Total Points (Rows): {len(df):,}")
print(f"Total Features (Columns): {len(df.columns)}")

# 3. Visualization 1: The Pressure Field (Scalar)
mesh.plot(
    scalars="pressure",
    cmap="jet",
    # FIX: Use 'scalar_bar_args' dictionary instead of 'title' directly
    scalar_bar_args={'title': "Ground Truth Pressure (Pa)"},
    show_edges=False
)

# 4. Visualization 2: The Velocity Arrows (Vectors)
# We use 'glyphs' to turn points into arrows
arrows = mesh.glyph(scale="velocity", orient="velocity", factor=0.1)
arrows.plot(
    cmap="viridis",
    scalar_bar_args={'title': "Velocity Magnitude"}
)
