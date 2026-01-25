import pyvista as pv
import numpy as np
# 1. Load the mesh
mesh = pv.read("..\VelocityData3D\Velocity_06400.vtu")

# 2. Print what's inside (Check variable names!)
print("Available Data Arrays:", mesh.array_names)

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
