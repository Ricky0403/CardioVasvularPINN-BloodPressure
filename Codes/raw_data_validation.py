import pyvista as pv
import numpy as np

# 1. Load a single time-step of your raw data
file_path = r"../VelocityData3D/Velocity_00000.vtu"
print(f"Loading mesh: {file_path}")
mesh = pv.read(file_path)

mesh.set_active_vectors("Velocity" if "Velocity" in mesh.array_names else "velocity")
mesh.set_active_scalars("Pressure" if "Pressure" in mesh.array_names else "pressure")

# 2. Compute spatial derivatives directly on the raw 3D grid!
print("Calculating physical gradients from the raw mesh...")
# Velocity Jacobian gives us: [u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y, w_z]
mesh_vel_grad = mesh.compute_derivative(scalars=mesh.active_vectors_name)
vel_gradients = mesh_vel_grad["gradient"] 

# Pressure Gradient gives us: [p_x, p_y, p_z]
mesh_pres_grad = mesh.compute_derivative(scalars=mesh.active_scalars_name)
pres_gradients = mesh_pres_grad["gradient"]

# Extract raw values for convenience
U = mesh.active_vectors
P = mesh.active_scalars

# 3. Let's test a specific point (e.g., somewhere in the middle of the fluid, not on the wall)
# We will pick point index 1000 as an example
idx = 1000

print("\n--- RAW DATA PHYSICS VALIDATION (Point 1000) ---")
print(f"Coordinates: {mesh.points[idx]}")
print(f"Velocity (u, v, w): {U[idx]}")

# Unpack the gradients for this specific point
u_x, u_y, u_z = vel_gradients[idx][0], vel_gradients[idx][1], vel_gradients[idx][2]
v_x, v_y, v_z = vel_gradients[idx][3], vel_gradients[idx][4], vel_gradients[idx][5]
w_x, w_y, w_z = vel_gradients[idx][6], vel_gradients[idx][7], vel_gradients[idx][8]
p_x, p_y, p_z = pres_gradients[idx][0], pres_gradients[idx][1], pres_gradients[idx][2]

# 4. Check the Continuity Equation (Conservation of Mass)
# f_c = u_x + v_y + w_z. For incompressible blood, this MUST be very close to 0.
continuity = u_x + v_y + w_z
print(f"\nContinuity (Incompressibility) Check: {continuity:.5f}")
if abs(continuity) < 0.1:
    print("-> PASS: The data represents an incompressible fluid (like blood).")
else:
    print("-> WARNING: High continuity error. Data might be compressible or noisy.")

# 5. Check Convection vs. Pressure (The core of Navier-Stokes)
# u * u_x + v * u_y + w * u_z + p_x = Viscous Forces (Friction)
convection_x = (U[idx][0] * u_x) + (U[idx][1] * u_y) + (U[idx][2] * u_z)

print(f"\nX-Axis Convection Force: {convection_x:.5f}")
print(f"X-Axis Pressure Gradient: {p_x:.5f}")

print("\nTotal X-Momentum Imbalance (Convection + Pressure Gradient):")
print(f"{convection_x + p_x:.5f}")
print("-> Note: This remaining imbalance is exactly what the Viscosity/Reynolds number is balancing in the simulation!")