import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import your custom modules exactly like train.py
from normalizer import MinMaxNormalizer
from data_loader import DataLoader as PINN_DataLoader
from model import PINNModel as PINN

# ==========================================
# 1. MODIFIED PHYSICS FUNCTION FOR DIAGNOSTICS
# ==========================================
def get_raw_residuals(prediction, x_norm, viscosity, ones_tensor, scales):
    """
    Identical to your get_physics_loss, but returns the raw tensors 
    instead of the average mean, so we can inspect individual points!
    """
    s_u, s_v, s_w, s_p = scales['u'], scales['v'], scales['w'], scales['p']
    s_x, s_y, s_z, s_t = scales['x'], scales['y'], scales['z'], scales['t']

    u_norm, v_norm, w_norm, p_norm = prediction[:,0:1], prediction[:,1:2], prediction[:,2:3], prediction[:,3:4]
    
    # First Derivatives
    u_g = torch.autograd.grad(u_norm, x_norm, grad_outputs=ones_tensor, create_graph=True)[0]
    v_g = torch.autograd.grad(v_norm, x_norm, grad_outputs=ones_tensor, create_graph=True)[0]
    w_g = torch.autograd.grad(w_norm, x_norm, grad_outputs=ones_tensor, create_graph=True)[0]
    p_g = torch.autograd.grad(p_norm, x_norm, grad_outputs=ones_tensor, create_graph=True)[0]
    
    u_x, u_y, u_z, u_t = u_g[:,0:1]*(s_u/s_x), u_g[:,1:2]*(s_u/s_y), u_g[:,2:3]*(s_u/s_z), u_g[:,3:4]*(s_u/s_t)
    v_x, v_y, v_z, v_t = v_g[:,0:1]*(s_v/s_x), v_g[:,1:2]*(s_v/s_y), v_g[:,2:3]*(s_v/s_z), v_g[:,3:4]*(s_v/s_t)
    w_x, w_y, w_z, w_t = w_g[:,0:1]*(s_w/s_x), w_g[:,1:2]*(s_w/s_y), w_g[:,2:3]*(s_w/s_z), w_g[:,3:4]*(s_w/s_t)
    p_x, p_y, p_z = p_g[:,0:1]*(s_p/s_x), p_g[:,1:2]*(s_p/s_y), p_g[:,2:3]*(s_p/s_z)

    # Second Derivatives (Correctly Isolated!)
    u_x_raw, u_y_raw, u_z_raw = u_g[:, 0:1], u_g[:, 1:2], u_g[:, 2:3]
    v_x_raw, v_y_raw, v_z_raw = v_g[:, 0:1], v_g[:, 1:2], v_g[:, 2:3]
    w_x_raw, w_y_raw, w_z_raw = w_g[:, 0:1], w_g[:, 1:2], w_g[:, 2:3]

    u_xx = torch.autograd.grad(u_x_raw, x_norm, grad_outputs=ones_tensor, create_graph=True)[0][:,0:1] * (s_u/s_x**2)
    u_yy = torch.autograd.grad(u_y_raw, x_norm, grad_outputs=ones_tensor, create_graph=True)[0][:,1:2] * (s_u/s_y**2)
    u_zz = torch.autograd.grad(u_z_raw, x_norm, grad_outputs=ones_tensor, create_graph=True)[0][:,2:3] * (s_u/s_z**2)

    v_xx = torch.autograd.grad(v_x_raw, x_norm, grad_outputs=ones_tensor, create_graph=True)[0][:,0:1] * (s_v/s_x**2)
    v_yy = torch.autograd.grad(v_y_raw, x_norm, grad_outputs=ones_tensor, create_graph=True)[0][:,1:2] * (s_v/s_y**2)
    v_zz = torch.autograd.grad(v_z_raw, x_norm, grad_outputs=ones_tensor, create_graph=True)[0][:,2:3] * (s_v/s_z**2)

    w_xx = torch.autograd.grad(w_x_raw, x_norm, grad_outputs=ones_tensor, create_graph=True)[0][:,0:1] * (s_w/s_x**2)
    w_yy = torch.autograd.grad(w_y_raw, x_norm, grad_outputs=ones_tensor, create_graph=True)[0][:,1:2] * (s_w/s_y**2)
    w_zz = torch.autograd.grad(w_z_raw, x_norm, grad_outputs=ones_tensor, create_graph=True)[0][:,2:3] * (s_w/s_z**2)

    u_real, v_real, w_real = u_norm * s_u, v_norm * s_v, w_norm * s_w

    # Navier-Stokes Equations
    f_u = u_t + (u_real*u_x + v_real*u_y + w_real*u_z) + p_x - viscosity * (u_xx + u_yy + u_zz)
    f_v = v_t + (u_real*v_x + v_real*v_y + w_real*v_z) + p_y - viscosity * (v_xx + v_yy + v_zz)
    f_w = w_t + (u_real*w_x + v_real*w_y + w_real*w_z) + p_z - viscosity * (w_xx + w_yy + w_zz)
    f_c = u_x + v_y + w_z 

    return f_u, f_v, f_w, f_c

# ==========================================
# 2. SETUP AND DATA LOADING
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load exactly as you do in train.py
data_path = r"../VelocityData3D" 
wall_path = os.path.join(data_path, "WallMesh", "wall.vtp")
raw_loader = PINN_DataLoader(data_path, wall_path)
X, U, P = raw_loader.load(time_step=1.0/len(raw_loader.files))

spatial_normalizer = MinMaxNormalizer(X, method='column-wise', device=device)
velocity_normalizer = MinMaxNormalizer(U, method='global', device=device)
pressure_normalizer = MinMaxNormalizer(P, method='global', device=device)

scales = {
    'x': (spatial_normalizer.max[0] - spatial_normalizer.min[0]) / 2.0,
    'y': (spatial_normalizer.max[1] - spatial_normalizer.min[1]) / 2.0,
    'z': (spatial_normalizer.max[2] - spatial_normalizer.min[2]) / 2.0,
    't': (spatial_normalizer.max[3] - spatial_normalizer.min[3]) / 2.0,
    'd': (spatial_normalizer.max[4] - spatial_normalizer.min[4]) / 2.0,
    'u': (velocity_normalizer.max - velocity_normalizer.min) / 2.0,
    'v': (velocity_normalizer.max - velocity_normalizer.min) / 2.0,
    'w': (velocity_normalizer.max - velocity_normalizer.min) / 2.0,
    'p': (pressure_normalizer.max - pressure_normalizer.min) / 2.0
}
for key in scales:
    scales[key] = torch.tensor(scales[key]).to(device) if not isinstance(scales[key], torch.Tensor) else scales[key].to(device)

# Load the model
model = PINN(layers=[5, 64, 64, 64, 64, 64, 64, 64, 4], activation=nn.SiLU()).to(device)

# ==========================================
# 3. TEST EXACTLY 100 POINTS
# ==========================================
# Grab the first 100 points
NUM_POINTS = 100
X_test = spatial_normalizer.encode(X[:NUM_POINTS]).to(device)
X_test.requires_grad_(True)

ones_wrapper = torch.ones((NUM_POINTS, 1), device=device, requires_grad=False)

# Forward Pass
predictions = model(X_test)
current_visc = F.softplus(model.viscosity)

# Get Raw Physics Errors
f_u, f_v, f_w, f_c = get_raw_residuals(predictions, X_test, current_visc, ones_wrapper, scales)

print("\n--- NAVIER-STOKES RESIDUAL ANALYSIS ---")
print("Target: A perfect physical model will have residuals equal to 0.0000")
print("-" * 50)

for i in range(5): # Let's just print the first 5 so it doesn't flood your screen
    print(f"Point {i+1}:")
    print(f"  Coordinates (x,y,z,t,d): {X[i].tolist()}")
    print(f"  X-Momentum Error (f_u) : {f_u[i].item():.5f}")
    print(f"  Y-Momentum Error (f_v) : {f_v[i].item():.5f}")
    print(f"  Z-Momentum Error (f_w) : {f_w[i].item():.5f}")
    print(f"  Continuity Error (f_c) : {f_c[i].item():.5f}")
    print("-" * 50)

print(f"Average Absolute f_u error across all 100 points: {torch.mean(torch.abs(f_u)).item():.5f}")
print(f"Average Absolute f_v error across all 100 points: {torch.mean(torch.abs(f_v)).item():.5f}")
print(f"Average Absolute f_w error across all 100 points: {torch.mean(torch.abs(f_w)).item():.5f}")
print(f"Average Absolute f_c error across all 100 points: {torch.mean(torch.abs(f_c)).item():.5f}")