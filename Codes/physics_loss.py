import torch

def get_physics_loss(prediction, x_norm, viscosity, ones_tensor, scales):
    """
    Calculates Navier-Stokes Residuals using the Chain Rule for Normalized Data.
    """
    
    # 1. Unpack Scales (Real World Range / Normalized Range)
    s_u, s_v, s_w = scales['u'], scales['v'], scales['w']
    s_p = scales['p']
    s_x, s_y, s_z = scales['x'], scales['y'], scales['z']
    s_t = scales['t'] 

    # 2. Get Normalized Predictions
    u_norm, v_norm, w_norm, p_norm = prediction[:,0:1], prediction[:,1:2], prediction[:,2:3], prediction[:,3:4]
    
    # Ensure ones_tensor matches batch size
    current_batch_size = x_norm.shape[0]
    ones_tensor = ones_tensor[:current_batch_size]

    # 3. Calculate Normalized Gradients (The network's raw guess)
    u_g = torch.autograd.grad(u_norm, x_norm, grad_outputs=ones_tensor, create_graph=True)[0]
    v_g = torch.autograd.grad(v_norm, x_norm, grad_outputs=ones_tensor, create_graph=True)[0]
    w_g = torch.autograd.grad(w_norm, x_norm, grad_outputs=ones_tensor, create_graph=True)[0]
    p_g = torch.autograd.grad(p_norm, x_norm, grad_outputs=ones_tensor, create_graph=True)[0]
    
    # First Derivatives (Velocity)
    u_x = u_g[:, 0:1] * (s_u / s_x)
    u_y = u_g[:, 1:2] * (s_u / s_y)
    u_z = u_g[:, 2:3] * (s_u / s_z)
    u_t = u_g[:, 3:4] * (s_u / s_t)

    v_x = v_g[:, 0:1] * (s_v / s_x)
    v_y = v_g[:, 1:2] * (s_v / s_y)
    v_z = v_g[:, 2:3] * (s_v / s_z)
    v_t = v_g[:, 3:4] * (s_v / s_t)

    w_x = w_g[:, 0:1] * (s_w / s_x)
    w_y = w_g[:, 1:2] * (s_w / s_y)
    w_z = w_g[:, 2:3] * (s_w / s_z)
    w_t = w_g[:, 3:4] * (s_w / s_t)

    # First Derivatives (Pressure)
    p_x = p_g[:, 0:1] * (s_p / s_x)
    p_y = p_g[:, 1:2] * (s_p / s_y)
    p_z = p_g[:, 2:3] * (s_p / s_z)

    # 4. Second Derivatives (Viscosity term)
    
    # Gradients of the normalized first derivatives
    u_gg = torch.autograd.grad(u_g, x_norm, grad_outputs=torch.ones_like(u_g), create_graph=True)[0]
    v_gg = torch.autograd.grad(v_g, x_norm, grad_outputs=torch.ones_like(v_g), create_graph=True)[0]
    w_gg = torch.autograd.grad(w_g, x_norm, grad_outputs=torch.ones_like(w_g), create_graph=True)[0]

    u_xx = u_gg[:, 0:1] * (s_u / s_x**2)
    u_yy = u_gg[:, 1:2] * (s_u / s_y**2)
    u_zz = u_gg[:, 2:3] * (s_u / s_z**2)

    v_xx = v_gg[:, 0:1] * (s_v / s_x**2)
    v_yy = v_gg[:, 1:2] * (s_v / s_y**2)
    v_zz = v_gg[:, 2:3] * (s_v / s_z**2)

    w_xx = w_gg[:, 0:1] * (s_w / s_x**2)
    w_yy = w_gg[:, 1:2] * (s_w / s_y**2)
    w_zz = w_gg[:, 2:3] * (s_w / s_z**2)

    # 5. Reconstruct Real Values for Interaction Terms
    u_real = u_norm * s_u
    v_real = v_norm * s_v
    w_real = w_norm * s_w

    # 6. Navier-Stokes Equations (Incompressible)
    f_u = u_t + (u_real*u_x + v_real*u_y + w_real*u_z) + p_x - viscosity * (u_xx + u_yy + u_zz)
    f_v = v_t + (u_real*v_x + v_real*v_y + w_real*v_z) + p_y - viscosity * (v_xx + v_yy + v_zz)
    f_w = w_t + (u_real*w_x + v_real*w_y + w_real*w_z) + p_z - viscosity * (w_xx + w_yy + w_zz)
    f_c = u_x + v_y + w_z 

    # 7. Return Total Physics Loss
    loss_f = torch.mean(f_u**2) + torch.mean(f_v**2) + torch.mean(f_w**2) + 10.0*torch.mean(f_c**2)
    
    return loss_f