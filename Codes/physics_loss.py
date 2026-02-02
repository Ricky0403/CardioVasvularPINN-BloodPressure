import torch

def get_physics_loss(prediction, x, viscosity, ones_tensor):
    """
    Calculates the Navier-Stokes Residuals (The Physics Error).
    If the output is 0, the physics is perfect.
    """
    
    # 1. Unpack existing predictions
    # prediction shape is (N, 4) -> u, v, w, p
    u = prediction[:, 0:1]
    v = prediction[:, 1:2]
    w = prediction[:, 2:3]
    p = prediction[:, 3:4]

    current_batch_size = x.shape[0]
    ones_tensor = ones_tensor[:current_batch_size]
    # 2. Calculate First Derivatives (gradients)
    # We use create_graph=True so we can take the derivative of this derivative later (for viscosity)
    u_g = torch.autograd.grad(u, x, grad_outputs=ones_tensor, create_graph=True)[0]
    v_g = torch.autograd.grad(v, x, grad_outputs=ones_tensor, create_graph=True)[0]
    w_g = torch.autograd.grad(w, x, grad_outputs=ones_tensor, create_graph=True)[0]
    p_g = torch.autograd.grad(p, x, grad_outputs=ones_tensor, create_graph=True)[0]

    # Unpack gradients: [du/dx, du/dy, du/dz, du/dt]
    u_x, u_y, u_z, u_t = u_g[:, 0:1], u_g[:, 1:2], u_g[:, 2:3], u_g[:, 3:4]
    v_x, v_y, v_z, v_t = v_g[:, 0:1], v_g[:, 1:2], v_g[:, 2:3], v_g[:, 3:4]
    w_x, w_y, w_z, w_t = w_g[:, 0:1], w_g[:, 1:2], w_g[:, 2:3], w_g[:, 3:4]
    
    p_x, p_y, p_z = p_g[:, 0:1], p_g[:, 1:2], p_g[:, 2:3]

    # 3. Calculate Second Derivatives (Viscosity term)
    # Laplacian u (u_xx + u_yy + u_zz)
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=ones_tensor, create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, x, grad_outputs=ones_tensor, create_graph=True)[0][:, 1:2]
    u_zz = torch.autograd.grad(u_z, x, grad_outputs=ones_tensor, create_graph=True)[0][:, 2:3]

    v_xx = torch.autograd.grad(v_x, x, grad_outputs=ones_tensor, create_graph=True)[0][:, 0:1]
    v_yy = torch.autograd.grad(v_y, x, grad_outputs=ones_tensor, create_graph=True)[0][:, 1:2]
    v_zz = torch.autograd.grad(v_z, x, grad_outputs=ones_tensor, create_graph=True)[0][:, 2:3]

    w_xx = torch.autograd.grad(w_x, x, grad_outputs=ones_tensor, create_graph=True)[0][:, 0:1]
    w_yy = torch.autograd.grad(w_y, x, grad_outputs=ones_tensor, create_graph=True)[0][:, 1:2]
    w_zz = torch.autograd.grad(w_z, x, grad_outputs=ones_tensor, create_graph=True)[0][:, 2:3]

    # 4. Navier-Stokes Equations (Incompressible)
    # Momentum u
    f_u = u_t + (u*u_x + v*u_y + w*u_z) + p_x - viscosity * (u_xx + u_yy + u_zz)
    # Momentum v
    f_v = v_t + (u*v_x + v*v_y + w*v_z) + p_y - viscosity * (v_xx + v_yy + v_zz)
    # Momentum w
    f_w = w_t + (u*w_x + v*w_y + w*w_z) + p_z - viscosity * (w_xx + w_yy + w_zz)
    # Continuity (Mass conservation)
    f_c = u_x + v_y + w_z

    # 5. Return Total Physics Loss
    loss_f = torch.mean(f_u**2) + torch.mean(f_v**2) + torch.mean(f_w**2) + torch.mean(f_c**2)
    
    return loss_f