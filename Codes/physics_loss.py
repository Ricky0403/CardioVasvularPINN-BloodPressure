import torch
from torch.func import functional_call, vmap, jacrev, hessian

# FIX 2: Added 'prediction' to the arguments
def get_physics_loss(model, x_norm, prediction, viscosity, scales):
    """
    Calculates Navier-Stokes Residuals using high-speed torch.func (vmap/jacrev/hessian).
    """
    
    # 1. Unpack Scales 
    s_u, s_v, s_w = scales['u'], scales['v'], scales['w']
    s_p = scales['p']
    s_x, s_y, s_z = scales['x'], scales['y'], scales['z']
    s_t = scales['t'] 

    # 2. Extract model state for stateless functional execution
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    # 3. Define a "Pure Function" for a single data point
    def forward_single(x_single):
        # x_single shape: (4,) -> [x, y, z, t]
        pred = functional_call(model, (params, buffers), x_single.unsqueeze(0))
        return pred.squeeze(0)

    # 4. Define the Calculus Operations (Jacobian for 1st derivatives, Hessian for 2nd)
    compute_J = jacrev(forward_single, argnums=0)
    compute_H = hessian(forward_single, argnums=0)

    # 5. Vectorize across the entire batch size simultaneously
    J_batch = vmap(compute_J, in_dims=(0,))(x_norm)  
    H_batch = vmap(compute_H, in_dims=(0,))(x_norm)  

    # 6. Use the pre-computed prediction tensor instead of running the model again!
    u_norm, v_norm, w_norm = prediction[:, 0], prediction[:, 1], prediction[:, 2]
    
    # 7. Scale the derivatives according to the normalization factors
    u_x = J_batch[:, 0, 0] * (s_u / s_x)
    u_y = J_batch[:, 0, 1] * (s_u / s_y)
    u_z = J_batch[:, 0, 2] * (s_u / s_z)
    u_t = J_batch[:, 0, 3] * (s_u / s_t)

    v_x = J_batch[:, 1, 0] * (s_v / s_x)
    v_y = J_batch[:, 1, 1] * (s_v / s_y)
    v_z = J_batch[:, 1, 2] * (s_v / s_z)
    v_t = J_batch[:, 1, 3] * (s_v / s_t)

    w_x = J_batch[:, 2, 0] * (s_w / s_x)
    w_y = J_batch[:, 2, 1] * (s_w / s_y)
    w_z = J_batch[:, 2, 2] * (s_w / s_z)
    w_t = J_batch[:, 2, 3] * (s_w / s_t)

    p_x = J_batch[:, 3, 0] * (s_p / s_x)
    p_y = J_batch[:, 3, 1] * (s_p / s_y)
    p_z = J_batch[:, 3, 2] * (s_p / s_z)

    # 8. Second Derivatives (Scaled Laplacians)
    u_xx = H_batch[:, 0, 0, 0] * (s_u / s_x**2)
    u_yy = H_batch[:, 0, 1, 1] * (s_u / s_y**2)
    u_zz = H_batch[:, 0, 2, 2] * (s_u / s_z**2)

    v_xx = H_batch[:, 1, 0, 0] * (s_v / s_x**2)
    v_yy = H_batch[:, 1, 1, 1] * (s_v / s_y**2)
    v_zz = H_batch[:, 1, 2, 2] * (s_v / s_z**2)

    w_xx = H_batch[:, 2, 0, 0] * (s_w / s_x**2)
    w_yy = H_batch[:, 2, 1, 1] * (s_w / s_y**2)
    w_zz = H_batch[:, 2, 2, 2] * (s_w / s_z**2)

    # 9. Reconstruct Real Values for Interaction Terms
    u_real = u_norm * s_u
    v_real = v_norm * s_v
    w_real = w_norm * s_w

    # 10. Navier-Stokes Equations (Incompressible)
    f_u = u_t + (u_real*u_x + v_real*u_y + w_real*u_z) + p_x - viscosity * (u_xx + u_yy + u_zz)
    f_v = v_t + (u_real*v_x + v_real*v_y + w_real*v_z) + p_y - viscosity * (v_xx + v_yy + v_zz)
    f_w = w_t + (u_real*w_x + v_real*w_y + w_real*w_z) + p_z - viscosity * (w_xx + w_yy + w_zz)
    f_c = u_x + v_y + w_z 

    # 11. Return Total Physics Loss
    loss_f = torch.mean(f_u**2) + torch.mean(f_v**2) + torch.mean(f_w**2) + 10.0*torch.mean(f_c**2)
    
    return loss_f