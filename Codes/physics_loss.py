import torch

def get_physics_loss(model, inputs):
    """
    Calculates the Navier-Stokes Residuals (The Physics Error).
    If the output is 0, the physics is perfect.
    """
    
    # 1. Enable Gradients for the Input
    # We need to track how x,y,z,t change to calculate derivatives
    inputs.requires_grad = True
    
    # 2. Get Predictions from the Model
    predictions = model(inputs)
    
    # Split predictions into u, v, w, p
    # predictions[:, 0:1] means get column 0 but keep it as a column
    u = predictions[:, 0:1]
    v = predictions[:, 1:2]
    w = predictions[:, 2:3]
    p = predictions[:, 3:4]
    
    # 3. CALCULATE DERIVATIVES (The Hard Part)
    # We use torch.autograd.grad to find gradients
    
    def get_grad(output_var, input_var):
        return torch.autograd.grad(
            output_var, input_var, 
            grad_outputs=torch.ones_like(output_var), 
            create_graph=True, # Essential for training
            retain_graph=True
        )[0]

    # Gradients of Velocity (u)
    du_dinput = get_grad(u, inputs)
    u_x = du_dinput[:, 0:1]
    u_y = du_dinput[:, 1:2]
    u_z = du_dinput[:, 2:3]
    u_t = du_dinput[:, 3:4]
    
    # Gradients of Velocity (v)
    dv_dinput = get_grad(v, inputs)
    v_x = dv_dinput[:, 0:1]
    v_y = dv_dinput[:, 1:2]
    v_z = dv_dinput[:, 2:3]
    v_t = dv_dinput[:, 3:4]
    
    # Gradients of Velocity (w)
    dw_dinput = get_grad(w, inputs)
    w_x = dw_dinput[:, 0:1]
    w_y = dw_dinput[:, 1:2]
    w_z = dw_dinput[:, 2:3]
    w_t = dw_dinput[:, 3:4]
    
    # Gradients of Pressure (p)
    dp_dinput = get_grad(p, inputs)
    p_x = dp_dinput[:, 0:1]
    p_y = dp_dinput[:, 1:2]
    p_z = dp_dinput[:, 2:3]

    # Second Derivatives (Laplacian) for Viscosity term
    # u_xx + u_yy + u_zz
    u_xx = get_grad(u_x, inputs)[:, 0:1]
    u_yy = get_grad(u_y, inputs)[:, 1:2]
    u_zz = get_grad(u_z, inputs)[:, 2:3]
    
    v_xx = get_grad(v_x, inputs)[:, 0:1]
    v_yy = get_grad(v_y, inputs)[:, 1:2]
    v_zz = get_grad(v_z, inputs)[:, 2:3]
    
    w_xx = get_grad(w_x, inputs)[:, 0:1]
    w_yy = get_grad(w_y, inputs)[:, 1:2]
    w_zz = get_grad(w_z, inputs)[:, 2:3]

    # 4. NAVIER-STOKES EQUATIONS
    # Momentum Equation in X, Y, Z directions
    
    # Read the Viscosity from the model (It learns this!)
    mu = model.viscosity

    # X-Momentum Residual
    f_u = u_t + (u*u_x + v*u_y + w*u_z) + p_x - mu*(u_xx + u_yy + u_zz)
    
    # Y-Momentum Residual
    f_v = v_t + (u*v_x + v*v_y + w*v_z) + p_y - mu*(v_xx + v_yy + v_zz)
    
    # Z-Momentum Residual
    f_w = w_t + (u*w_x + v*w_y + w*w_z) + p_z - mu*(w_xx + w_yy + w_zz)

    # Continuity Equation (Conservation of Mass: In = Out)
    f_mass = u_x + v_y + w_z

    # 5. Combine Errors (Mean Squared Error)
    loss_f = torch.mean(f_u**2) + torch.mean(f_v**2) + torch.mean(f_w**2) + torch.mean(f_mass**2)
    
    return loss_f