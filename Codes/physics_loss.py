"""
Physics losses for structured-grid cardiovascular flow prediction.
"""

import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
#  FINITE-DIFFERENCE LOSSES  (structured 3D grids)
# ═══════════════════════════════════════════════════════════════════════════

def _central_diff_padded(f, axis, dx=1.0):
    """
    Central finite difference along a spatial axis (axis: 2=X, 3=Y, 4=Z).
    Result is padded back to the original shape using edge replication,
    so all tensors remain (B, C, X, Y, Z) with no size change.
    F.pad for 5D tensors supports exactly 6 values:
        (z_before, z_after, y_before, y_after, x_before, x_after)
    """
    slc_fwd = [slice(None)] * 5
    slc_bwd = [slice(None)] * 5
    slc_fwd[axis] = slice(2, None)
    slc_bwd[axis] = slice(None, -2)
    diff = (f[tuple(slc_fwd)] - f[tuple(slc_bwd)]) / (2.0 * dx)

    # Build a 6-element pad tuple for the last 3 spatial dims (X, Y, Z)
    # F.pad order (innermost first): z, y, x
    # axis 4 = Z → pad positions 0,1
    # axis 3 = Y → pad positions 2,3
    # axis 2 = X → pad positions 4,5
    pad = [0, 0, 0, 0, 0, 0]
    if axis == 4:
        pad[0], pad[1] = 1, 1
    elif axis == 3:
        pad[2], pad[3] = 1, 1
    elif axis == 2:
        pad[4], pad[5] = 1, 1

    return F.pad(diff, pad, mode='replicate')


def _interior(t):
    """Trim 1 voxel from each side of all 3 spatial dims → interior only."""
    return t[:, :, 1:-1, 1:-1, 1:-1]


def fd_continuity_loss(pred, mask_dev, dx=1.0):
    """
    Divergence-free residual: ∂u/∂x + ∂v/∂y + ∂w/∂z = 0
    """
    u, v, w = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

    du_dx = _central_diff_padded(u, axis=2, dx=dx)
    dv_dy = _central_diff_padded(v, axis=3, dx=dx)
    dw_dz = _central_diff_padded(w, axis=4, dx=dx)

    div = _interior(du_dx) + _interior(dv_dy) + _interior(dw_dz)
    m   = _interior(mask_dev).float()

    n_valid = m.sum().clamp(min=1)
    return (div ** 2 * m).sum() / n_valid


def fd_momentum_loss(pred, prev_field, mask_dev, stats, dt=1.0, dx=1.0, viscosity=0.035):
    """
    Navier-Stokes momentum residual using finite differences.
    All derivatives computed at full resolution then cropped to interior.
    """
    def denorm(field):
        out = field.clone().float()
        for c in range(4):
            out[:, c:c+1] = out[:, c:c+1] * stats[f'std_{c}'] + stats[f'mean_{c}']
        return out

    pred_phys = denorm(pred)
    prev_phys = denorm(prev_field)

    u1, v1, w1, p1 = pred_phys[:,0:1], pred_phys[:,1:2], pred_phys[:,2:3], pred_phys[:,3:4]
    u0, v0, w0     = prev_phys[:,0:1], prev_phys[:,1:2], prev_phys[:,2:3]

    # Temporal derivatives (full resolution)
    du_dt = (u1 - u0) / dt
    dv_dt = (v1 - v0) / dt
    dw_dt = (w1 - w0) / dt

    # Spatial first derivatives (full resolution, padded)
    du_dx = _central_diff_padded(u1, 2, dx)
    du_dy = _central_diff_padded(u1, 3, dx)
    du_dz = _central_diff_padded(u1, 4, dx)

    dv_dx = _central_diff_padded(v1, 2, dx)
    dv_dy = _central_diff_padded(v1, 3, dx)
    dv_dz = _central_diff_padded(v1, 4, dx)

    dw_dx = _central_diff_padded(w1, 2, dx)
    dw_dy = _central_diff_padded(w1, 3, dx)
    dw_dz = _central_diff_padded(w1, 4, dx)

    dp_dx = _central_diff_padded(p1, 2, dx)
    dp_dy = _central_diff_padded(p1, 3, dx)
    dp_dz = _central_diff_padded(p1, 4, dx)

    # Laplacians (computed in-place at full resolution, zero at boundary)
    def laplacian(f):
        lap = torch.zeros_like(f)
        lap[:, :, 1:-1, :, :] += (f[:, :, 2:, :, :] - 2*f[:, :, 1:-1, :, :] + f[:, :, :-2, :, :]) / dx**2
        lap[:, :, :, 1:-1, :] += (f[:, :, :, 2:, :] - 2*f[:, :, :, 1:-1, :] + f[:, :, :, :-2, :]) / dx**2
        lap[:, :, :, :, 1:-1] += (f[:, :, :, :, 2:] - 2*f[:, :, :, :, 1:-1] + f[:, :, :, :, :-2]) / dx**2
        return lap

    lap_u = laplacian(u1)
    lap_v = laplacian(v1)
    lap_w = laplacian(w1)

    # Crop everything to interior — all tensors are now full resolution so
    # a single consistent crop gives matching shapes
    u_i     = _interior(u1);     v_i     = _interior(v1);     w_i     = _interior(w1)
    du_dt_i = _interior(du_dt);  dv_dt_i = _interior(dv_dt);  dw_dt_i = _interior(dw_dt)
    du_dx_i = _interior(du_dx);  du_dy_i = _interior(du_dy);  du_dz_i = _interior(du_dz)
    dv_dx_i = _interior(dv_dx);  dv_dy_i = _interior(dv_dy);  dv_dz_i = _interior(dv_dz)
    dw_dx_i = _interior(dw_dx);  dw_dy_i = _interior(dw_dy);  dw_dz_i = _interior(dw_dz)
    dp_dx_i = _interior(dp_dx);  dp_dy_i = _interior(dp_dy);  dp_dz_i = _interior(dp_dz)
    lap_u_i = _interior(lap_u);  lap_v_i = _interior(lap_v);  lap_w_i = _interior(lap_w)
    m       = _interior(mask_dev).float()

    # NS momentum residuals
    R_x = du_dt_i + (u_i*du_dx_i + v_i*du_dy_i + w_i*du_dz_i) + dp_dx_i - viscosity*lap_u_i
    R_y = dv_dt_i + (u_i*dv_dx_i + v_i*dv_dy_i + w_i*dv_dz_i) + dp_dy_i - viscosity*lap_v_i
    R_z = dw_dt_i + (u_i*dw_dx_i + v_i*dw_dy_i + w_i*dw_dz_i) + dp_dz_i - viscosity*lap_w_i

    n_valid = m.sum().clamp(min=1)
    loss_rx = (R_x**2 * m).sum() / n_valid
    loss_ry = (R_y**2 * m).sum() / n_valid
    loss_rz = (R_z**2 * m).sum() / n_valid

    # Normalize by characteristic scale so loss stays O(1)
    U_char = max(abs(stats.get('std_0', 1.0)),
                 abs(stats.get('std_1', 1.0)),
                 abs(stats.get('std_2', 1.0)))
    P_char = abs(stats.get('std_3', 1.0))
    char_scale = max(U_char**2 / max(dx, 1e-8),
                     P_char   / max(dx, 1e-8),
                     U_char   / max(dt, 1e-8),
                     1e-8)

    return (loss_rx + loss_ry + loss_rz) / (char_scale ** 2)


def fd_physics_loss(pred, prev_field, mask_dev, stats, dt=1.0, dx=1.0, viscosity=0.035):
    """Combined: continuity + momentum, both normalized to O(1)."""
    loss_cont = fd_continuity_loss(pred, mask_dev, dx)
    loss_mom  = fd_momentum_loss(pred, prev_field, mask_dev, stats, dt, dx, viscosity)
    # Equal weighting — both are already normalized, continuity×10 was too aggressive
    return loss_cont + loss_mom


def bc_loss(pred, wall_mask_dev):
    """No-slip: velocity = 0 at vessel walls."""
    vel_at_wall = pred[:, :3] * wall_mask_dev.float()
    return torch.mean(vel_at_wall ** 2)


def pressure_stability_loss(pred, mask_dev):
    """Penalize large pressure spatial gradients."""
    p = pred[:, 3:4]
    loss = (
        ((p[:,:,1:,:,:] - p[:,:,:-1,:,:])**2 * mask_dev[:,:,1:,:,:]).mean() +
        ((p[:,:,:,1:,:] - p[:,:,:,:-1,:])**2 * mask_dev[:,:,:,1:,:]).mean() +
        ((p[:,:,:,:,1:] - p[:,:,:,:,:-1])**2 * mask_dev[:,:,:,:,1:]).mean()
    )
    return loss



# ═══════════════════════════════════════════════════════════════════════════
#  AUTOGRAD-BASED LOSSES  (original PINN pipeline, unstructured points)
# ═══════════════════════════════════════════════════════════════════════════

def get_physics_loss(prediction, x_norm, viscosity, scales, ones_tensor):
    """
    Calculates Navier-Stokes Residuals using the Chain Rule for Normalized Data.
    (Retained for backwards compatibility with the unstructured PINN pipeline.)
    """
    s_u, s_v, s_w = scales['u'], scales['v'], scales['w']
    s_p = scales['p']
    s_x, s_y, s_z = scales['x'], scales['y'], scales['z']
    s_t = scales['t']

    min_u, min_v, min_w = scales['min_u'], scales['min_v'], scales['min_w']

    u_norm, v_norm, w_norm, p_norm = prediction[:,0:1], prediction[:,1:2], prediction[:,2:3], prediction[:,3:4]

    u_g = torch.autograd.grad(u_norm, x_norm, grad_outputs=ones_tensor, create_graph=True)[0]
    v_g = torch.autograd.grad(v_norm, x_norm, grad_outputs=ones_tensor, create_graph=True)[0]
    w_g = torch.autograd.grad(w_norm, x_norm, grad_outputs=ones_tensor, create_graph=True)[0]
    p_g = torch.autograd.grad(p_norm, x_norm, grad_outputs=ones_tensor, create_graph=True)[0]

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

    p_x = p_g[:, 0:1] * (s_p / s_x)
    p_y = p_g[:, 1:2] * (s_p / s_y)
    p_z = p_g[:, 2:3] * (s_p / s_z)

    u_x_raw, u_y_raw, u_z_raw = u_g[:, 0:1], u_g[:, 1:2], u_g[:, 2:3]
    v_x_raw, v_y_raw, v_z_raw = v_g[:, 0:1], v_g[:, 1:2], v_g[:, 2:3]
    w_x_raw, w_y_raw, w_z_raw = w_g[:, 0:1], w_g[:, 1:2], w_g[:, 2:3]

    u_xx = torch.autograd.grad(u_x_raw, x_norm, grad_outputs=ones_tensor, create_graph=True)[0][:, 0:1] * (s_u / s_x**2)
    u_yy = torch.autograd.grad(u_y_raw, x_norm, grad_outputs=ones_tensor, create_graph=True)[0][:, 1:2] * (s_u / s_y**2)
    u_zz = torch.autograd.grad(u_z_raw, x_norm, grad_outputs=ones_tensor, create_graph=True)[0][:, 2:3] * (s_u / s_z**2)
    v_xx = torch.autograd.grad(v_x_raw, x_norm, grad_outputs=ones_tensor, create_graph=True)[0][:, 0:1] * (s_v / s_x**2)
    v_yy = torch.autograd.grad(v_y_raw, x_norm, grad_outputs=ones_tensor, create_graph=True)[0][:, 1:2] * (s_v / s_y**2)
    v_zz = torch.autograd.grad(v_z_raw, x_norm, grad_outputs=ones_tensor, create_graph=True)[0][:, 2:3] * (s_v / s_z**2)
    w_xx = torch.autograd.grad(w_x_raw, x_norm, grad_outputs=ones_tensor, create_graph=True)[0][:, 0:1] * (s_w / s_x**2)
    w_yy = torch.autograd.grad(w_y_raw, x_norm, grad_outputs=ones_tensor, create_graph=True)[0][:, 1:2] * (s_w / s_y**2)
    w_zz = torch.autograd.grad(w_z_raw, x_norm, grad_outputs=ones_tensor, create_graph=True)[0][:, 2:3] * (s_w / s_z**2)

    u_real = (u_norm + 1.0) * s_u + min_u
    v_real = (v_norm + 1.0) * s_v + min_v
    w_real = (w_norm + 1.0) * s_w + min_w

    f_u = u_t + (u_real*u_x + v_real*u_y + w_real*u_z) + p_x - viscosity * (u_xx + u_yy + u_zz)
    f_v = v_t + (u_real*v_x + v_real*v_y + w_real*v_z) + p_y - viscosity * (v_xx + v_yy + v_zz)
    f_w = w_t + (u_real*w_x + v_real*w_y + w_real*w_z) + p_z - viscosity * (w_xx + w_yy + w_zz)
    f_c = u_x + v_y + w_z

    loss_f = torch.mean(f_u**2) + torch.mean(f_v**2) + torch.mean(f_w**2) + 10.0*torch.mean(f_c**2)
    return loss_f


def get_wss_loss(prediction_boundary, x_boundary, target_wss_real, viscosity, scales, ones_tensor):
    """Anchors the viscosity using the spatial derivatives at the wall in the physical domain."""
    s_u, s_v, s_w = scales['u'], scales['v'], scales['w']
    s_x, s_y, s_z = scales['x'], scales['y'], scales['z']

    u, v, w = prediction_boundary[:,0:1], prediction_boundary[:,1:2], prediction_boundary[:,2:3]

    u_g = torch.autograd.grad(u, x_boundary, grad_outputs=ones_tensor, create_graph=True)[0]
    v_g = torch.autograd.grad(v, x_boundary, grad_outputs=ones_tensor, create_graph=True)[0]
    w_g = torch.autograd.grad(w, x_boundary, grad_outputs=ones_tensor, create_graph=True)[0]

    u_y = u_g[:, 1:2] * (s_u / s_y)
    u_z = u_g[:, 2:3] * (s_u / s_z)
    v_x = v_g[:, 0:1] * (s_v / s_x)
    v_z = v_g[:, 2:3] * (s_v / s_z)
    w_x = w_g[:, 0:1] * (s_w / s_x)
    w_y = w_g[:, 1:2] * (s_w / s_y)

    pred_wss_x = viscosity * (u_y + v_x)
    pred_wss_y = viscosity * (v_z + w_y)
    pred_wss_z = viscosity * (u_z + w_x)

    pred_wss = torch.cat([pred_wss_x, pred_wss_y, pred_wss_z], dim=1)
    return torch.nn.functional.mse_loss(pred_wss, target_wss_real)
