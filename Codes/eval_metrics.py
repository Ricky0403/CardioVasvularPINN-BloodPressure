"""
eval_metrics.py  —  Comprehensive evaluation of the PINN blood-flow model.

Produces every metric category required to impress IIT external examiners:

  1. Data accuracy     — Rel-L2, NMAE, RMSE for velocity (u,v,w) and pressure
  2. WSS accuracy      — predicted vs ground-truth wall shear stress (3 components)
  3. Learned viscosity — how close model.viscosity is to real blood (0.004 Pa·s)
  4. Physics residuals — NS momentum (f_u, f_v, f_w) and continuity (f_c) via autograd
  5. Boundary compliance — velocity at wall points (no-slip)
  6. Inference speed   — ms per query point vs CFD
  7. Per-timestep breakdown — accuracy at each cardiac phase

Outputs
  ../Results/pinn_metrics_report.json
  ../Results/pinn_metrics_table.csv
  ../Results/pinn_plots/             ← all PNGs

IMPORTANT: This script reloads and refits the normalizers exactly as train.py
does, because the checkpoint does not save normalizer state.
"""

import json
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import DataLoader as PINN_DataLoader
from model import PINNModel as PINN
from normalizer import MinMaxNormalizer
from physics_loss import get_physics_loss

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
#  PATHS  —  must match train.py exactly
# ═══════════════════════════════════════════════════════════════════════════
DATA_PATH    = "../VelocityData3D"
WALL_PATH    = "../VelocityData3D/WallMesh/wall.vtp"
RESULTS_DIR  = "../Results"
PLOT_DIR     = os.path.join(RESULTS_DIR, "pinn_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

CHECKPOINT_PATH  = "../Models/pinn_checkpoint.pth"
FINAL_MODEL_PATH = "../Models/pinn_model_tanh.pth"
MODEL_PATH = FINAL_MODEL_PATH if os.path.exists(FINAL_MODEL_PATH) else CHECKPOINT_PATH

TIME_STEP    = 0.2          # must match train.py
BATCH_SIZE   = 15000        # for batched inference (avoids OOM)
TRUE_VISCOSITY = 0.004      # Pa·s — true dynamic viscosity of blood

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"Loading model from: {MODEL_PATH}\n")


# ═══════════════════════════════════════════════════════════════════════════
#  1. DATA LOADING + NORMALIZER REFIT  (exact mirror of train.py)
# ═══════════════════════════════════════════════════════════════════════════
print("Loading data and refitting normalizers...")
loader = PINN_DataLoader(DATA_PATH, WALL_PATH)
coords_t, vel, pres, wss_gt, b_mask = loader.load(time_step=TIME_STEP)

norm_coords = MinMaxNormalizer(coords_t, method='column-wise', device=DEVICE)
norm_vel    = MinMaxNormalizer(vel,      method='global',      device=DEVICE)
norm_pres   = MinMaxNormalizer(pres,     method='global',      device=DEVICE)
norm_wss    = MinMaxNormalizer(wss_gt,   method='global',      device=DEVICE)

X      = norm_coords.encode(coords_t).to(DEVICE)[:, :4]   # <--- SLICE TO 4 COLUMNS
Y_vel  = norm_vel.encode(vel).to(DEVICE)            # (N, 3) normalised velocity
Y_pres = norm_pres.encode(pres).to(DEVICE)          # (N, 1) normalised pressure
b_mask = b_mask.to(DEVICE).squeeze()               # (N,)   wall boundary mask

N = X.shape[0]
print(f"Total points: {N:,}  |  boundary points: {b_mask.sum().item():,}\n")


# ═══════════════════════════════════════════════════════════════════════════
#  2. SCALES  (exact mirror of train.py)
# ═══════════════════════════════════════════════════════════════════════════
def get_range(normalizer, dim=None):
    diff = normalizer.max - normalizer.min
    if normalizer.method == 'column-wise':
        return diff[dim].item() / 2.0
    return diff.item() / 2.0

def get_min(normalizer, dim=None):
    if normalizer.method == 'column-wise':
        return normalizer.min[dim].item()
    return normalizer.min.item()

scales = {
    'x': get_range(norm_coords, 0),
    'y': get_range(norm_coords, 1),
    'z': get_range(norm_coords, 2),
    't': get_range(norm_coords, 3),
    'u': get_range(norm_vel),
    'v': get_range(norm_vel),
    'w': get_range(norm_vel),
    'p': get_range(norm_pres),
    'min_u': get_min(norm_vel),
    'min_v': get_min(norm_vel),
    'min_w': get_min(norm_vel),
}


# ═══════════════════════════════════════════════════════════════════════════
#  3. MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════
model = PINN(layers=[4, 64, 64, 64, 64, 64, 64, 64, 4],
             activation=nn.Tanh()).to(DEVICE)

ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
if 'model_state_dict' in ckpt:
    model.load_state_dict(ckpt['model_state_dict'])
    trained_epoch = ckpt.get('epoch', '?')
    print(f"Checkpoint loaded (epoch {trained_epoch})")
else:
    model.load_state_dict(ckpt)
    print("Final model state dict loaded")

model.eval()
n_params = sum(p.numel() for p in model.parameters())
learned_visc = F.softplus(model.viscosity).item()
print(f"Parameters   : {n_params:,}")
print(f"Learned μ    : {learned_visc:.6f} Pa·s  (true blood: {TRUE_VISCOSITY:.4f} Pa·s)")
print(f"Viscosity err: {abs(learned_visc - TRUE_VISCOSITY)/TRUE_VISCOSITY*100:.2f}%\n")


# ═══════════════════════════════════════════════════════════════════════════
#  HELPER: batched inference (avoids OOM on large datasets)
# ═══════════════════════════════════════════════════════════════════════════
def batch_predict(X_in, batch_size=BATCH_SIZE, requires_grad=False):
    """Run model inference in batches. Returns (N,4) predictions."""
    preds = []
    for i in range(0, X_in.shape[0], batch_size):
        xb = X_in[i:i+batch_size]
        if requires_grad:
            xb = xb.clone().detach().requires_grad_(True)
        with torch.set_grad_enabled(requires_grad):
            preds.append(model(xb))
    return torch.cat(preds, dim=0)


# ═══════════════════════════════════════════════════════════════════════════
#  METRIC HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def rel_l2(pred, target):
    """Relative L2 error."""
    return (torch.norm(pred - target) / (torch.norm(target) + 1e-8)).item()

def nmae_pct(pred, target):
    """Normalised MAE as percentage: mean|err| / (max−min) × 100."""
    rng = (target.max() - target.min()).clamp(min=1e-8)
    return (torch.abs(pred - target).mean() / rng * 100).item()

def rmse(pred, target):
    return torch.sqrt(F.mse_loss(pred, target)).item()


# ═══════════════════════════════════════════════════════════════════════════
#  4. FULL DATASET ACCURACY  (velocity + pressure)
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  Computing full-dataset accuracy...")
print("=" * 60)

t0 = time.perf_counter()
with torch.no_grad():
    pred_all = batch_predict(X)
inference_time_total = time.perf_counter() - t0
inference_ms_per_point = inference_time_total / N * 1e6   # μs per point (PINN is per-point)

pred_vel_norm  = pred_all[:, 0:3]
pred_pres_norm = pred_all[:, 3:4]

# Denormalise to physical units for reporting
pred_vel_phys  = norm_vel.decode(pred_vel_norm)
pred_pres_phys = norm_pres.decode(pred_pres_norm)
true_vel_phys  = vel.to(DEVICE)
true_pres_phys = pres.to(DEVICE)

# Velocity per-component
vel_components = ['u (x)', 'v (y)', 'w (z)']
vel_rel_l2  = [rel_l2( pred_vel_phys[:,i], true_vel_phys[:,i]) for i in range(3)]
vel_nmae    = [nmae_pct(pred_vel_phys[:,i], true_vel_phys[:,i]) for i in range(3)]
vel_rmse    = [rmse(   pred_vel_phys[:,i], true_vel_phys[:,i]) for i in range(3)]

# Velocity magnitude (combined)
pred_vmag = torch.norm(pred_vel_phys, dim=1)
true_vmag = torch.norm(true_vel_phys, dim=1)
vmag_rel_l2 = rel_l2(pred_vmag, true_vmag)
vmag_nmae   = nmae_pct(pred_vmag, true_vmag)
vmag_rmse   = rmse(pred_vmag, true_vmag)

# Pressure
pres_rel_l2 = rel_l2( pred_pres_phys, true_pres_phys)
pres_nmae   = nmae_pct(pred_pres_phys, true_pres_phys)
pres_rmse   = rmse(   pred_pres_phys, true_pres_phys)

print(f"\n  ── Velocity accuracy (physical units) ──────────────────")
for i, c in enumerate(vel_components):
    print(f"  {c:8s} | Rel-L2 {vel_rel_l2[i]:.5f} | "
          f"NMAE {vel_nmae[i]:.3f}% | RMSE {vel_rmse[i]:.5f}")
print(f"  |u| mag  | Rel-L2 {vmag_rel_l2:.5f} | "
      f"NMAE {vmag_nmae:.3f}% | RMSE {vmag_rmse:.5f}")

print(f"\n  ── Pressure accuracy (physical units) ──────────────────")
print(f"  Pressure | Rel-L2 {pres_rel_l2:.5f} | "
      f"NMAE {pres_nmae:.3f}% | RMSE {pres_rmse:.5f}")


# ═══════════════════════════════════════════════════════════════════════════
#  5. WALL SHEAR STRESS ACCURACY  (ground truth from data_loader)
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n  ── Wall shear stress accuracy ───────────────────────────")

# Predict at boundary points with grad for WSS computation
x_bnd = X[b_mask].clone().detach().requires_grad_(True)
ones_bnd = torch.ones(x_bnd.shape[0], 1, device=DEVICE)
pred_bnd = model(x_bnd)

# Compute predicted WSS using same formula as physics_loss.get_wss_loss.
# retain_graph=True on first two calls — all three outputs (u_b, v_b, w_b) share
# the same compute graph from model(x_bnd). autograd.grad frees the graph after
# the final call; retain_graph keeps it alive for the intermediate ones.
u_b, v_b, w_b = pred_bnd[:,0:1], pred_bnd[:,1:2], pred_bnd[:,2:3]
u_g = torch.autograd.grad(u_b, x_bnd, grad_outputs=ones_bnd, create_graph=False, retain_graph=True)[0]
v_g = torch.autograd.grad(v_b, x_bnd, grad_outputs=ones_bnd, create_graph=False, retain_graph=True)[0]
w_g = torch.autograd.grad(w_b, x_bnd, grad_outputs=ones_bnd, create_graph=False, retain_graph=False)[0]

sx, sy, sz = scales['x'], scales['y'], scales['z']
su, sv, sw = scales['u'], scales['v'], scales['w']

mu = F.softplus(model.viscosity)

pred_wss_x = mu * (u_g[:,1:2]*(su/sy) + v_g[:,0:1]*(sv/sx))
pred_wss_y = mu * (v_g[:,2:3]*(sv/sz) + w_g[:,1:2]*(sw/sy))
pred_wss_z = mu * (u_g[:,2:3]*(su/sz) + w_g[:,0:1]*(sw/sx))
pred_wss   = torch.cat([pred_wss_x, pred_wss_y, pred_wss_z], dim=1)  # (N_wall, 3)

true_wss_bnd = wss_gt[b_mask.cpu()].to(DEVICE)

wss_components = ['WSS_x', 'WSS_y', 'WSS_z']
wss_rel_l2  = [rel_l2( pred_wss[:,i], true_wss_bnd[:,i]) for i in range(3)]
wss_nmae    = [nmae_pct(pred_wss[:,i], true_wss_bnd[:,i]) for i in range(3)]
wss_rmse    = [rmse(   pred_wss[:,i], true_wss_bnd[:,i]) for i in range(3)]

pred_wss_mag = torch.norm(pred_wss, dim=1)
true_wss_mag = torch.norm(true_wss_bnd, dim=1)
wss_mag_rel_l2 = rel_l2(pred_wss_mag, true_wss_mag)
wss_mag_nmae   = nmae_pct(pred_wss_mag, true_wss_mag)
mean_wss_gt    = true_wss_mag.mean().item()
mean_wss_pred  = pred_wss_mag.mean().item()

for i, c in enumerate(wss_components):
    print(f"  {c:7s}  | Rel-L2 {wss_rel_l2[i]:.5f} | "
          f"NMAE {wss_nmae[i]:.3f}% | RMSE {wss_rmse[i]:.5f}")
print(f"  |WSS|    | Rel-L2 {wss_mag_rel_l2:.5f} | NMAE {wss_mag_nmae:.3f}%")
print(f"  Mean |WSS| GT: {mean_wss_gt:.6f}  |  Predicted: {mean_wss_pred:.6f}")


# ═══════════════════════════════════════════════════════════════════════════
#  6. PHYSICS RESIDUALS  (Navier-Stokes, computed via autograd)
#     Sample a fixed subset — computing for all N points is too slow
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n  ── Navier-Stokes residuals (sampled interior) ──────────")

PHYSICS_SAMPLE = min(5000, int((~b_mask).sum().item()))
interior_idx   = (~b_mask).nonzero(as_tuple=True)[0]
sample_idx     = interior_idx[torch.randperm(len(interior_idx))[:PHYSICS_SAMPLE]]

x_phys = X[sample_idx].clone().detach().requires_grad_(True)
ones_p  = torch.ones(PHYSICS_SAMPLE, 1, device=DEVICE)
pred_p  = model(x_phys)

mu_pos = F.softplus(model.viscosity)

# Reuse get_physics_loss but capture individual residuals.
#
# retain_graph=True explanation:
#   u_n, v_n, w_n, p_n all share the same forward graph from model(x_phys).
#   Each autograd.grad call would normally free that graph after running.
#   retain_graph=True keeps it alive for subsequent calls.
#   The final call (p_g) frees it with retain_graph=False (default).
#   create_graph=True builds a NEW graph over the gradient so we can
#   differentiate again for second derivatives.
u_n, v_n, w_n, p_n = pred_p[:,0:1], pred_p[:,1:2], pred_p[:,2:3], pred_p[:,3:4]
u_g = torch.autograd.grad(u_n, x_phys, grad_outputs=ones_p, create_graph=True, retain_graph=True)[0]
v_g = torch.autograd.grad(v_n, x_phys, grad_outputs=ones_p, create_graph=True, retain_graph=True)[0]
w_g = torch.autograd.grad(w_n, x_phys, grad_outputs=ones_p, create_graph=True, retain_graph=True)[0]
p_g = torch.autograd.grad(p_n, x_phys, grad_outputs=ones_p, create_graph=True, retain_graph=True)[0]

sx, sy, sz, st = scales['x'], scales['y'], scales['z'], scales['t']
su, sv, sw, sp = scales['u'], scales['v'], scales['w'], scales['p']

u_x = u_g[:,0:1]*(su/sx); u_y = u_g[:,1:2]*(su/sy)
u_z = u_g[:,2:3]*(su/sz); u_t = u_g[:,3:4]*(su/st)
v_x = v_g[:,0:1]*(sv/sx); v_y = v_g[:,1:2]*(sv/sy)
v_z = v_g[:,2:3]*(sv/sz); v_t = v_g[:,3:4]*(sv/st)
w_x = w_g[:,0:1]*(sw/sx); w_y = w_g[:,1:2]*(sw/sy)
w_z = w_g[:,2:3]*(sw/sz); w_t = w_g[:,3:4]*(sw/st)
p_x = p_g[:,0:1]*(sp/sx); p_y = p_g[:,1:2]*(sp/sy); p_z = p_g[:,2:3]*(sp/sz)

# Second derivatives.
# u_g, v_g, w_g each have their own graphs (built by create_graph=True above).
# We make 9 calls total: 3 on u_g's graph, 3 on v_g's, 3 on w_g's.
# retain_graph=True on the first two calls within each group keeps the graph
# alive; the third call (u_zz / v_zz / w_zz) frees it normally.
def dd(raw_g, dim, scale, retain=True):
    return torch.autograd.grad(
        raw_g, x_phys, grad_outputs=ones_p,
        create_graph=False, retain_graph=retain
    )[0][:, dim:dim+1] * scale

u_xx = dd(u_g[:,0:1], 0, su/sx**2, retain=True)
u_yy = dd(u_g[:,1:2], 1, su/sy**2, retain=True)
u_zz = dd(u_g[:,2:3], 2, su/sz**2, retain=True)   # MUST BE TRUE

v_xx = dd(v_g[:,0:1], 0, sv/sx**2, retain=True)
v_yy = dd(v_g[:,1:2], 1, sv/sy**2, retain=True)
v_zz = dd(v_g[:,2:3], 2, sv/sz**2, retain=True)   # MUST BE TRUE

w_xx = dd(w_g[:,0:1], 0, sw/sx**2, retain=True)
w_yy = dd(w_g[:,1:2], 1, sw/sy**2, retain=True)
w_zz = dd(w_g[:,2:3], 2, sw/sz**2, retain=False)  # <--- ONLY THIS ONE IS FALSE

u_r = (u_n + 1.0)*su + scales['min_u']
v_r = (v_n + 1.0)*sv + scales['min_v']
w_r = (w_n + 1.0)*sw + scales['min_w']

f_u = u_t + (u_r*u_x + v_r*u_y + w_r*u_z) + p_x - mu_pos*(u_xx+u_yy+u_zz)
f_v = v_t + (u_r*v_x + v_r*v_y + w_r*v_z) + p_y - mu_pos*(v_xx+v_yy+v_zz)
f_w = w_t + (u_r*w_x + v_r*w_y + w_r*w_z) + p_z - mu_pos*(w_xx+w_yy+w_zz)
f_c = u_x + v_y + w_z

mean_fu = torch.abs(f_u).mean().item()
mean_fv = torch.abs(f_v).mean().item()
mean_fw = torch.abs(f_w).mean().item()
mean_fc = torch.abs(f_c).mean().item()

print(f"  X-momentum |f_u|: {mean_fu:.6f}")
print(f"  Y-momentum |f_v|: {mean_fv:.6f}")
print(f"  Z-momentum |f_w|: {mean_fw:.6f}")
print(f"  Continuity |f_c|: {mean_fc:.6f}")
print(f"  (target: all close to 0.0)")

# Detach residual tensors for plotting
f_u_np = f_u.detach().cpu().numpy().ravel()
f_v_np = f_v.detach().cpu().numpy().ravel()
f_w_np = f_w.detach().cpu().numpy().ravel()
f_c_np = f_c.detach().cpu().numpy().ravel()


# ═══════════════════════════════════════════════════════════════════════════
#  7. BOUNDARY COMPLIANCE  (no-slip: velocity at wall = 0)
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n  ── Boundary compliance (no-slip) ───────────────────────")
with torch.no_grad():
    pred_bnd_nograds = batch_predict(X[b_mask])
    vel_at_wall = pred_bnd_nograds[:, 0:3]
    # In normalised space, 0 velocity corresponds to norm_vel.encode(0)
    zero_vel_norm = norm_vel.encode(torch.zeros(1, 3)).to(DEVICE)
    bc_violation = torch.norm(vel_at_wall - zero_vel_norm, dim=1).mean().item()
    # Physical velocity magnitude at wall
    vel_wall_phys = norm_vel.decode(vel_at_wall)
    mean_wall_vel = torch.norm(vel_wall_phys, dim=1).mean().item()

print(f"  Mean velocity magnitude at wall: {mean_wall_vel:.6f} m/s (should be ~0)")
print(f"  Normalised BC violation metric:  {bc_violation:.6f}")


# ═══════════════════════════════════════════════════════════════════════════
#  8. PER-TIMESTEP BREAKDOWN
#     coords_t[:,3] is the raw time value — group by it
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n  ── Per-timestep accuracy ───────────────────────────────")
unique_times = coords_t[:, 3].unique().sort()[0]
ts_vel_acc, ts_pres_acc = [], []

print(f"  {'Time':>8} | {'Vel Rel-L2':>10} | {'Vel Acc%':>9} | {'Pres Rel-L2':>11} | {'Pres Acc%':>9}")
print(f"  {'-'*58}")

with torch.no_grad():
    for t_val in unique_times:
        idx_t = (coords_t[:, 3] == t_val).nonzero(as_tuple=True)[0]
        X_t   = X[idx_t]
        p_t   = batch_predict(X_t)
        pv_t  = norm_vel.decode(p_t[:, 0:3])
        pp_t  = norm_pres.decode(p_t[:, 3:4])
        tv_t  = vel[idx_t].to(DEVICE)
        tp_t  = pres[idx_t].to(DEVICE)

        v_rl  = rel_l2(pv_t, tv_t)
        p_rl  = rel_l2(pp_t, tp_t)
        ts_vel_acc.append((1 - v_rl) * 100)
        ts_pres_acc.append((1 - p_rl) * 100)

        print(f"  {t_val.item():>8.3f} | {v_rl:>10.5f} | "
              f"{(1-v_rl)*100:>8.1f}% | {p_rl:>11.5f} | {(1-p_rl)*100:>8.1f}%")

print(f"\n  Mean vel acc across timesteps:  {np.mean(ts_vel_acc):.2f}%")
print(f"  Mean pres acc across timesteps: {np.mean(ts_pres_acc):.2f}%")
print(f"  Inference speed: {inference_ms_per_point:.3f} μs/point  "
      f"({1e6/inference_ms_per_point:.0f} points/sec)")


# ═══════════════════════════════════════════════════════════════════════════
#  9. PLOTTING
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}\n  Generating plots...\n{'='*60}")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 13, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white"
})

# ── Plot 1: Predicted vs ground truth scatter  ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("PINN predictions vs CFD ground truth", fontsize=14)

with torch.no_grad():
    pv_cpu  = pred_vel_phys.cpu().numpy()
    tv_cpu  = true_vel_phys.cpu().numpy()
    pp_cpu  = pred_pres_phys.cpu().numpy().ravel()
    tp_cpu  = true_pres_phys.cpu().numpy().ravel()

# Subsample for scatter (plotting all N points is slow)
idx_plot = np.random.choice(N, min(8000, N), replace=False)

for i, (comp, ax) in enumerate(zip(['u (x)', 'v (y)', 'w (z)'], axes)):
    ax.scatter(tv_cpu[idx_plot, i], pv_cpu[idx_plot, i],
               alpha=0.25, s=4, color="#7C5CBF")
    lo = min(tv_cpu[:, i].min(), pv_cpu[:, i].min())
    hi = max(tv_cpu[:, i].max(), pv_cpu[:, i].max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label="Perfect")
    ax.set_xlabel(f"CFD {comp} (m/s)")
    ax.set_ylabel(f"Predicted {comp} (m/s)")
    ax.set_title(f"Velocity {comp}\nRel-L2={vel_rel_l2[i]:.4f}  NMAE={vel_nmae[i]:.2f}%")
    ax.legend(fontsize=9)

plt.tight_layout()
path = os.path.join(PLOT_DIR, "velocity_scatter.png")
plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
print(f"  Saved: {path}")

# ── Plot 2: Pressure scatter  ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(tp_cpu[idx_plot], pp_cpu[idx_plot],
           alpha=0.25, s=4, color="#D85A30")
lo, hi = min(tp_cpu.min(), pp_cpu.min()), max(tp_cpu.max(), pp_cpu.max())
ax.plot([lo, hi], [lo, hi], 'b--', lw=1.5, label="Perfect")
ax.set_xlabel("CFD pressure (Pa)"); ax.set_ylabel("Predicted pressure (Pa)")
ax.set_title(f"Pressure prediction\nRel-L2={pres_rel_l2:.4f}  NMAE={pres_nmae:.2f}%")
ax.legend()
plt.tight_layout()
path = os.path.join(PLOT_DIR, "pressure_scatter.png")
plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
print(f"  Saved: {path}")

# ── Plot 3: WSS predicted vs ground truth  ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
twss_np  = true_wss_mag.detach().cpu().numpy()
pwss_np  = pred_wss_mag.detach().cpu().numpy()
idx_wss  = np.random.choice(len(twss_np), min(5000, len(twss_np)), replace=False)

axes[0].scatter(twss_np[idx_wss], pwss_np[idx_wss], alpha=0.3, s=5, color="#1D9E75")
lo, hi = min(twss_np.min(), pwss_np.min()), max(twss_np.max(), pwss_np.max())
axes[0].plot([lo, hi], [lo, hi], 'r--', lw=1.5, label="Perfect")
axes[0].set_xlabel("|WSS| CFD (Pa)"); axes[0].set_ylabel("|WSS| Predicted (Pa)")
axes[0].set_title(f"WSS magnitude\nRel-L2={wss_mag_rel_l2:.4f}  NMAE={wss_mag_nmae:.2f}%")
axes[0].legend()

bins = np.linspace(0, np.percentile(np.concatenate([twss_np, pwss_np]), 98), 40)
axes[1].hist(twss_np, bins=bins, alpha=0.6, color="#7C5CBF", label="CFD TAWSS", density=True)
axes[1].hist(pwss_np, bins=bins, alpha=0.6, color="#1D9E75", label="Predicted TAWSS", density=True)
axes[1].set_xlabel("|WSS| (Pa)"); axes[1].set_ylabel("Density")
axes[1].set_title("WSS magnitude distribution")
axes[1].legend()

plt.tight_layout()
path = os.path.join(PLOT_DIR, "wss_accuracy.png")
plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
print(f"  Saved: {path}")

# ── Plot 4: Navier-Stokes residual histograms  ────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
colors = ["#7C5CBF", "#1D9E75", "#D85A30", "#378ADD"]
for ax, data, label, color in zip(
    axes,
    [f_u_np, f_v_np, f_w_np, f_c_np],
    ["f_u (X-momentum)", "f_v (Y-momentum)", "f_w (Z-momentum)", "f_c (Continuity)"],
    colors
):
    ax.hist(data, bins=60, color=color, alpha=0.8, density=True)
    ax.axvline(0, color='black', linestyle='--', lw=1.5, label="Ideal (0)")
    ax.set_xlabel("Residual value"); ax.set_ylabel("Density")
    ax.set_title(f"{label}\nmean|res|={np.abs(data).mean():.4f}")
    ax.legend(fontsize=9)

plt.suptitle("Navier-Stokes residuals — PINN interior points\n"
             "(Perfect model → all residuals = 0)", fontsize=13, y=1.02)
plt.tight_layout()
path = os.path.join(PLOT_DIR, "ns_residuals.png")
plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
print(f"  Saved: {path}")

# ── Plot 5: Per-timestep accuracy  ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
t_vals = [t.item() for t in unique_times]
ax.plot(t_vals, ts_vel_acc,  color="#7C5CBF", lw=2.5, marker='o', label="Velocity accuracy%")
ax.plot(t_vals, ts_pres_acc, color="#D85A30", lw=2.5, marker='s', label="Pressure accuracy%")
ax.axhline(70, color='gray', linestyle='--', lw=1.5, label="70% threshold")
ax.set_xlabel("Cardiac cycle time (s)"); ax.set_ylabel("Accuracy (1 − Rel-L2) × 100%")
ax.set_title("PINN accuracy across cardiac cycle timesteps")
ax.set_ylim(0, 105); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
path = os.path.join(PLOT_DIR, "per_timestep_accuracy.png")
plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
print(f"  Saved: {path}")

# ── Plot 6: Ablation summary table figure  ───────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
ax.axis("off")
rows = [
    ["Metric",                    "Value",                "Interpretation"],
    ["Vel magnitude Rel-L2 ↓",    f"{vmag_rel_l2:.5f}",   "lower = better"],
    ["Vel magnitude NMAE ↓",      f"{vmag_nmae:.3f}%",    "lower = better"],
    ["Pressure Rel-L2 ↓",         f"{pres_rel_l2:.5f}",   "lower = better"],
    ["Pressure NMAE ↓",           f"{pres_nmae:.3f}%",    "lower = better"],
    ["|WSS| Rel-L2 ↓",            f"{wss_mag_rel_l2:.5f}","lower = better"],
    ["|WSS| NMAE ↓",              f"{wss_mag_nmae:.3f}%", "lower = better"],
    ["NS continuity |f_c| ↓",     f"{mean_fc:.6f}",       "→ 0 = incompressible"],
    ["NS momentum |f_u| ↓",       f"{mean_fu:.6f}",       "→ 0 = NS satisfied"],
    ["Learned viscosity (Pa·s)",  f"{learned_visc:.6f}",  f"true = {TRUE_VISCOSITY:.4f}"],
    ["Viscosity error",           f"{abs(learned_visc-TRUE_VISCOSITY)/TRUE_VISCOSITY*100:.2f}%", "0% = perfect"],
    ["BC violation (wall vel)",   f"{mean_wall_vel:.6f} m/s", "→ 0 = no-slip satisfied"],
    ["Inference speed",           f"{inference_ms_per_point:.3f} μs/pt", "vs seconds for CFD"],
    ["Parameters",                f"{n_params:,}",        "compact PINN"],
]
table = ax.table(cellText=rows[1:], colLabels=rows[0],
                 cellLoc='center', loc='center', colWidths=[0.38, 0.22, 0.40])
table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1.0, 2.0)
for j in range(3):
    table[0, j].set_facecolor("#3C3489")
    table[0, j].set_text_props(color="white", fontweight="bold")
for i in range(1, len(rows)):
    bg = "#F7F7FF" if i % 2 == 0 else "white"
    for j in range(3): table[i, j].set_facecolor(bg)
plt.title("PINN Blood-Flow Surrogate — Summary Metrics",
          fontsize=14, fontweight="bold", pad=20)
path = os.path.join(PLOT_DIR, "summary_table.png")
plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  10. SAVE JSON + CSV
# ═══════════════════════════════════════════════════════════════════════════
report = {
    "model_path": MODEL_PATH,
    "n_params": n_params,
    "learned_viscosity_Pa_s": learned_visc,
    "true_viscosity_Pa_s": TRUE_VISCOSITY,
    "viscosity_error_pct": abs(learned_visc - TRUE_VISCOSITY) / TRUE_VISCOSITY * 100,
    "velocity": {
        "u_rel_l2": vel_rel_l2[0], "v_rel_l2": vel_rel_l2[1], "w_rel_l2": vel_rel_l2[2],
        "u_nmae_pct": vel_nmae[0], "v_nmae_pct": vel_nmae[1], "w_nmae_pct": vel_nmae[2],
        "u_rmse": vel_rmse[0],     "v_rmse": vel_rmse[1],     "w_rmse": vel_rmse[2],
        "magnitude_rel_l2": vmag_rel_l2, "magnitude_nmae_pct": vmag_nmae,
    },
    "pressure": {
        "rel_l2": pres_rel_l2, "nmae_pct": pres_nmae, "rmse": pres_rmse,
    },
    "wss": {
        "x_rel_l2": wss_rel_l2[0], "y_rel_l2": wss_rel_l2[1], "z_rel_l2": wss_rel_l2[2],
        "magnitude_rel_l2": wss_mag_rel_l2, "magnitude_nmae_pct": wss_mag_nmae,
        "mean_gt_Pa": mean_wss_gt, "mean_pred_Pa": mean_wss_pred,
    },
    "ns_residuals": {
        "mean_abs_f_u": mean_fu, "mean_abs_f_v": mean_fv,
        "mean_abs_f_w": mean_fw, "mean_abs_f_c": mean_fc,
        "sample_size": PHYSICS_SAMPLE,
    },
    "boundary": {
        "mean_wall_velocity_m_s": mean_wall_vel,
        "bc_violation_normalised": bc_violation,
    },
    "per_timestep": {
        "times": t_vals,
        "vel_acc_pct": ts_vel_acc,
        "pres_acc_pct": ts_pres_acc,
    },
    "inference_us_per_point": inference_ms_per_point,
}

json_path = os.path.join(RESULTS_DIR, "pinn_metrics_report.json")
with open(json_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"\n  JSON report: {json_path}")

import csv
csv_path = os.path.join(RESULTS_DIR, "pinn_metrics_table.csv")
rows_csv = [
    ["Metric", "Value"],
    ["Vel u Rel-L2",          f"{vel_rel_l2[0]:.5f}"],
    ["Vel v Rel-L2",          f"{vel_rel_l2[1]:.5f}"],
    ["Vel w Rel-L2",          f"{vel_rel_l2[2]:.5f}"],
    ["Vel |u| Rel-L2",        f"{vmag_rel_l2:.5f}"],
    ["Vel |u| NMAE%",         f"{vmag_nmae:.3f}"],
    ["Pressure Rel-L2",       f"{pres_rel_l2:.5f}"],
    ["Pressure NMAE%",        f"{pres_nmae:.3f}"],
    ["|WSS| Rel-L2",          f"{wss_mag_rel_l2:.5f}"],
    ["|WSS| NMAE%",           f"{wss_mag_nmae:.3f}"],
    ["NS |f_u|",              f"{mean_fu:.6f}"],
    ["NS |f_v|",              f"{mean_fv:.6f}"],
    ["NS |f_w|",              f"{mean_fw:.6f}"],
    ["NS |f_c|",              f"{mean_fc:.6f}"],
    ["Learned viscosity Pa.s",f"{learned_visc:.6f}"],
    ["Viscosity error %",     f"{abs(learned_visc-TRUE_VISCOSITY)/TRUE_VISCOSITY*100:.2f}"],
    ["BC violation m/s",      f"{mean_wall_vel:.6f}"],
    ["Inference us/pt",       f"{inference_ms_per_point:.3f}"],
]
with open(csv_path, "w", newline="") as f:
    csv.writer(f).writerows(rows_csv)
print(f"  CSV table:   {csv_path}")

print(f"\n{'='*60}")
print(f"  All plots saved to: {PLOT_DIR}")
print(f"{'='*60}\n")