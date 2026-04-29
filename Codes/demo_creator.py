"""
demo_creator.py — Generates all demo assets for IIT external review.

Creates the following in ../Demo/:
  comparison_t{N}.png     — CFD vs model vs error for pressure + velocity
                            at 3 timepoints (systole, diastole, mid)
  rollout_animation.gif   — 50-step autoregressive rollout of vel-mag and pressure
  pressure_drop_overlay.png — ΔP over cardiac cycle: model vs CFD
  error_accumulation.png  — per-step Rel-L2 with 70% stability threshold marked
  wss_3panel.png          — WSS distribution: histogram + 2D map + TAWSS vs CFD

Usage
  python demo_creator.py

All outputs are high-res (≥150 DPI) PNGs plus one GIF.
The GIF is the most impactful demo visual — show it on loop during the presentation.
"""

import gc
import os
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm, LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import imageio.v2 as imageio
from io import BytesIO

from model import HUFNO3d
from fno_data_loader import FNODataLoader

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
#  PATHS & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
DATA_PATH   = "../VelocityData3D"
WALL_PATH   = "../VelocityData3D/WallMesh/wall.vtp"
DEMO_DIR    = "../Demo"
os.makedirs(DEMO_DIR, exist_ok=True)

BEST_MODEL  = "../Models/fno_best.pth"
FALLBACK    = "../Models/fno_checkpoint.pth"
MODEL_PATH  = BEST_MODEL if os.path.exists(BEST_MODEL) else FALLBACK

MU_BLOOD    = 0.004   # Pa·s
DPI_HIGH    = 180     # for presentation-quality PNGs
DPI_ANIM    = 120     # for GIF frames (balance quality vs file size)

# Matplotlib style — clean, professional
plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.facecolor":"white",
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ═══════════════════════════════════════════════════════════════════════════
#  LOAD MODEL + DATA
# ═══════════════════════════════════════════════════════════════════════════
print(f"Loading model from: {MODEL_PATH}")
ckpt        = torch.load(MODEL_PATH, map_location=device, weights_only=False)
mask        = ckpt["mask"].to(device)
grid_coords = ckpt["grid_coords"].to(device)
stats       = ckpt["stats"]
res         = mask.shape[0]
print(f"Resolution: {res}³")

model = HUFNO3d(
    modes=8, width=32, in_channels=9, out_channels=5, num_layers=4
).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")

print("Loading data...")
loader = FNODataLoader(DATA_PATH, wall_file_path=WALL_PATH, resolution=res)
fields, mask_cpu, grid_coords_cpu, _ = loader.load()
del loader; gc.collect()

T = fields.shape[0]
print(f"Timesteps: {T}")

# Derived
mask_dev  = mask.unsqueeze(0).unsqueeze(0)           # (1,1,R,R,R)
_not_mask = (~mask.bool()).float().unsqueeze(0).unsqueeze(0)
_dilated  = F.max_pool3d(_not_mask, kernel_size=3, stride=1, padding=1)
wall_mask = (mask_dev.float() * _dilated).bool()     # (1,1,R,R,R)
dx        = 1.0 / (res - 1)


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def denorm(field_std, channel):
    if channel >= 4: return field_std
    return field_std * stats[f"std_{channel}"] + stats[f"mean_{channel}"]


def build_inp(field_t):
    return torch.cat([field_t, mask.unsqueeze(0), grid_coords], dim=0).unsqueeze(0)


def vel_mag(pred_or_field):
    """Velocity magnitude in standardised space."""
    return (pred_or_field[:3] ** 2).sum(0).sqrt().cpu().numpy()


def vel_mag_phys(pred_or_field):
    """Velocity magnitude in physical (denormalised) units."""
    u = denorm(pred_or_field[0], 0).cpu().numpy()
    v = denorm(pred_or_field[1], 1).cpu().numpy()
    w = denorm(pred_or_field[2], 2).cpu().numpy()
    return np.sqrt(u**2 + v**2 + w**2)


def pressure_phys(pred_or_field):
    """Pressure in physical (denormalised) units."""
    return denorm(pred_or_field[3], 3).cpu().numpy()


def mask_np():
    return mask.cpu().numpy().astype(bool)


def mid_slice(arr3d, axis=2):
    """Return the mid-plane slice along given axis."""
    s = arr3d.shape[axis] // 2
    return np.take(arr3d, s, axis=axis)


def add_colorbar(ax, im, label="", size="4%", pad=0.05):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    plt.colorbar(im, cax=cax, label=label)


def imshow_slice(ax, data2d, mask2d, cmap, title, unit="",
                 vmin=None, vmax=None, symmetric=False):
    """Show a masked 2D slice with proper colormap and colorbar."""
    d = np.where(mask2d, data2d, np.nan)
    if symmetric:
        absmax = np.nanmax(np.abs(d)) + 1e-8
        vmin, vmax = -absmax, absmax
        cmap = "RdBu_r"
    im = ax.imshow(d.T, cmap=cmap, origin="lower", aspect="equal",
                   vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    add_colorbar(ax, im, unit)
    ax.tick_params(labelsize=8)
    return im


# ═══════════════════════════════════════════════════════════════════════════
#  1. FULL AUTOREGRESSIVE ROLLOUT (cache everything)
# ═══════════════════════════════════════════════════════════════════════════
print("\nRunning full autoregressive rollout...")

preds       = []    # list of (5,R,R,R) tensors on CPU
rollout_rel = []
rollout_vel = []
rollout_pres= []
pres_drops  = []

mask_np_arr  = mask_np()
mask_cpu_dev = mask_dev.cpu()

z_has_mask = mask_np_arr.any(axis=(0, 1))
z_idxs     = np.where(z_has_mask)[0]
z_in       = int(z_idxs[0])
z_out      = int(z_idxs[-1])

current = fields[0].unsqueeze(0).to(device)
t0 = time.perf_counter()

with torch.no_grad():
    for s in range(T - 1):
        inp    = torch.cat([current[0], mask.unsqueeze(0), grid_coords], dim=0).unsqueeze(0)
        pred   = model(inp)
        current = pred

        pred_cpu = pred[0].cpu()
        preds.append(pred_cpu)

        tgt    = fields[s + 1]
        mask_c = mask_cpu_dev[0]

        # rel L2
        d_all  = (pred_cpu - tgt) * mask_c
        t_all  = tgt * mask_c
        rl = torch.sqrt((d_all**2).sum() / ((t_all**2).sum() + 1e-8)).item()

        d_vel  = (pred_cpu[:3] - tgt[:3]) * mask_c[:3]
        t_vel  = tgt[:3] * mask_c[:3]
        rv = torch.sqrt((d_vel**2).sum() / ((t_vel**2).sum() + 1e-8)).item()

        d_pres = (pred_cpu[3:4] - tgt[3:4]) * mask_c[0:1]
        t_pres = tgt[3:4] * mask_c[0:1]
        rp = torch.sqrt((d_pres**2).sum() / ((t_pres**2).sum() + 1e-8)).item()

        rollout_rel.append(rl)
        rollout_vel.append(rv)
        rollout_pres.append(rp)

        # pressure drop
        p_field = pred_cpu[3:4]
        m1d     = mask_cpu_dev[0, 0]
        pin  = (p_field[0,:,:,z_in]  * m1d[:,:,z_in]).sum()  / m1d[:,:,z_in].sum().clamp(min=1)
        pout = (p_field[0,:,:,z_out] * m1d[:,:,z_out]).sum() / m1d[:,:,z_out].sum().clamp(min=1)
        pres_drops.append(
            (denorm(pin, 3) - denorm(pout, 3)).item()
        )

inference_total = time.perf_counter() - t0
print(f"  Rollout complete — {inference_total*1000:.0f} ms total, "
      f"{inference_total/(T-1)*1000:.2f} ms/step")

# CFD ground-truth pressure drops
gt_pres_drops = []
for s in range(T - 1):
    tgt = fields[s + 1]
    m1d = mask_cpu.numpy().astype(bool)
    pin  = tgt[3][m1d[:, :, z_in]].mean() if m1d[:,:,z_in].any() else 0
    pout = tgt[3][m1d[:, :, z_out]].mean() if m1d[:,:,z_out].any() else 0
    gt_pres_drops.append(
        (denorm(torch.tensor(pin), 3) - denorm(torch.tensor(pout), 3)).item()
    )


# ═══════════════════════════════════════════════════════════════════════════
#  2. COMPARISON PANELS  —  CFD | Model | Error  ×  3 timepoints
# ═══════════════════════════════════════════════════════════════════════════
print("\nGenerating comparison panels...")

# Systole ~ peak velocity (find step with max mean vel inside mask)
vel_mags_mean = [
    float((fields[s+1, :3] ** 2).sum(0).sqrt().numpy()[mask_np_arr].mean())
    for s in range(T - 1)
]
step_systole  = int(np.argmax(vel_mags_mean))
step_diastole = int(np.argmin(vel_mags_mean))
step_mid      = (T - 1) // 2

timepoints = [
    (step_systole,  "systole (peak flow)"),
    (step_mid,      "mid-cycle"),
    (step_diastole, "diastole (low flow)"),
]

for step_idx, step_label in timepoints:
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"CFD ground truth  vs  HUFNO3d prediction  vs  absolute error\n"
                 f"Timestep {step_idx+1} — {step_label}", fontsize=14, y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    pred_f = preds[step_idx]                  # (5,R,R,R)
    gt_f   = fields[step_idx + 1]             # (5,R,R,R)

    m2d = mid_slice(mask_np_arr, axis=2)      # (R,R) XY mid-plane

    # ── Row 0: velocity magnitude ──────────────────────────────────────────
    vm_gt   = mid_slice(vel_mag(gt_f),   2)
    vm_pred = mid_slice(vel_mag(pred_f), 2)
    vm_err  = np.abs(vm_pred - vm_gt)
    vm_vmax = np.nanmax(np.where(m2d, vm_gt, np.nan)) + 1e-8

    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax02 = fig.add_subplot(gs[0, 2])

    imshow_slice(ax00, vm_gt,   m2d, "plasma",  "CFD  — velocity magnitude", "|u| (std)", 0, vm_vmax)
    imshow_slice(ax01, vm_pred, m2d, "plasma",  "HUFNO3d — velocity magnitude", "|u| (std)", 0, vm_vmax)
    im_err = imshow_slice(ax02, vm_err, m2d, "hot", "Absolute error", "|Δu|", 0)

    # ── Row 1: pressure ────────────────────────────────────────────────────
    p_gt_ph   = mid_slice(pressure_phys(gt_f),   2)
    p_pred_ph = mid_slice(pressure_phys(pred_f), 2)
    p_err     = np.abs(p_pred_ph - p_gt_ph)

    gt_vals  = p_gt_ph[m2d]
    p_vmin   = np.nanmin(gt_vals) if len(gt_vals) else -1
    p_vmax   = np.nanmax(gt_vals) if len(gt_vals) else 1

    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    ax12 = fig.add_subplot(gs[1, 2])

    imshow_slice(ax10, p_gt_ph,   m2d, "RdBu_r", "CFD  — pressure", "p (Pa)", p_vmin, p_vmax)
    imshow_slice(ax11, p_pred_ph, m2d, "RdBu_r", "HUFNO3d — pressure", "p (Pa)", p_vmin, p_vmax)
    imshow_slice(ax12, p_err,     m2d, "hot",     "Absolute error", "|Δp| (Pa)")

    path = os.path.join(DEMO_DIR, f"comparison_t{step_idx+1:02d}_{step_label.replace(' ','_').replace('(','').replace(')','')}.png")
    plt.savefig(path, dpi=DPI_HIGH, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  3. ROLLOUT ANIMATION GIF
# ═══════════════════════════════════════════════════════════════════════════
print("\nGenerating rollout animation GIF...")

frames = []
m2d    = mid_slice(mask_np_arr, axis=2)

# Compute global colour limits from all predicted frames
all_vm  = [mid_slice(vel_mag(p), 2)      for p in preds]
all_p   = [mid_slice(pressure_phys(p), 2) for p in preds]
vm_max  = max(np.nanmax(np.where(m2d, v, np.nan)) for v in all_vm) + 1e-8
p_vals  = np.concatenate([np.where(m2d, pp, np.nan).ravel() for pp in all_p])
p_vmin  = np.nanpercentile(p_vals, 2)
p_vmax  = np.nanpercentile(p_vals, 98)

for s, pred_f in enumerate(preds):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("white")

    step_frac = (s + 1) / (T - 1)
    cardiac_label = (
        "systole (peak)"   if step_frac < 0.25 else
        "early diastole"   if step_frac < 0.50 else
        "late diastole"    if step_frac < 0.75 else
        "pre-systole"
    )
    fig.suptitle(
        f"HUFNO3d autoregressive prediction — step {s+1}/{T-1}  ({cardiac_label})\n"
        f"Inference: {inference_total/(T-1)*1000:.1f} ms/step  |  "
        f"Rel-L2: {rollout_rel[s]:.3f}  |  Acc: {(1-rollout_rel[s])*100:.1f}%",
        fontsize=11, y=0.99
    )

    # Velocity magnitude
    vm = np.where(m2d, all_vm[s], np.nan)
    im0 = axes[0].imshow(vm.T, cmap="plasma", origin="lower", aspect="equal",
                          vmin=0, vmax=vm_max)
    axes[0].set_title("Velocity magnitude (standardised)")
    axes[0].set_xlabel("X"); axes[0].set_ylabel("Y")
    add_colorbar(axes[0], im0, "|u|")
    axes[0].tick_params(labelsize=8)

    # Pressure
    pp = np.where(m2d, all_p[s], np.nan)
    im1 = axes[1].imshow(pp.T, cmap="RdBu_r", origin="lower", aspect="equal",
                          vmin=p_vmin, vmax=p_vmax)
    axes[1].set_title("Pressure (physical units, Pa)")
    axes[1].set_xlabel("X"); axes[1].set_ylabel("Y")
    add_colorbar(axes[1], im1, "p (Pa)")
    axes[1].tick_params(labelsize=8)

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=DPI_ANIM, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    frames.append(imageio.imread(buf))
    buf.close()

    if (s + 1) % 10 == 0:
        print(f"  Frame {s+1}/{T-1} done")

gif_path = os.path.join(DEMO_DIR, "rollout_animation.gif")
imageio.mimsave(gif_path, frames, fps=4, loop=0)
print(f"  Saved: {gif_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  4. PRESSURE DROP OVERLAY  — model vs CFD ground truth
# ═══════════════════════════════════════════════════════════════════════════
print("\nGenerating pressure drop overlay...")

steps = list(range(1, T))
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(steps, gt_pres_drops, color="#7C5CBF", lw=2.5, label="CFD ground truth")
ax.plot(steps, pres_drops,    color="#1D9E75", lw=2.5, linestyle="--",
        label="HUFNO3d prediction")
ax.fill_between(steps, gt_pres_drops, pres_drops, alpha=0.15, color="#D85A30",
                label="Error band")

ax.set_xlabel("Cardiac cycle step", fontsize=12)
ax.set_ylabel("Pressure drop ΔP (Pa)", fontsize=12)
ax.set_title("Pressure drop (inlet → outlet) over the cardiac cycle\n"
             "HUFNO3d vs CFD ground truth", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Annotate correlation
corr = np.corrcoef(gt_pres_drops, pres_drops)[0, 1]
ax.text(0.97, 0.05, f"Pearson r = {corr:.4f}",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=11, bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))

plt.tight_layout()
path = os.path.join(DEMO_DIR, "pressure_drop_overlay.png")
plt.savefig(path, dpi=DPI_HIGH, bbox_inches="tight")
plt.close()
print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  5. ERROR ACCUMULATION CURVE  (the key Figure 1 for any paper)
# ═══════════════════════════════════════════════════════════════════════════
print("\nGenerating error accumulation curve...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

acc_list  = [(1 - e) * 100 for e in rollout_rel]
acc_vel   = [(1 - e) * 100 for e in rollout_vel]
acc_pres  = [(1 - e) * 100 for e in rollout_pres]

# Panel A: accuracy %
axes[0].plot(steps, acc_list,  color="#7C5CBF", lw=2.5, label="Overall")
axes[0].plot(steps, acc_vel,   color="#1D9E75", lw=2,   linestyle="--", label="Velocity")
axes[0].plot(steps, acc_pres,  color="#D85A30", lw=2,   linestyle=":",  label="Pressure")
axes[0].axhline(70, color="gray", linestyle="--", lw=1.5, label="70% stability threshold")
axes[0].fill_between(steps, 70, acc_list, where=[a > 70 for a in acc_list],
                     alpha=0.12, color="#1D9E75", label="Stable region")
axes[0].set_xlabel("Rollout step (cardiac phase)", fontsize=12)
axes[0].set_ylabel("Accuracy  (1 − Rel-L2) × 100%", fontsize=12)
axes[0].set_title("Rollout accuracy over the cardiac cycle", fontsize=13)
axes[0].set_ylim(0, 105)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

stable_steps = sum(1 for a in acc_list if a > 70)
axes[0].text(0.02, 0.05,
    f"Stable steps: {stable_steps}/{T-1}\nMean acc: {np.mean(acc_list):.1f}%",
    transform=axes[0].transAxes, fontsize=10,
    bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))

# Panel B: Rel-L2 (linear scale)
axes[1].plot(steps, rollout_rel,   color="#7C5CBF", lw=2.5, label="Overall")
axes[1].plot(steps, rollout_vel,   color="#1D9E75", lw=2, linestyle="--", label="Velocity")
axes[1].plot(steps, rollout_pres,  color="#D85A30", lw=2, linestyle=":",  label="Pressure")
axes[1].axhline(0.30, color="gray", linestyle="--", lw=1.5, label="30% error threshold")
axes[1].set_xlabel("Rollout step (cardiac phase)", fontsize=12)
axes[1].set_ylabel("Relative L2 error", fontsize=12)
axes[1].set_title("Rollout error accumulation", fontsize=13)
axes[1].set_ylim(bottom=0)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
path = os.path.join(DEMO_DIR, "error_accumulation.png")
plt.savefig(path, dpi=DPI_HIGH, bbox_inches="tight")
plt.close()
print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  6. WSS 3-PANEL  — histogram + 2D map + TAWSS vs CFD TAWSS
# ═══════════════════════════════════════════════════════════════════════════
print("\nGenerating WSS figure...")

# Compute WSS for all prediction timesteps
wall_np  = wall_mask[0, 0].cpu().numpy().astype(bool)
m2d_wall = mid_slice(wall_np, axis=2)
wss_all  = []

for pred_f in preds:
    u_p = denorm(pred_f[0:1], 0).numpy()[0]
    v_p = denorm(pred_f[1:2], 1).numpy()[0]
    w_p = denorm(pred_f[2:3], 2).numpy()[0]
    grad_u = np.sqrt(
        np.gradient(u_p, dx, axis=0)**2 +
        np.gradient(v_p, dx, axis=1)**2 +
        np.gradient(w_p, dx, axis=2)**2
    )
    wss_vol = MU_BLOOD * grad_u * wall_np
    wss_all.append(wss_vol)

wss_stack = np.stack(wss_all, axis=0)           # (T-1, R, R, R)
tawss     = wss_stack.mean(axis=0)              # (R, R, R)

# OSI
mean_vec  = wss_stack.mean(axis=0)
mean_abs  = np.abs(mean_vec)
mean_mag  = np.abs(wss_stack).mean(axis=0) + 1e-12
osi_map   = 0.5 * (1.0 - mean_abs / mean_mag) * wall_np

# CFD ground-truth WSS (from the loaded fields)
wss_gt_all = []
for s in range(T - 1):
    f = fields[s + 1]
    u_g = denorm(f[0:1], 0).numpy()[0]
    v_g = denorm(f[1:2], 1).numpy()[0]
    w_g = denorm(f[2:3], 2).numpy()[0]
    grad_u_g = np.sqrt(
        np.gradient(u_g, dx, axis=0)**2 +
        np.gradient(v_g, dx, axis=1)**2 +
        np.gradient(w_g, dx, axis=2)**2
    )
    wss_gt_all.append(MU_BLOOD * grad_u_g * wall_np)
tawss_gt  = np.stack(wss_gt_all, axis=0).mean(axis=0)

fig = plt.figure(figsize=(18, 5))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)

# Panel A: WSS histogram — model vs CFD
ax0 = fig.add_subplot(gs[0])
wss_model_vals = tawss[wall_np].ravel()
wss_gt_vals    = tawss_gt[wall_np].ravel()
bins = np.linspace(0, np.percentile(
    np.concatenate([wss_model_vals, wss_gt_vals]), 98), 40)
ax0.hist(wss_gt_vals,    bins=bins, alpha=0.6, color="#7C5CBF", label="CFD (TAWSS)",    density=True)
ax0.hist(wss_model_vals, bins=bins, alpha=0.6, color="#1D9E75", label="Model (TAWSS)", density=True)
ax0.set_xlabel("TAWSS (Pa)", fontsize=11)
ax0.set_ylabel("Density", fontsize=11)
ax0.set_title("TAWSS distribution\nmodel vs CFD ground truth", fontsize=12)
ax0.legend(fontsize=10)
ax0.grid(True, alpha=0.3)

# Panel B: TAWSS map (mid-slice)
ax1 = fig.add_subplot(gs[1])
tawss_2d = np.where(m2d_wall, mid_slice(tawss, axis=2), np.nan)
im1 = ax1.imshow(tawss_2d.T, cmap="hot", origin="lower", aspect="equal")
ax1.set_title("TAWSS map — mid XY slice\n(bright = high shear stress)", fontsize=12)
ax1.set_xlabel("X"); ax1.set_ylabel("Y")
add_colorbar(ax1, im1, "TAWSS (Pa)")
ax1.tick_params(labelsize=8)

# Panel C: OSI map (mid-slice)
ax2 = fig.add_subplot(gs[2])
osi_2d = np.where(m2d_wall, mid_slice(osi_map, axis=2), np.nan)
im2 = ax2.imshow(osi_2d.T, cmap="RdYlBu_r", origin="lower", aspect="equal",
                  vmin=0, vmax=0.5)
ax2.set_title("OSI map — mid XY slice\n(high OSI = atherosclerosis risk zone)", fontsize=12)
ax2.set_xlabel("X"); ax2.set_ylabel("Y")
add_colorbar(ax2, im2, "OSI (0 → 0.5)")
ax2.tick_params(labelsize=8)

plt.suptitle("Wall shear stress analysis  |  HUFNO3d blood-flow surrogate",
             fontsize=13, y=1.02, fontweight="bold")
path = os.path.join(DEMO_DIR, "wss_3panel.png")
plt.savefig(path, dpi=DPI_HIGH, bbox_inches="tight")
plt.close()
print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  7. SUMMARY METRICS CARD  (one-slide summary for the panel)
# ═══════════════════════════════════════════════════════════════════════════
print("\nGenerating summary metrics card...")

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis("off")

rows = [
    ["Metric",                     "Value",          "Interpretation"],
    ["One-step vel Rel-L2",        f"{np.mean(rollout_vel):.4f}",
     "↓ lower is better"],
    ["One-step pres Rel-L2",       f"{np.mean(rollout_pres):.4f}",
     "↓ lower is better"],
    ["Mean accuracy (all steps)",  f"{np.mean(acc_list):.1f}%",
     "↑ higher is better"],
    ["Stable steps (>70%)",        f"{stable_steps}/{T-1}",
     "more = better rollout stability"],
    ["Mean TAWSS",                 f"{np.mean(wss_stack[:,wall_np]):.5f} Pa",
     "haemodynamic WSS"],
    ["Inference per step",         f"{inference_total/(T-1)*1000:.2f} ms",
     "vs hours for CFD"],
    ["Parameters",                 f"{n_params:,}",
     "8.7M — efficient architecture"],
    ["Pressure drop corr (r)",     f"{np.corrcoef(gt_pres_drops, pres_drops)[0,1]:.4f}",
     "1.0 = perfect agreement with CFD"],
]

table = ax.table(cellText=rows[1:], colLabels=rows[0],
                 cellLoc="center", loc="center", colWidths=[0.38, 0.22, 0.40])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.0, 2.2)

for j in range(3):
    table[0, j].set_facecolor("#3C3489")
    table[0, j].set_text_props(color="white", fontweight="bold")

for i in range(1, len(rows)):
    bg = "#F7F7FF" if i % 2 == 0 else "white"
    for j in range(3):
        table[i, j].set_facecolor(bg)

plt.title("HUFNO3d — Blood Flow Surrogate Model\nSummary Metrics",
          fontsize=14, fontweight="bold", pad=20)
path = os.path.join(DEMO_DIR, "summary_metrics_card.png")
plt.savefig(path, dpi=DPI_HIGH, bbox_inches="tight")
plt.close()
print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  DONE
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print("  All demo assets saved to:", DEMO_DIR)
print(f"{'='*55}")
print(f"  comparison_t*.png        — CFD vs model vs error (3 timepoints)")
print(f"  rollout_animation.gif    — cardiac cycle animation (show on loop)")
print(f"  pressure_drop_overlay.png — ΔP: model vs CFD")
print(f"  error_accumulation.png   — Figure 1 for your presentation")
print(f"  wss_3panel.png           — TAWSS, OSI maps + histogram")
print(f"  summary_metrics_card.png — one slide with all key numbers")
print(f"\n  Inference speed: {inference_total*1000:.0f} ms for {T-1} steps")