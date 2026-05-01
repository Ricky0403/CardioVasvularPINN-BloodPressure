"""
Training script — Hierarchical U-Net Fourier Neural Operator (HUFNO3d).

Goal: predict 3-D blood velocity + pressure fields over a cardiac cycle.

Physics losses added for pressure prediction
────────────────────────────────────────────
1. fd_physics_loss        — continuity: ∇·u = 0 (incompressible blood)
2. pressure_poisson_loss  — simplified Pressure Poisson Equation:
                            ∇²p ≈ -(du/dx)² - (dv/dy)² - (dw/dz)²
                            (diagonal terms only; cross-derivatives add noise)
3. momentum_residual_loss — Euler momentum: ∇p ≈ -(u_next - u_prev)/Δt
                            links pressure gradient to velocity time derivative
4. bc_loss                — no-slip: u=v=w=0 at vessel wall voxels
5. pressure_stability_loss— smoothness: penalise |∇p|² to prevent pressure spikes
6. anchor_loss            — tethers optimizer_phys to real CFD data (prevents
                            trivial zero-velocity solution)

Pressure channel (ch 3) gets 3× weight in the data loss because it is the
primary clinical quantity of interest (wall shear stress, haemodynamic loading).

All fixes from previous iterations are included:
  • retain_graph=False (phys does its own fresh forward)
  • loss accumulators initialised as tensors, not floats
  • step_weights length tied to ROLLOUT_STEPS
  • explicit LR_DATA / LR_PHYS
  • CosineAnnealingWarmRestarts (escapes plateau via periodic LR kicks)
  • anchor weight = 0.1  (was 1.0 — physics was being drowned out)
  • stronger data augmentation (50% temporal reversal + scale jitter)
  • gc imported at top, inline max_pool3d import removed
  • arch-mismatch caught gracefully with clear message
"""

import gc
import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import HUFNO3d
from fno_data_loader import FNODataLoader

torch.set_float32_matmul_precision("high")


# ═══════════════════════════════════════════════════════════════════════════
#  PHYSICS LOSSES
# ═══════════════════════════════════════════════════════════════════════════

def fd_physics_loss(pred, mask_dev, dx=1.0):
    """
    Continuity equation: ∇·u = 0  (incompressible blood).

    Uses one-sided finite differences on interior vessel voxels only
    (mask_int = product of neighbouring masks, so only fully-interior
    voxels contribute — wall voxels and their immediate neighbours are
    excluded to avoid artefacts from the sharp mask boundary).
    """
    u, v, w = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

    du_dx = (u[:, :, 1:, :, :] - u[:, :, :-1, :, :]) / dx
    dv_dy = (v[:, :, :, 1:, :] - v[:, :, :, :-1, :]) / dx
    dw_dz = (w[:, :, :, :, 1:] - w[:, :, :, :, :-1]) / dx

    mask_x = mask_dev[:, :, 1:, :, :] * mask_dev[:, :, :-1, :, :]
    mask_y = mask_dev[:, :, :, 1:, :] * mask_dev[:, :, :, :-1, :]
    mask_z = mask_dev[:, :, :, :, 1:] * mask_dev[:, :, :, :, :-1]

    mx = min(du_dx.shape[2], dv_dy.shape[2], dw_dz.shape[2])
    my = min(du_dx.shape[3], dv_dy.shape[3], dw_dz.shape[3])
    mz = min(du_dx.shape[4], dv_dy.shape[4], dw_dz.shape[4])

    div = (du_dx[:, :, :mx, :my, :mz] +
           dv_dy[:, :, :mx, :my, :mz] +
           dw_dz[:, :, :mx, :my, :mz])

    mask_int = (mask_x[:, :, :mx, :my, :mz] *
                mask_y[:, :, :mx, :my, :mz] *
                mask_z[:, :, :mx, :my, :mz])

    return (div ** 2 * mask_int).sum() / mask_int.sum().clamp(min=1)


def pressure_poisson_loss(pred, mask_dev, dx=1.0):
    """
    Simplified Pressure Poisson Equation (PPE):

        ∇²p = -ρ (∂uᵢ/∂xⱼ)(∂uⱼ/∂xᵢ)

    We use only diagonal terms for numerical stability:
        ∇²p ≈ -[ (∂u/∂x)² + (∂v/∂y)² + (∂w/∂z)² ]

    The Laplacian of pressure is computed with second-order central FD:
        ∇²p = (p_{i+1} - 2p_i + p_{i-1})/dx²  summed over x,y,z.

    Enforcing this loss teaches the model that pressure must satisfy the
    Poisson equation driven by velocity gradients — the core link between
    velocity and pressure in incompressible NS.
    """
    p = pred[:, 3:4].float()    # pressure channel, float32 for FD stability
    u = pred[:, 0:1].float()
    v = pred[:, 1:2].float()
    w = pred[:, 2:3].float()

    # Second-order central Laplacian of pressure.
    # Each term reduces a DIFFERENT dimension by 2, so they must be trimmed
    # to the common interior (X-2, Y-2, Z-2) before summing.
    #   lap_x: (B,1,X-2,Y,  Z  ) → trim Y,Z → (B,1,X-2,Y-2,Z-2)
    #   lap_y: (B,1,X,  Y-2,Z  ) → trim X,Z → (B,1,X-2,Y-2,Z-2)
    #   lap_z: (B,1,X,  Y,  Z-2) → trim X,Y → (B,1,X-2,Y-2,Z-2)
    lap_x = (p[:, :, 2:,  :,   :  ] - 2*p[:, :, 1:-1, :,    :   ] + p[:, :, :-2, :,   :  ]) / dx**2
    lap_y = (p[:, :, :,   2:,  :  ] - 2*p[:, :, :,    1:-1, :   ] + p[:, :, :,   :-2, :  ]) / dx**2
    lap_z = (p[:, :, :,   :,   2: ] - 2*p[:, :, :,    :,    1:-1] + p[:, :, :,   :,   :-2]) / dx**2

    # Trim each to the common interior
    lap_p = (lap_x[:, :, :,    1:-1, 1:-1] +   # (B,1,X-2,Y-2,Z-2)
             lap_y[:, :, 1:-1, :,    1:-1] +
             lap_z[:, :, 1:-1, 1:-1, :   ])

    # Diagonal velocity gradient terms (RHS of PPE) trimmed to same interior
    du_dx = (u[:, :, 1:, :, :] - u[:, :, :-1, :, :]) / dx   # (B,1,X-1,Y,Z)
    dv_dy = (v[:, :, :, 1:, :] - v[:, :, :, :-1, :]) / dx   # (B,1,X,Y-1,Z)
    dw_dz = (w[:, :, :, :, 1:] - w[:, :, :, :, :-1]) / dx   # (B,1,X,Y,Z-1)

    # Trim each gradient to the (X-2, Y-2, Z-2) interior.
    # du_dx: (B,1,X-1,Y,Z) — the X-dim is already 1 short; trim X with [:-1],
    #        and trim Y,Z with [1:-1].  Same pattern for dv_dy (Y short) and dw_dz (Z short).
    rhs = -(du_dx[:, :, :-1,  1:-1, 1:-1] ** 2 +   # (B,1,X-2,Y-2,Z-2) ✓
            dv_dy[:, :, 1:-1, :-1,  1:-1] ** 2 +   # (B,1,X-2,Y-2,Z-2) ✓
            dw_dz[:, :, 1:-1, 1:-1, :-1 ] ** 2)    # (B,1,X-2,Y-2,Z-2) ✓

    # Trim to common interior size (handles odd/even res edge cases)
    mx = min(lap_p.shape[2], rhs.shape[2])
    my = min(lap_p.shape[3], rhs.shape[3])
    mz = min(lap_p.shape[4], rhs.shape[4])

    lap_p = lap_p[:, :, :mx, :my, :mz]
    rhs   = rhs  [:, :, :mx, :my, :mz]

    # Interior vessel mask (trimmed to interior)
    mask_int = (mask_dev[:, :, 1:-1, 1:-1, 1:-1])[:, :, :mx, :my, :mz]

    residual = (lap_p - rhs) ** 2 * mask_int
    return residual.sum() / mask_int.sum().clamp(min=1)


def momentum_residual_loss(pred_next, pred_curr, mask_dev, dx=1.0):
    """
    Simplified Euler momentum: ρ ∂u/∂t = -∇p

    Approximation: ∂u/∂t ≈ u_next - u_curr  (unit dt, standardised fields)
    Therefore: ∇p ≈ -(u_next - u_curr)

    We compute the FD gradient of the predicted pressure and compare it to
    the negative velocity time-derivative.  This directly links the pressure
    channel to the velocity dynamics — the key to learning haemodynamics.
    """
    p = pred_next[:, 3:4].float()

    # FD pressure gradient (one-sided)
    dp_dx = (p[:, :, 1:, :, :] - p[:, :, :-1, :, :]) / dx
    dp_dy = (p[:, :, :, 1:, :] - p[:, :, :, :-1, :]) / dx
    dp_dz = (p[:, :, :, :, 1:] - p[:, :, :, :, :-1]) / dx

    # Velocity time derivatives (approximate ∂u/∂t)
    du = (pred_next[:, 0:1] - pred_curr[:, 0:1]).float()
    dv = (pred_next[:, 1:2] - pred_curr[:, 1:2]).float()
    dw = (pred_next[:, 2:3] - pred_curr[:, 2:3]).float()

    # Trim to common interior
    mx = min(dp_dx.shape[2], du.shape[2]) - 1
    my = min(dp_dy.shape[3], dv.shape[3]) - 1
    mz = min(dp_dz.shape[4], dw.shape[4]) - 1

    mask_x = mask_dev[:, :, 1:, :, :][:, :, :mx, :my, :mz]
    mask_y = mask_dev[:, :, :, 1:, :][:, :, :mx, :my, :mz]
    mask_z = mask_dev[:, :, :, :, 1:][:, :, :mx, :my, :mz]

    loss_x = ((dp_dx[:, :, :mx, :my, :mz] + du[:, :, :mx, :my, :mz]) ** 2 * mask_x).sum()
    loss_y = ((dp_dy[:, :, :mx, :my, :mz] + dv[:, :, :mx, :my, :mz]) ** 2 * mask_y).sum()
    loss_z = ((dp_dz[:, :, :mx, :my, :mz] + dw[:, :, :mx, :my, :mz]) ** 2 * mask_z).sum()

    denom = (mask_x.sum() + mask_y.sum() + mask_z.sum()).clamp(min=1)
    return (loss_x + loss_y + loss_z) / denom


def bc_loss(pred, wall_mask_dev):
    """No-slip: velocity = 0 at vessel wall voxels."""
    return torch.mean((pred[:, :3] * wall_mask_dev.float()) ** 2)


def pressure_stability_loss(pred, mask_dev):
    """Penalise large pressure spatial gradients — prevents pressure spikes."""
    p = pred[:, 3:4]
    return (
        ((p[:, :, 1:] - p[:, :, :-1]) ** 2 * mask_dev[:, :, 1:]).mean() +
        ((p[:, :, :, 1:] - p[:, :, :, :-1]) ** 2 * mask_dev[:, :, :, 1:]).mean() +
        ((p[:, :, :, :, 1:] - p[:, :, :, :, :-1]) ** 2 * mask_dev[:, :, :, :, 1:]).mean()
    )


def smoothness_loss(pred, prev_pred, mask_dev):
    """Penalise large vel+pres changes between consecutive predictions."""
    return torch.mean(((pred[:, :4] - prev_pred[:, :4]) * mask_dev) ** 2)


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
GRID_RES   = 64      # 8 GB VRAM handles 64³ comfortably with HUFNO3d
MODES      = 8       # must satisfy: modes ≤ (GRID_RES // 4) // 2 = 8
WIDTH      = 32      # base hidden width; try 48 or 64 if VRAM allows
NUM_LAYERS = 4       # HUFNO layers (each has SpectralConv + MiniUNet)

BATCH_SIZE    = 1
ROLLOUT_STEPS = 8
EPOCHS        = 10000

# Data optimizer: learns coarse global flow patterns, moderate LR
LR_DATA      = 1e-4    # higher start — CosineWarmRestarts will drive it down
# Physics optimizer: fine-grained PDE-constraint sculpting, always lower
LR_PHYS      = 5e-6
WEIGHT_DECAY = 1e-4
NOISE_STD    = 0.01

# Physics loss lambdas
PHYS_RAMP_START  = 0
PHYS_RAMP_END    = 400
LAMBDA_PHYS_MAX  = 0.05    # continuity (divergence-free)
LAMBDA_PPE_MAX   = 0.03    # pressure Poisson equation
LAMBDA_MOM_MAX   = 0.02    # momentum residual (∇p ≈ -∂u/∂t)
LAMBDA_BC_MAX    = 0.02    # no-slip boundary condition
PHYS_TARGET      = 0.10    # adaptive: aim to push physics loss below this

# Pressure gets 3× weight in the data MSE (primary clinical quantity)
PRESSURE_WEIGHT  = 3.0

# step_weights ramp up to emphasise long-horizon accuracy
assert ROLLOUT_STEPS <= 10, "Extend _base_weights if ROLLOUT_STEPS > 10."
_base_weights = [1.0, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5]
step_weights  = _base_weights[:ROLLOUT_STEPS]

DATA_PATH  = "../VelocityData3D"
WALL_PATH  = "../VelocityData3D/WallMesh/wall.vtp"
SAVE_DIR   = "../Models"

CHECKPOINT_PATH = os.path.join(SAVE_DIR, "fno_checkpoint.pth")
SAVE_PATH       = os.path.join(SAVE_DIR, "fno_model.pth")
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "fno_best.pth")
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ═══════════════════════════════════════════════════════════════════════════
#  1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════
loader = FNODataLoader(DATA_PATH, wall_file_path=WALL_PATH, resolution=GRID_RES)
fields, mask, grid_coords, stats = loader.load()
del loader
gc.collect()
torch.cuda.empty_cache()

# fields      : (T, 5, res, res, res)  — vel(3) + pres + time, standardised
# mask        : (res, res, res)         — binary vessel mask
# grid_coords : (3, res, res, res)      — normalised coords [0,1]
fields      = fields.cpu()
mask        = mask.cpu()
grid_coords = grid_coords.cpu()

T         = fields.shape[0]
val_split = int(0.8 * T)
train_fields = fields[:val_split]
val_fields   = fields[val_split:]
print(f"Timesteps: {T}  |  train pairs: {val_split - ROLLOUT_STEPS}  |  val pairs: {T - val_split - ROLLOUT_STEPS}")


# ═══════════════════════════════════════════════════════════════════════════
#  2. DATASET
# ═══════════════════════════════════════════════════════════════════════════
class TimeStepDataset(Dataset):
    """
    Returns (input_tensor, target_sequence).

    input_tensor    : (5+1+3, res, res, res)  — field | mask | coords
    target_sequence : (rollout_steps, 5, res, res, res)

    Augmentation (training only):
      1. Gaussian noise on vel+pres (not the time channel).
      2. Temporal reversal: flip velocity sign with 50% probability.
      3. Velocity scale jitter: ×U[0.9, 1.1] with 40% probability.
         (pressure is not scaled — it is not linearly related to velocity
          magnitude after standardisation; scaling pressure would break
          the Poisson physics.)
    """

    def __init__(self, fields, mask, coords,
                 rollout_steps=ROLLOUT_STEPS, noise_std=0.0):
        self.fields        = fields
        self.mask_ch       = mask.unsqueeze(0)    # (1, res, res, res)
        self.coords        = coords
        self.rollout_steps = rollout_steps
        self.noise_std     = noise_std
        self.n_pairs       = fields.shape[0] - rollout_steps
        self.training_mode = True

    def __len__(self):
        return self.n_pairs

    def __getitem__(self, idx):
        field_in = self.fields[idx]

        if self.training_mode:
            # 1. Gaussian noise on vel + pres channels
            if self.noise_std > 0:
                noise    = self.noise_std * torch.randn_like(field_in[:4])
                field_in = field_in.clone()
                field_in[:4] = field_in[:4] + noise * self.mask_ch

            # 2. Temporal reversal (50%)
            if torch.rand(1).item() < 0.5:
                field_in = field_in.clone()
                field_in[:3] = -field_in[:3]    # velocity sign only

            # 3. Velocity scale jitter (40%) — pressure NOT scaled
            if torch.rand(1).item() < 0.4:
                scale    = 0.9 + 0.2 * torch.rand(1).item()
                field_in = field_in.clone()
                field_in[:3] = field_in[:3] * scale

        inp     = torch.cat([field_in, self.mask_ch, self.coords], dim=0)
        targets = self.fields[idx + 1 : idx + 1 + self.rollout_steps]
        return inp, targets


def build_loaders(rollout_steps):
    train_ds = TimeStepDataset(
        train_fields, mask, grid_coords,
        rollout_steps=rollout_steps, noise_std=NOISE_STD,
    )
    val_ds = TimeStepDataset(
        val_fields, mask, grid_coords,
        rollout_steps=rollout_steps, noise_std=0.0,
    )
    return (
        train_ds,
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=False),
        val_ds,
        DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=False),
    )


# ═══════════════════════════════════════════════════════════════════════════
#  3. MODEL, OPTIMIZERS, SCHEDULER
# ═══════════════════════════════════════════════════════════════════════════
in_ch  = 5 + 1 + 3    # 5 field channels + 1 mask + 3 coords
out_ch = 5

model = HUFNO3d(
    modes=MODES, width=WIDTH,
    in_channels=in_ch, out_channels=out_ch,
    num_layers=NUM_LAYERS,
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"HUFNO3d — {n_params:,} parameters  ({n_params/1e6:.1f} M)")

# Data optimizer: higher LR, learns global flow from data
optimizer_data = optim.Adam(model.parameters(), lr=LR_DATA, weight_decay=WEIGHT_DECAY)
# Physics optimizer: lower LR, fine-tunes to satisfy PDE constraints
optimizer_phys = optim.Adam(model.parameters(), lr=LR_PHYS, weight_decay=WEIGHT_DECAY)

# CosineAnnealingWarmRestarts: periodic LR kicks escape loss plateaus.
# T_0=300 → first restart at epoch 300, then 600, 1200, 2400, ...
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer_data, T_0=300, T_mult=2, eta_min=1e-7,
)


# ═══════════════════════════════════════════════════════════════════════════
#  4. CHECKPOINT LOADING
# ═══════════════════════════════════════════════════════════════════════════
start_epoch   = 0
best_val_loss = float('inf')

if os.path.exists(CHECKPOINT_PATH):
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, weights_only=False)
    try:
        model.load_state_dict(ckpt["model_state_dict"])

        if "optimizer_data_state_dict" in ckpt:
            optimizer_data.load_state_dict(ckpt["optimizer_data_state_dict"])
        if "optimizer_phys_state_dict" in ckpt:
            optimizer_phys.load_state_dict(ckpt["optimizer_phys_state_dict"])

        start_epoch = ckpt["epoch"] + 1

        # Resume LRs
        for pg in optimizer_data.param_groups:
            pg["lr"] = 5e-5;  pg["initial_lr"] = 5e-5
        for pg in optimizer_phys.param_groups:
            pg["lr"] = LR_PHYS; pg["initial_lr"] = LR_PHYS

        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer_data, T_0=300, T_mult=2, eta_min=1e-7,
        )
        print(f"Resuming from epoch {start_epoch}")

    except RuntimeError as e:
        print(f"\n⚠  Checkpoint architecture mismatch — starting fresh.")
        print(f"   ({e})\n")
        start_epoch = 0

    dataset, train_loader, val_dataset, val_loader = build_loaders(ROLLOUT_STEPS)
else:
    dataset, train_loader, val_dataset, val_loader = build_loaders(ROLLOUT_STEPS)
    print("No checkpoint — starting fresh.")


# ═══════════════════════════════════════════════════════════════════════════
#  5. MASKS AND METRICS
# ═══════════════════════════════════════════════════════════════════════════
mask_dev      = mask.unsqueeze(0).unsqueeze(0).to(device)   # (1,1,res,res,res)
_not_mask     = (~mask.bool()).float().unsqueeze(0).unsqueeze(0).to(device)
_dilated      = F.max_pool3d(_not_mask, kernel_size=3, stride=1, padding=1)
wall_mask_dev = (mask_dev.float() * _dilated).bool()
print(f"Wall voxels: {wall_mask_dev.sum().item()}")

grid_coords_dev = grid_coords.unsqueeze(0).to(device)


def masked_mse(pred, target, pressure_weight=1.0):
    """
    MSE inside the vessel.  Optionally up-weight the pressure channel.
    If pressure_weight > 1, channel 3 contributes pressure_weight× more.
    """
    sq = (pred - target) ** 2 * mask_dev   # (B, 5, res, res, res)
    if pressure_weight != 1.0:
        # Build per-channel weight tensor: [1,1,1,pw,1]
        w = pred.new_ones(1, pred.shape[1], 1, 1, 1)
        w[0, 3] = pressure_weight
        sq = sq * w
    return sq.sum() / (mask_dev.sum() * pred.shape[1])


def pressure_mse(pred, target):
    """MSE for pressure channel only — tracked separately for monitoring."""
    sq = (pred[:, 3:4] - target[:, 3:4]) ** 2 * mask_dev
    return sq.sum() / mask_dev.sum().clamp(min=1)


@torch.no_grad()
def masked_rel_l2(pred, target):
    d = (pred - target) * mask_dev
    t = target * mask_dev
    return torch.sqrt((d**2).sum() / ((t**2).sum() + 1e-8))


@torch.no_grad()
def pressure_rel_l2(pred, target):
    d = (pred[:, 3:4] - target[:, 3:4]) * mask_dev
    t = target[:, 3:4] * mask_dev
    return torch.sqrt((d**2).sum() / ((t**2).sum() + 1e-8))


def build_input(field_t):
    return torch.cat(
        [field_t, mask.unsqueeze(0), grid_coords], dim=0
    ).unsqueeze(0)


def get_grad_norm(model):
    total = sum(
        p.grad.data.norm(2).item() ** 2
        for p in model.parameters() if p.grad is not None
    )
    return total ** 0.5


# ═══════════════════════════════════════════════════════════════════════════
#  6. TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════
print("\n--- Starting HUFNO3d Training ---")
t_start        = time.time()
t_epoch        = t_start
epoch_phys_avg = 1.0
lambda_phys    = 0.0
lambda_ppe     = 0.0
lambda_mom     = 0.0
lambda_bc      = 0.0

for epoch in range(start_epoch, EPOCHS):
    model.train()
    dataset.training_mode = True

    epoch_loss     = 0.0
    epoch_mse      = 0.0
    epoch_pres_mse = 0.0
    epoch_phys     = 0.0
    epoch_ppe      = 0.0
    epoch_mom      = 0.0
    epoch_bc       = 0.0
    grad_norm_data = 0.0
    grad_norm_phys = 0.0

    # ── Lambda scheduling ─────────────────────────────────────────────────
    # Phase 1 (epoch ≤ PHYS_RAMP_END): linear ramp from 0 → max.
    # Phase 2 (epoch > PHYS_RAMP_END): adaptive only, adjusted every 50 epochs.
    #   The old code had an else branch that reset lambdas to the ramp value on
    #   every non-50 epoch, so adaptive changes only survived for 1 epoch.
    if epoch <= PHYS_RAMP_END:
        progress = max(0.0, min(1.0,
            (epoch - PHYS_RAMP_START) / max(1, PHYS_RAMP_END - PHYS_RAMP_START)
        ))
        lambda_phys = LAMBDA_PHYS_MAX * progress
        lambda_ppe  = LAMBDA_PPE_MAX  * progress
        lambda_mom  = LAMBDA_MOM_MAX  * progress
        lambda_bc   = LAMBDA_BC_MAX   * progress
    elif epoch % 50 == 0:
        # Adaptive: nudge based on how physics is doing
        if epoch_phys_avg > PHYS_TARGET * 2:       # physics still bad → push harder
            lambda_phys = min(lambda_phys * 1.10, LAMBDA_PHYS_MAX)
            lambda_ppe  = min(lambda_ppe  * 1.10, LAMBDA_PPE_MAX)
            lambda_mom  = min(lambda_mom  * 1.10, LAMBDA_MOM_MAX)
            lambda_bc   = min(lambda_bc   * 1.10, LAMBDA_BC_MAX)
        elif epoch_phys_avg < PHYS_TARGET:          # physics good → ease off
            lambda_phys = max(lambda_phys * 0.95, 0.005)
            lambda_ppe  = max(lambda_ppe  * 0.95, 0.003)
            lambda_mom  = max(lambda_mom  * 0.95, 0.002)
            lambda_bc   = max(lambda_bc   * 0.95, 0.002)
    # else: between adjustment epochs → keep current lambda values unchanged

    for inp, tgts in train_loader:
        inp  = inp.to(device)
        tgts = tgts.to(device)

        # ── Pass 1: Data loss ─────────────────────────────────────────────
        optimizer_data.zero_grad()
        loss_data_total = torch.zeros(1, device=device)
        current_data    = inp
        prev_pred_data  = None

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for s in range(dataset.rollout_steps):
                pred  = model(current_data)
                tgt_s = tgts[:, s]

                # Pressure-weighted MSE: ch3 contributes PRESSURE_WEIGHT× more
                mse_s      = masked_mse(pred, tgt_s, pressure_weight=PRESSURE_WEIGHT)
                loss_step  = mse_s * step_weights[s]
                epoch_mse += mse_s.item()
                epoch_pres_mse += pressure_mse(pred, tgt_s).item()

                if s > 0 and prev_pred_data is not None:
                    loss_step = loss_step + 0.01 * smoothness_loss(
                        pred, prev_pred_data, mask_dev
                    )
                prev_pred_data  = pred
                loss_data_total = loss_data_total + loss_step

                if s < dataset.rollout_steps - 1:
                    current_data = torch.cat([
                        pred.detach(),
                        mask_dev.expand(inp.shape[0], -1, -1, -1, -1),
                        grid_coords_dev.expand(inp.shape[0], -1, -1, -1, -1),
                    ], dim=1)

        # Gradient clip: tighter at high LR to prevent early divergence
        clip_val = 0.5 if optimizer_data.param_groups[0]["lr"] > 1e-4 else 1.0
        loss_data_total.backward()   # no retain_graph — phys does fresh forward
        grad_norm_data = get_grad_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)
        optimizer_data.step()

        # ── Pass 2: Physics loss + anchor ────────────────────────────────
        optimizer_phys.zero_grad()
        loss_phys_total = torch.zeros(1, device=device)
        current_phys    = inp
        prev_pred_phys  = None

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for s in range(dataset.rollout_steps):
                pred_phys = model(current_phys)
                tgt_s     = tgts[:, s]

                # --- Core physics constraints ---
                l_div  = fd_physics_loss(pred_phys, mask_dev)
                l_ppe  = pressure_poisson_loss(pred_phys, mask_dev)
                l_bc   = bc_loss(pred_phys, wall_mask_dev)
                l_pstab = pressure_stability_loss(pred_phys, mask_dev)

                # Momentum residual: links ∇p to velocity change
                l_mom = torch.zeros(1, device=device)
                if prev_pred_phys is not None:
                    l_mom = momentum_residual_loss(pred_phys, prev_pred_phys, mask_dev)

                if torch.isnan(l_div): l_div = torch.zeros(1, device=device)
                if torch.isnan(l_ppe): l_ppe = torch.zeros(1, device=device)
                if torch.isnan(l_bc):  l_bc  = torch.zeros(1, device=device)
                if torch.isnan(l_mom): l_mom = torch.zeros(1, device=device)

                # Anchor loss (weight=0.1): tethers physics optimizer to real
                # CFD data so it cannot satisfy div=0 by making everything zero.
                anchor = masked_mse(pred_phys, tgt_s, pressure_weight=PRESSURE_WEIGHT)

                loss_phys_total = (
                    loss_phys_total
                    + lambda_phys * l_div
                    + lambda_ppe  * l_ppe
                    + lambda_mom  * l_mom
                    + lambda_bc   * l_bc
                    + 0.05        * l_pstab
                    + 0.1         * anchor     # 0.1 not 1.0 — physics must compete
                )

                epoch_phys += l_div.item()
                epoch_ppe  += l_ppe.item()
                epoch_mom  += l_mom.item()
                epoch_bc   += l_bc.item()
                prev_pred_phys = pred_phys

                if s < dataset.rollout_steps - 1:
                    current_phys = torch.cat([
                        pred_phys.detach(),
                        mask_dev.expand(inp.shape[0], -1, -1, -1, -1),
                        grid_coords_dev.expand(inp.shape[0], -1, -1, -1, -1),
                    ], dim=1)

        loss_phys_total.backward()
        grad_norm_phys = get_grad_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer_phys.step()

        epoch_loss += (loss_data_total + loss_phys_total).item()

    scheduler.step()

    n_batch        = max(1, len(train_loader))
    avg_loss       = epoch_loss  / n_batch
    epoch_phys_avg = epoch_phys  / n_batch

    # ── Light logging every 10 epochs ─────────────────────────────────────
    if epoch % 10 == 0 and epoch % 100 != 0:
        lr_now = optimizer_data.param_groups[0]["lr"]
        print(f"  Epoch {epoch:5d} | loss {avg_loss:.4f} | "
              f"div {epoch_phys_avg:.4f} | ppe {epoch_ppe/n_batch:.4f} | "
              f"pres_mse {epoch_pres_mse/n_batch:.4f} | LR {lr_now:.1e}")

    # ── Full evaluation every 100 epochs ──────────────────────────────────
    if epoch % 100 == 0:
        model.eval()
        dataset.training_mode = False
        elapsed = time.time() - t_epoch

        # Validation: overall + pressure-only MSE
        val_loss = 0.0
        val_pres = 0.0
        with torch.no_grad():
            for inp_v, tgt_v in val_loader:
                inp_v, tgt_v = inp_v.to(device), tgt_v.to(device)
                pred_v = model(inp_v)
                val_loss += masked_mse(pred_v, tgt_v[:, 0]).item()
                val_pres += pressure_mse(pred_v, tgt_v[:, 0]).item()
        val_loss /= max(1, len(val_loader))
        val_pres /= max(1, len(val_loader))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "stats": stats, "mask": mask, "grid_coords": grid_coords,
            }, BEST_MODEL_PATH)
            print(f"  ★ New best model saved (val={val_loss:.5f}, pres_val={val_pres:.5f})")

        # One-step accuracy (velocity + pressure separately)
        total_rel = 0.0
        total_pres_rel = 0.0
        with torch.no_grad():
            for i in range(T - 1):
                inp_i  = build_input(fields[i]).to(device)
                tgt_i  = fields[i + 1].unsqueeze(0).to(device)
                pred_i = model(inp_i)
                total_rel      += masked_rel_l2(pred_i, tgt_i).item()
                total_pres_rel += pressure_rel_l2(pred_i, tgt_i).item()
        acc_1step    = (1.0 - total_rel      / (T - 1)) * 100.0
        pres_acc     = (1.0 - total_pres_rel / (T - 1)) * 100.0

        # Autoregressive rollout from t=0
        rollout_n = min(10, T - 1)
        current   = fields[0].unsqueeze(0).to(device)
        with torch.no_grad():
            for s in range(rollout_n):
                inp_s = torch.cat([
                    current[0],
                    mask.unsqueeze(0).to(device),
                    grid_coords.to(device),
                ], dim=0).unsqueeze(0)
                current = model(inp_s)
        tgt_roll   = fields[rollout_n].unsqueeze(0).to(device)
        rollout_acc = (1.0 - masked_rel_l2(current, tgt_roll).item()) * 100.0
        pres_roll   = (1.0 - pressure_rel_l2(current, tgt_roll).item()) * 100.0

        lr_now = optimizer_data.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:5d} | "
            f"MSE {epoch_mse/n_batch:.5f} | "
            f"PRes {epoch_pres_mse/n_batch:.5f} | "
            f"Div {epoch_phys/n_batch:.5f} | "
            f"PPE {epoch_ppe/n_batch:.5f} | "
            f"Mom {epoch_mom/n_batch:.5f} | "
            f"BC {epoch_bc/n_batch:.5f} | "
            f"1-step {acc_1step:.1f}% | "
            f"P-1step {pres_acc:.1f}% | "
            f"Rollout-{rollout_n} {rollout_acc:.1f}% | "
            f"P-Roll {pres_roll:.1f}% | "
            f"Val {val_loss:.5f} | "
            f"ValP {val_pres:.5f} | "
            f"LR {lr_now:.1e} | "
            f"{elapsed:.0f}s"
        )
        print(f"  ∇data {grad_norm_data:.4f} | ∇phys {grad_norm_phys:.4f} | "
              f"lp {lambda_phys:.3f} | lppe {lambda_ppe:.3f} | lmom {lambda_mom:.3f}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_data_state_dict": optimizer_data.state_dict(),
            "optimizer_phys_state_dict": optimizer_phys.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": avg_loss,
            "stats": stats, "mask": mask, "grid_coords": grid_coords,
        }, CHECKPOINT_PATH)
        t_epoch = time.time()


# ═══════════════════════════════════════════════════════════════════════════
#  7. SAVE FINAL MODEL
# ═══════════════════════════════════════════════════════════════════════════
total_mins = (time.time() - t_start) / 60
print(f"\nTraining complete in {total_mins:.2f} min.")
torch.save({
    "model_state_dict": model.state_dict(),
    "stats": stats, "mask": mask, "grid_coords": grid_coords,
}, SAVE_PATH)
print(f"Model saved → {SAVE_PATH}")