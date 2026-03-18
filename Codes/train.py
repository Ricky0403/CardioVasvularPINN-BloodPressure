"""
Training script for the 3-D Fourier Neural Operator (FNO).

Paradigm shift from the original PINN training:
  - PINN:  maps single point (x,y,z,t) → (u,v,w,p)       (point-wise)
  - FNO:   maps entire 3-D field at time t → field at t+Δt (operator)

The network is trained with pure supervised MSE loss on voxelized fields.
No physics loss (autograd derivatives) is needed — the spectral convolutions
in Fourier space capture the PDE dynamics directly from data.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import FNO3d
from fno_data_loader import FNODataLoader

torch.set_float32_matmul_precision("high")


def fd_physics_loss(pred, mask_dev, dx=1.0):
    """
    Finite-difference continuity (divergence-free) loss.
    Only applied at interior vessel voxels with valid neighbors.
    """
    u = pred[:, 0:1]
    v = pred[:, 1:2]
    w = pred[:, 2:3]

    # One-sided differences at interior points (avoids wrap-around artifacts)
    du_dx = (u[:, :, 1:, :, :] - u[:, :, :-1, :, :]) / dx
    dv_dy = (v[:, :, :, 1:, :] - v[:, :, :, :-1, :]) / dx
    dw_dz = (w[:, :, :, :, 1:] - w[:, :, :, :, :-1]) / dx

    # Trim mask to match the smaller tensor sizes
    mask_x = mask_dev[:, :, 1:, :, :] * mask_dev[:, :, :-1, :, :]
    mask_y = mask_dev[:, :, :, 1:, :] * mask_dev[:, :, :, :-1, :]
    mask_z = mask_dev[:, :, :, :, 1:] * mask_dev[:, :, :, :, :-1]

    # Divergence on the trimmed interior
    min_x = min(du_dx.shape[2], dv_dy.shape[2], dw_dz.shape[2])
    min_y = min(du_dx.shape[3], dv_dy.shape[3], dw_dz.shape[3])
    min_z = min(du_dx.shape[4], dv_dy.shape[4], dw_dz.shape[4])

    div = (du_dx[:, :, :min_x, :min_y, :min_z] +
           dv_dy[:, :, :min_x, :min_y, :min_z] +
           dw_dz[:, :, :min_x, :min_y, :min_z])

    mask_int = (mask_x[:, :, :min_x, :min_y, :min_z] *
                mask_y[:, :, :min_x, :min_y, :min_z] *
                mask_z[:, :, :min_x, :min_y, :min_z])

    n_valid = mask_int.sum().clamp(min=1)
    return (div ** 2 * mask_int).sum() / n_valid


def bc_loss(pred, wall_mask_dev):
    vel_at_wall = pred[:, :3] * wall_mask_dev.float()
    return torch.mean(vel_at_wall ** 2)


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
GRID_RES   = 32        # back to 32; at 4GB this is the realistic ceiling
MODES      = 8         # back to 8
WIDTH      = 32        # back to 32; width=64 alone uses ~4x the activation memory
NUM_LAYERS = 4

BATCH_SIZE = 1         # must be 1 for 8-step rollout on 4GB
ROLLOUT_STEPS = 8
EPOCHS = 22000
LR = 1e-5                # very low — just nudging, not retraining
LR_STEP    = 100       # Halve the LR every LR_STEP epochs (paper: 100)
LR_GAMMA   = 0.5
WEIGHT_DECAY = 1e-4
NOISE_STD  = 0.01      # Gaussian noise injected into inputs (regularization)
PHYS_RAMP_START = 0      # ramp is already complete by epoch 12901
PHYS_RAMP_END   = 400    # so lambda will be max from epoch 1 of resume
LAMBDA_PHYS_MAX = 0.05
LAMBDA_BC_MAX   = 0.02

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
#  1. LOAD & VOXELIZE DATA
# ═══════════════════════════════════════════════════════════════════════════
loader = FNODataLoader(DATA_PATH, wall_file_path=WALL_PATH, resolution=GRID_RES)
fields, mask, grid_coords, stats = loader.load()
del loader
import gc; gc.collect()
torch.cuda.empty_cache()

# Keep fields, mask, grid_coords on CPU — DataLoader will move batches to GPU
fields      = fields.cpu()
mask        = mask.cpu()
grid_coords = grid_coords.cpu()

# fields      : (T, 4, res, res, res)  — standardized velocity(3) + pressure(1)
# mask        : (res, res, res)         — binary vessel mask
# grid_coords : (3, res, res, res)      — normalised spatial coordinates [0,1]

T = fields.shape[0]
val_split = int(0.8 * T)
train_fields = fields[:val_split]
val_fields   = fields[val_split:]
print(f"Total timesteps: {T}, training pairs: {val_split - ROLLOUT_STEPS}")


# ═══════════════════════════════════════════════════════════════════════════
#  2. DATASET  — consecutive timestep pairs
# ═══════════════════════════════════════════════════════════════════════════
class TimeStepDataset(Dataset):
    def __init__(self, fields, mask, coords, rollout_steps=ROLLOUT_STEPS, noise_std=0.0):
        self.fields        = fields
        self.mask_ch       = mask.unsqueeze(0)
        self.coords        = coords
        self.rollout_steps = rollout_steps
        self.noise_std     = noise_std
        self.n_pairs       = fields.shape[0] - rollout_steps
        self.training_mode = True

    def __len__(self):
        return self.n_pairs

    def __getitem__(self, idx):
        # Return a sequence of rollout_steps consecutive targets
        field_in = self.fields[idx]
        if self.training_mode:
            # 1. Gaussian noise on velocity only
            if self.noise_std > 0:
                noise = self.noise_std * torch.randn_like(field_in[:4])
                field_in = field_in.clone()
                field_in[:4] = field_in[:4] + noise * self.mask_ch

            # 2. Random temporal reversal (flow can run backwards)
            if torch.rand(1).item() < 0.3:
                field_in = field_in.clone()
                field_in[:3] = -field_in[:3]   # flip velocity sign

        inp = torch.cat([field_in, self.mask_ch, self.coords], dim=0)
        targets = self.fields[idx + 1 : idx + 1 + self.rollout_steps]
        return inp, targets


def build_loaders(rollout_steps):
    train_ds = TimeStepDataset(
        train_fields,
        mask,
        grid_coords,
        rollout_steps=rollout_steps,
        noise_std=NOISE_STD,
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=False,
    )
    val_ds = TimeStepDataset(
        val_fields,
        mask,
        grid_coords,
        rollout_steps=rollout_steps,
        noise_std=0.0,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=False,
    )
    return train_ds, train_dl, val_ds, val_dl


# ═══════════════════════════════════════════════════════════════════════════
#  3. MODEL, OPTIMIZER, SCHEDULER
# ═══════════════════════════════════════════════════════════════════════════
in_ch  = 5 + 1 + 3   # field(5: vel+pres+time) + mask + coords
out_ch = 5

model = FNO3d(
    modes1=MODES, modes2=MODES, modes3=MODES,
    width=WIDTH,
    in_channels=in_ch,
    out_channels=out_ch,
    num_layers=NUM_LAYERS,
).to(device)

# model = torch.compile(model)

n_params = sum(p.numel() for p in model.parameters())
print(f"FNO3d — {n_params:,} parameters")

optimizer_data = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
optimizer_phys = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer_data,
    T_max=2000,
    eta_min=1e-6,
)


# ═══════════════════════════════════════════════════════════════════════════
#  4. CHECKPOINT LOADING
# ═══════════════════════════════════════════════════════════════════════════
start_epoch = 0
if os.path.exists(CHECKPOINT_PATH):
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if "optimizer_data_state_dict" in ckpt and "optimizer_phys_state_dict" in ckpt:
        optimizer_data.load_state_dict(ckpt["optimizer_data_state_dict"])
        optimizer_phys.load_state_dict(ckpt["optimizer_phys_state_dict"])
    elif "optimizer_state_dict" in ckpt:
        optimizer_data.load_state_dict(ckpt["optimizer_state_dict"])
        optimizer_phys.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = ckpt["epoch"] + 1

    resume_lr = 5e-6
    for param_group in optimizer_data.param_groups:
        param_group["lr"] = resume_lr
        param_group["initial_lr"] = resume_lr
    for param_group in optimizer_phys.param_groups:
        param_group["lr"] = 1e-6
        param_group["initial_lr"] = 1e-6

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_data,
        T_max=3000,
        eta_min=1e-7,
    )
    dataset, train_loader, val_dataset, val_loader = build_loaders(ROLLOUT_STEPS)
    print(f"Resuming from epoch {start_epoch}, LR reset to {resume_lr}")
else:
    dataset, train_loader, val_dataset, val_loader = build_loaders(ROLLOUT_STEPS)
    print("No checkpoint found — starting fresh.")


# ═══════════════════════════════════════════════════════════════════════════
#  5. HELPER METRICS
# ═══════════════════════════════════════════════════════════════════════════
mask_dev = mask.unsqueeze(0).unsqueeze(0).to(device)   # (1, 1, res, res, res)
from torch.nn.functional import max_pool3d
_not_mask = (~mask.bool()).float().unsqueeze(0).unsqueeze(0).to(device)
_dilated  = max_pool3d(_not_mask, kernel_size=3, stride=1, padding=1)
wall_mask_dev = (mask_dev.float() * _dilated).bool()
print(f"Wall voxels: {wall_mask_dev.sum().item()}")


def masked_mse(pred, target):
    """MSE loss inside the vessel only."""
    sq = (pred - target) ** 2 * mask_dev
    return sq.sum() / (mask_dev.sum() * pred.shape[1])


@torch.no_grad()
def masked_rel_l2(pred, target):
    """Relative L2 error inside the vessel."""
    d = (pred - target) * mask_dev
    t = target * mask_dev
    return torch.sqrt((d ** 2).sum() / ((t ** 2).sum() + 1e-8))


def build_input(field_t):
    """Construct a single FNO input from a field tensor."""
    return torch.cat([field_t, mask.unsqueeze(0), grid_coords], dim=0).unsqueeze(0)


def pressure_stability_loss(pred, mask_dev):
    """Penalize large pressure spatial gradients."""
    p = pred[:, 3:4]
    dp_dx = (p[:, :, 1:, :, :] - p[:, :, :-1, :, :])
    dp_dy = (p[:, :, :, 1:, :] - p[:, :, :, :-1, :])
    dp_dz = (p[:, :, :, :, 1:] - p[:, :, :, :, :-1])
    loss = (
        (dp_dx ** 2 * mask_dev[:, :, 1:, :, :]).mean() +
        (dp_dy ** 2 * mask_dev[:, :, :, 1:, :]).mean() +
        (dp_dz ** 2 * mask_dev[:, :, :, :, 1:]).mean()
    )
    return loss


def smoothness_loss(pred, prev_field, mask_dev):
    """Penalize large changes between consecutive predictions."""
    diff = (pred[:, :4] - prev_field[:, :4]) * mask_dev  # only vel+pres channels
    return torch.mean(diff ** 2)


def get_grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


# ═══════════════════════════════════════════════════════════════════════════
#  6. TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════
print("\n--- Starting FNO Training ---")
t_start = time.time()
t_epoch = t_start
grid_coords_dev = grid_coords.unsqueeze(0).to(device)
mask_inp_dev = mask_dev.to(device)
epoch_phys_avg = 1.0   # initialize high so lambda starts pushing
PHYS_TARGET = 0.20     # aim to get physics loss below this
lambda_phys = 0.0
lambda_bc = 0.0
best_val_loss = float('inf')
step_weights = [1.0, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5]

for epoch in range(start_epoch, EPOCHS):
    model.train()
    dataset.training_mode = True
    epoch_loss = 0.0
    epoch_mse = 0.0
    epoch_phys = 0.0
    epoch_bc = 0.0
    grad_norm_data = 0.0
    grad_norm_phys = 0.0

    progress = max(0.0, min(1.0, (epoch - PHYS_RAMP_START) / (PHYS_RAMP_END - PHYS_RAMP_START)))

    if epoch > 0 and epoch % 50 == 0:
        if epoch_phys_avg > PHYS_TARGET * 2:      # physics too bad
            lambda_phys = min(lambda_phys * 1.10, LAMBDA_PHYS_MAX)
            lambda_bc = min(lambda_bc * 1.10, LAMBDA_BC_MAX)
        elif epoch_phys_avg < PHYS_TARGET:         # physics good — relax
            lambda_phys = max(lambda_phys * 0.95, 0.01)
            lambda_bc = max(lambda_bc * 0.95, 0.005)
    else:
        lambda_phys = LAMBDA_PHYS_MAX * progress
        lambda_bc = LAMBDA_BC_MAX * progress

    for inp, tgts in train_loader:
        inp  = inp.to(device)
        tgts = tgts.to(device)

        optimizer_data.zero_grad()
        loss_data_total = 0.0
        current_data = inp
        prev_pred_data = None

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for s in range(dataset.rollout_steps):
                pred = model(current_data)
                tgt_s = tgts[:, s]

                mse_unweighted = masked_mse(pred, tgt_s)
                loss_data = mse_unweighted * step_weights[s]
                epoch_mse += mse_unweighted.item()

                if s > 0 and prev_pred_data is not None:
                    loss_smooth = smoothness_loss(pred, prev_pred_data, mask_dev)
                    loss_data = loss_data + 0.01 * loss_smooth  # small weight — just regularization
                prev_pred_data = pred

                loss_data_total += loss_data

                if s < dataset.rollout_steps - 1:
                    next_field = pred.detach()
                    current_data = torch.cat([
                        next_field,
                        mask_dev.expand(inp.shape[0], -1, -1, -1, -1),
                        grid_coords.unsqueeze(0).expand(inp.shape[0], -1, -1, -1, -1).to(device)
                    ], dim=1)

        current_lr = optimizer_data.param_groups[0]["lr"]
        clip_val = 0.3 if current_lr > 1e-4 else 1.0   # tighter clip at high LR

        loss_data_total.backward(retain_graph=True)
        grad_norm_data = get_grad_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)
        optimizer_data.step()

        optimizer_phys.zero_grad()
        loss_phys_only = 0.0
        current_phys = inp

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for s in range(dataset.rollout_steps):
                pred_phys = model(current_phys)

                loss_phys_val = fd_physics_loss(pred_phys, mask_dev)
                loss_bc_val = bc_loss(pred_phys, wall_mask_dev)
                loss_pres_stab = pressure_stability_loss(pred_phys, mask_dev)

                if torch.isnan(loss_phys_val):
                    loss_phys_val = torch.tensor(0.0, device=device)
                if torch.isnan(loss_bc_val):
                    loss_bc_val = torch.tensor(0.0, device=device)

                loss_phys_only = (
                    loss_phys_only
                    + lambda_phys * loss_phys_val
                    + lambda_bc * loss_bc_val
                    + 0.05 * loss_pres_stab
                )
                epoch_phys += loss_phys_val.item()
                epoch_bc += loss_bc_val.item()

                if s < dataset.rollout_steps - 1:
                    next_field_phys = pred_phys.detach()
                    current_phys = torch.cat([
                        next_field_phys,
                        mask_dev.expand(inp.shape[0], -1, -1, -1, -1),
                        grid_coords.unsqueeze(0).expand(inp.shape[0], -1, -1, -1, -1).to(device)
                    ], dim=1)

        loss_phys_only.backward()
        grad_norm_phys = get_grad_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer_phys.step()

        epoch_loss += (loss_data_total + loss_phys_only).item()

    scheduler.step()
    avg_loss = epoch_loss / len(train_loader)
    epoch_phys_avg = epoch_phys / len(train_loader)

    # ── Periodic evaluation ──────────────────────────────────────────────
    if epoch % 10 == 0 and epoch % 100 != 0:
        lr_now = optimizer_data.param_groups[0]["lr"]
        phys_avg = epoch_phys / max(1, len(train_loader))
        print(f"  Epoch {epoch} | loss {epoch_loss/len(train_loader):.4f} | phys {phys_avg:.4f} | lp {lambda_phys:.3f} | LR {lr_now:.1e}")

    if epoch % 100 == 0:
        model.eval()
        dataset.training_mode = False
        elapsed = time.time() - t_epoch

        val_loss = 0.0
        with torch.no_grad():
            for inp_v, tgt_v in val_loader:
                inp_v, tgt_v = inp_v.to(device), tgt_v.to(device)
                pred_v = model(inp_v)
                val_loss += masked_mse(pred_v, tgt_v[:, 0]).item()
        val_loss /= max(1, len(val_loader))
        print(f"  Val loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "stats": stats, "mask": mask, "grid_coords": grid_coords,
            }, BEST_MODEL_PATH)
            print(f"  New best model saved (val={val_loss:.5f})")

        # A. One-step accuracy (all pairs)
        total_rel = 0.0
        with torch.no_grad():
            for i in range(T - 1):
                inp_i = build_input(fields[i]).to(device)
                tgt_i = fields[i + 1].unsqueeze(0).to(device)
                pred_i = model(inp_i)
                total_rel += masked_rel_l2(pred_i, tgt_i).item()
        avg_rel = total_rel / (T - 1)
        acc_1step = (1.0 - avg_rel) * 100.0

        # B. Autoregressive rollout (from t=0)
        rollout_steps = min(10, T - 1)
        current = fields[0].unsqueeze(0).to(device)
        with torch.no_grad():
            for s in range(rollout_steps):
                inp_s = torch.cat(
                    [current[0], mask.unsqueeze(0).to(device),
                     grid_coords.to(device)], dim=0,
                ).unsqueeze(0)
                current = model(inp_s)
        tgt_roll = fields[rollout_steps].unsqueeze(0).to(device)
        rollout_acc = (1.0 - masked_rel_l2(current, tgt_roll).item()) * 100.0

        lr_now = optimizer_data.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:4d} | "
            f"MSE {epoch_mse/len(train_loader):.5f} | "
            f"Phys {epoch_phys/len(train_loader):.5f} | "
            f"BC {epoch_bc/len(train_loader):.5f} | "
            f"1-step {acc_1step:.1f}% | "
            f"Rollout-{rollout_steps} {rollout_acc:.1f}% | "
            f"Val {val_loss:.5f} | "
            f"lp {lambda_phys:.3f} | "
            f"LR {optimizer_data.param_groups[0]['lr']:.1e} | "
            f"{elapsed:.0f}s"
        )
        print(f"  Grad norm after data loss: {grad_norm_data:.4f} | after phys loss: {grad_norm_phys:.4f}")

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_data.state_dict(),
                "optimizer_data_state_dict": optimizer_data.state_dict(),
                "optimizer_phys_state_dict": optimizer_phys.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_loss,
                "stats": stats,
                "mask": mask,
                "grid_coords": grid_coords,
            },
            CHECKPOINT_PATH,
        )
        t_epoch = time.time()

# ═══════════════════════════════════════════════════════════════════════════
#  7. SAVE FINAL MODEL
# ═══════════════════════════════════════════════════════════════════════════
total_mins = (time.time() - t_start) / 60
print(f"\nTraining complete in {total_mins:.2f} min.")
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "stats": stats,
        "mask": mask,
        "grid_coords": grid_coords,
    },
    SAVE_PATH,
)
print(f"Model saved to {SAVE_PATH}")
