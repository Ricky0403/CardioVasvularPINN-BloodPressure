import gc
import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import HFNO3d
from fno_data_loader import FNODataLoader

torch.set_float32_matmul_precision("high")

# ═══════════════════════════════════════════════════════════════════════════
#  PHYSICS / BC HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def fd_physics_loss(pred, mask_dev, dx=1.0):
    """
    Finite-difference continuity (divergence-free) loss.
    Only applied at interior vessel voxels with valid neighbors on both sides.
    """
    u = pred[:, 0:1]
    v = pred[:, 1:2]
    w = pred[:, 2:3]

    du_dx = (u[:, :, 1:, :, :] - u[:, :, :-1, :, :]) / dx
    dv_dy = (v[:, :, :, 1:, :] - v[:, :, :, :-1, :]) / dx
    dw_dz = (w[:, :, :, :, 1:] - w[:, :, :, :, :-1]) / dx

    mask_x = mask_dev[:, :, 1:, :, :] * mask_dev[:, :, :-1, :, :]
    mask_y = mask_dev[:, :, :, 1:, :] * mask_dev[:, :, :, :-1, :]
    mask_z = mask_dev[:, :, :, :, 1:] * mask_dev[:, :, :, :, :-1]

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
    """No-slip boundary condition: velocity must be zero at wall voxels."""
    vel_at_wall = pred[:, :3] * wall_mask_dev.float()
    return torch.mean(vel_at_wall ** 2)


def pressure_stability_loss(pred, mask_dev):
    """Penalise large pressure spatial gradients inside the vessel."""
    p = pred[:, 3:4]
    dp_dx = (p[:, :, 1:, :, :] - p[:, :, :-1, :, :])
    dp_dy = (p[:, :, :, 1:, :] - p[:, :, :, :-1, :])
    dp_dz = (p[:, :, :, :, 1:] - p[:, :, :, :, :-1])
    return (
        (dp_dx ** 2 * mask_dev[:, :, 1:, :, :]).mean() +
        (dp_dy ** 2 * mask_dev[:, :, :, 1:, :]).mean() +
        (dp_dz ** 2 * mask_dev[:, :, :, :, 1:]).mean()
    )


def smoothness_loss(pred, prev_field, mask_dev):
    """Penalise large velocity/pressure changes between consecutive predictions.
    Channel 4 (time index) is excluded — it is a linear ramp, not a fluid var."""
    diff = (pred[:, :4] - prev_field[:, :4]) * mask_dev
    return torch.mean(diff ** 2)

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
GRID_RES   = 64    # HFNO bottleneck runs at 16³, so 64³ input is affordable
MODES      = 8     # modes ≤ GRID_RES//4//2 = 8  →  matches bottleneck size ✓
WIDTH      = 32    # base width; bottleneck uses WIDTH*2 = 64 channels
NUM_LAYERS = 4     # FNO layers at the bottleneck

BATCH_SIZE    = 1          # must be 1 for 8-step rollout on 4 GB VRAM
ROLLOUT_STEPS = 8
EPOCHS        = 10000

# Data optimizer: coarser global-pattern learning, moderate LR
LR_DATA      = 1e-5
# Physics optimizer: fine-grained PDE-constraint sculpting, kept lower
LR_PHYS      = 1e-6
WEIGHT_DECAY = 1e-4
NOISE_STD    = 0.01        # Gaussian noise on input field (regularisation)

PHYS_RAMP_START = 0
PHYS_RAMP_END   = 400
LAMBDA_PHYS_MAX = 0.05
LAMBDA_BC_MAX   = 0.02
PHYS_TARGET     = 0.20     # adaptive lambda: aim to get physics loss below this

# step_weights length is tied to ROLLOUT_STEPS — a mismatch can never occur.
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
#  1. LOAD & VOXELIZE DATA
# ═══════════════════════════════════════════════════════════════════════════
loader = FNODataLoader(DATA_PATH, wall_file_path=WALL_PATH, resolution=GRID_RES)
fields, mask, grid_coords, stats = loader.load()
del loader
gc.collect()
torch.cuda.empty_cache()

# fields      : (T, 5, res, res, res)  — vel(3) + pres + time, standardised
# mask        : (res, res, res)         — binary vessel mask
# grid_coords : (3, res, res, res)      — normalised spatial coordinates [0,1]
fields      = fields.cpu()
mask        = mask.cpu()
grid_coords = grid_coords.cpu()

T = fields.shape[0]
val_split    = int(0.8 * T)
train_fields = fields[:val_split]
val_fields   = fields[val_split:]
print(f"Total timesteps: {T}, training pairs: {val_split - ROLLOUT_STEPS}")


# ═══════════════════════════════════════════════════════════════════════════
#  2. DATASET  — consecutive timestep sequences
# ═══════════════════════════════════════════════════════════════════════════
class TimeStepDataset(Dataset):
    """
    Returns (input_tensor, target_sequence) pairs.

    input_tensor    : (5 + 1 + 3, res, res, res) = field | mask | coords
    target_sequence : (rollout_steps, 5, res, res, res)
    """

    def __init__(self, fields, mask, coords,
                 rollout_steps=ROLLOUT_STEPS, noise_std=0.0):
        self.fields        = fields
        self.mask_ch       = mask.unsqueeze(0)   # (1, res, res, res)
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
            # 1. Gaussian noise on vel + pres channels (not the time channel)
            if self.noise_std > 0:
                noise    = self.noise_std * torch.randn_like(field_in[:4])
                field_in = field_in.clone()
                field_in[:4] = field_in[:4] + noise * self.mask_ch

            # 2. Random temporal reversal (flow is approximately periodic)
            if torch.rand(1).item() < 0.5:
                field_in = field_in.clone()
                field_in[:3] = -field_in[:3]

            # After the reversal block, add scale jitter:
            if self.training_mode and torch.rand(1).item() < 0.4:
                scale = 0.9 + 0.2 * torch.rand(1).item()  # random scale in [0.9, 1.1]
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
    train_dl = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False,
    )
    val_ds = TimeStepDataset(
        val_fields, mask, grid_coords,
        rollout_steps=rollout_steps, noise_std=0.0,
    )
    val_dl = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False,
    )
    return train_ds, train_dl, val_ds, val_dl


# ═══════════════════════════════════════════════════════════════════════════
#  3. MODEL, OPTIMIZERS, SCHEDULER
# ═══════════════════════════════════════════════════════════════════════════
# 5 field channels + 1 mask + 3 coords = 9 input channels
# 5 output channels (vel×3 + pres + time)
in_ch  = 5 + 1 + 3
out_ch = 5

model = HFNO3d(
    modes=MODES,
    width=WIDTH,
    in_channels=in_ch,
    out_channels=out_ch,
    num_fno_layers=NUM_LAYERS,
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"HFNO3d — {n_params:,} parameters")

# Two separate optimisers — data learns coarse flow patterns, phys refines
# against PDE constraints.  Anchor loss (in phys loop) prevents collapse.
optimizer_data = optim.Adam(model.parameters(), lr=LR_DATA, weight_decay=WEIGHT_DECAY)
optimizer_phys = optim.Adam(model.parameters(), lr=LR_PHYS, weight_decay=WEIGHT_DECAY)

scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_data, T_0=300, T_mult=2, eta_min=1e-7)

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

        if "optimizer_data_state_dict" in ckpt and "optimizer_phys_state_dict" in ckpt:
            optimizer_data.load_state_dict(ckpt["optimizer_data_state_dict"])
            optimizer_phys.load_state_dict(ckpt["optimizer_phys_state_dict"])
        elif "optimizer_state_dict" in ckpt:
            optimizer_data.load_state_dict(ckpt["optimizer_state_dict"])

        start_epoch = ckpt["epoch"] + 1

        resume_lr_data = 5e-6
        resume_lr_phys = 1e-6
        for pg in optimizer_data.param_groups:
            pg["lr"] = resume_lr_data
            pg["initial_lr"] = resume_lr_data
        for pg in optimizer_phys.param_groups:
            pg["lr"] = resume_lr_phys
            pg["initial_lr"] = resume_lr_phys

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_data, T_max=3000, eta_min=1e-7,
        )
        print(f"Resuming from epoch {start_epoch}, "
              f"data LR={resume_lr_data:.1e}, phys LR={resume_lr_phys:.1e}")

    except RuntimeError as e:
        # Architecture mismatch (e.g. loading an FNO3d checkpoint into HFNO3d)
        print(f"\n⚠  Checkpoint architecture mismatch — starting fresh.\n"
              f"   Reason: {e}\n"
              f"   Delete {CHECKPOINT_PATH} and {BEST_MODEL_PATH} to silence this.\n")
        start_epoch = 0

    dataset, train_loader, val_dataset, val_loader = build_loaders(ROLLOUT_STEPS)
else:
    dataset, train_loader, val_dataset, val_loader = build_loaders(ROLLOUT_STEPS)
    print("No checkpoint found — starting fresh.")


# ═══════════════════════════════════════════════════════════════════════════
#  5. HELPER METRICS  (mask helpers defined after data loading)
# ═══════════════════════════════════════════════════════════════════════════
mask_dev     = mask.unsqueeze(0).unsqueeze(0).to(device)   # (1,1,res,res,res)
_not_mask    = (~mask.bool()).float().unsqueeze(0).unsqueeze(0).to(device)
_dilated     = F.max_pool3d(_not_mask, kernel_size=3, stride=1, padding=1)
wall_mask_dev = (mask_dev.float() * _dilated).bool()
print(f"Wall voxels: {wall_mask_dev.sum().item()}")


def masked_mse(pred, target):
    """MSE inside the vessel region only."""
    sq = (pred - target) ** 2 * mask_dev
    return sq.sum() / (mask_dev.sum() * pred.shape[1])


@torch.no_grad()
def masked_rel_l2(pred, target):
    """Relative L2 error inside the vessel."""
    d = (pred - target) * mask_dev
    t = target * mask_dev
    return torch.sqrt((d ** 2).sum() / ((t ** 2).sum() + 1e-8))


def build_input(field_t):
    """Construct a single HFNO input tensor from a (5, res, res, res) field."""
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
print("\n--- Starting HFNO3d Training ---")
t_start         = time.time()
t_epoch         = t_start
grid_coords_dev = grid_coords.unsqueeze(0).to(device)  # (1,3,res,res,res)

epoch_phys_avg = 1.0    # initialised high so adaptive lambda starts pushing
lambda_phys    = 0.0
lambda_bc      = 0.0

for epoch in range(start_epoch, EPOCHS):
    model.train()
    dataset.training_mode = True

    epoch_loss     = 0.0
    epoch_mse      = 0.0
    epoch_phys     = 0.0
    epoch_bc       = 0.0
    grad_norm_data = 0.0
    grad_norm_phys = 0.0

    # ── Adaptive lambda scheduling ────────────────────────────────────────
    progress = max(
        0.0,
        min(1.0, (epoch - PHYS_RAMP_START) / (PHYS_RAMP_END - PHYS_RAMP_START))
    )
    if epoch > 0 and epoch % 50 == 0:
        if epoch_phys_avg > PHYS_TARGET * 2:    # physics still bad → push harder
            lambda_phys = min(lambda_phys * 1.10, LAMBDA_PHYS_MAX)
            lambda_bc   = min(lambda_bc   * 1.10, LAMBDA_BC_MAX)
        elif epoch_phys_avg < PHYS_TARGET:      # physics good → ease off
            lambda_phys = max(lambda_phys * 0.95, 0.01)
            lambda_bc   = max(lambda_bc   * 0.95, 0.005)
    else:
        lambda_phys = LAMBDA_PHYS_MAX * progress
        lambda_bc   = LAMBDA_BC_MAX   * progress

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

                mse_unweighted  = masked_mse(pred, tgt_s)
                loss_step       = mse_unweighted * step_weights[s]
                epoch_mse      += mse_unweighted.item()

                if s > 0 and prev_pred_data is not None:
                    loss_step = loss_step + 0.01 * smoothness_loss(
                        pred, prev_pred_data, mask_dev
                    )
                prev_pred_data  = pred
                loss_data_total = loss_data_total + loss_step

                if s < dataset.rollout_steps - 1:
                    next_field   = pred.detach()
                    current_data = torch.cat([
                        next_field,
                        mask_dev.expand(inp.shape[0], -1, -1, -1, -1),
                        grid_coords_dev.expand(inp.shape[0], -1, -1, -1, -1),
                    ], dim=1)

        current_lr = optimizer_data.param_groups[0]["lr"]
        clip_val   = 0.3 if current_lr > 1e-4 else 1.0

        # [FIX] No retain_graph — phys pass does its own fresh forward.
        loss_data_total.backward()
        grad_norm_data = get_grad_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)
        optimizer_data.step()

        # ── Pass 2: Physics loss + anchor ────────────────────────────────
        optimizer_phys.zero_grad()
        loss_phys_total = torch.zeros(1, device=device)
        current_phys    = inp

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for s in range(dataset.rollout_steps):
                pred_phys = model(current_phys)
                tgt_s     = tgts[:, s]

                loss_phys_val  = fd_physics_loss(pred_phys, mask_dev)
                loss_bc_val    = bc_loss(pred_phys, wall_mask_dev)
                loss_pres_stab = pressure_stability_loss(pred_phys, mask_dev)

                if torch.isnan(loss_phys_val):
                    loss_phys_val = torch.zeros(1, device=device)
                if torch.isnan(loss_bc_val):
                    loss_bc_val = torch.zeros(1, device=device)

                # [KEY FIX] Anchor loss: prevents optimizer_phys from satisfying
                # div(u)=0 and no-slip by collapsing everything to zero.
                # Weight=1.0 keeps the physics optimizer tethered to real data.
                anchor_loss = masked_mse(pred_phys, tgt_s)

                loss_phys_total = (
                    loss_phys_total
                    + lambda_phys * loss_phys_val
                    + lambda_bc   * loss_bc_val
                    + 0.05        * loss_pres_stab
                    + 0.1         * anchor_loss
                )
                epoch_phys += loss_phys_val.item()
                epoch_bc   += loss_bc_val.item()

                if s < dataset.rollout_steps - 1:
                    next_field_phys = pred_phys.detach()
                    current_phys    = torch.cat([
                        next_field_phys,
                        mask_dev.expand(inp.shape[0], -1, -1, -1, -1),
                        grid_coords_dev.expand(inp.shape[0], -1, -1, -1, -1),
                    ], dim=1)

        loss_phys_total.backward()
        grad_norm_phys = get_grad_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer_phys.step()

        epoch_loss += (loss_data_total + loss_phys_total).item()

    scheduler.step()
    avg_loss       = epoch_loss  / len(train_loader)
    epoch_phys_avg = epoch_phys  / len(train_loader)

    # ── Light logging every 10 epochs ────────────────────────────────────
    if epoch % 10 == 0 and epoch % 100 != 0:
        lr_now = optimizer_data.param_groups[0]["lr"]
        print(f"  Epoch {epoch} | loss {avg_loss:.4f} | "
              f"phys {epoch_phys_avg:.4f} | lp {lambda_phys:.3f} | LR {lr_now:.1e}")

    # ── Full evaluation every 100 epochs ─────────────────────────────────
    if epoch % 100 == 0:
        model.eval()
        dataset.training_mode = False
        elapsed = time.time() - t_epoch

        # Validation MSE (one-step)
        val_loss = 0.0
        with torch.no_grad():
            for inp_v, tgt_v in val_loader:
                inp_v, tgt_v = inp_v.to(device), tgt_v.to(device)
                pred_v = model(inp_v)
                val_loss += masked_mse(pred_v, tgt_v[:, 0]).item()
        val_loss /= max(1, len(val_loader))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "stats": stats, "mask": mask, "grid_coords": grid_coords,
            }, BEST_MODEL_PATH)
            print(f"  ★ New best model saved (val={val_loss:.5f})")

        # A. One-step relative L2 across all timestep pairs
        total_rel = 0.0
        with torch.no_grad():
            for i in range(T - 1):
                inp_i  = build_input(fields[i]).to(device)
                tgt_i  = fields[i + 1].unsqueeze(0).to(device)
                pred_i = model(inp_i)
                total_rel += masked_rel_l2(pred_i, tgt_i).item()
        acc_1step = (1.0 - total_rel / (T - 1)) * 100.0

        # B. Autoregressive rollout accuracy from t=0
        rollout_eval_steps = min(10, T - 1)
        current = fields[0].unsqueeze(0).to(device)
        with torch.no_grad():
            for s in range(rollout_eval_steps):
                inp_s = torch.cat(
                    [current[0],
                     mask.unsqueeze(0).to(device),
                     grid_coords.to(device)],
                    dim=0,
                ).unsqueeze(0)
                current = model(inp_s)
        tgt_roll   = fields[rollout_eval_steps].unsqueeze(0).to(device)
        rollout_acc = (1.0 - masked_rel_l2(current, tgt_roll).item()) * 100.0

        lr_now = optimizer_data.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:4d} | "
            f"MSE {epoch_mse/len(train_loader):.5f} | "
            f"Phys {epoch_phys_avg:.5f} | "
            f"BC {epoch_bc/len(train_loader):.5f} | "
            f"1-step {acc_1step:.1f}% | "
            f"Rollout-{rollout_eval_steps} {rollout_acc:.1f}% | "
            f"Val {val_loss:.5f} | "
            f"lp {lambda_phys:.3f} | "
            f"LR {lr_now:.1e} | "
            f"{elapsed:.0f}s"
        )
        print(f"  ∇ data {grad_norm_data:.4f} | ∇ phys {grad_norm_phys:.4f}")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
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