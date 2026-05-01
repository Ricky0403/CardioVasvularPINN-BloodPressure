"""
Training script for the 3-D Fourier Neural Operator (FNO).
FIXED VERSION:
  - Single optimizer (no gradient interference)
  - Physics loss computed in float32 (fixes zero phys loss)
  - Time channel handled correctly in rollout
  - LR schedule fixed for 22000 epochs
  - retain_graph removed
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
    FIXED: Cast to float32 explicitly so bfloat16 underflow doesn't zero this out.
    """
    pred = pred.float()  # FIX: force float32
    mask_f = mask_dev.float()

    u = pred[:, 0:1]
    v = pred[:, 1:2]
    w = pred[:, 2:3]

    du_dx = (u[:, :, 1:, :, :] - u[:, :, :-1, :, :]) / dx
    dv_dy = (v[:, :, :, 1:, :] - v[:, :, :, :-1, :]) / dx
    dw_dz = (w[:, :, :, :, 1:] - w[:, :, :, :, :-1]) / dx

    mask_x = mask_f[:, :, 1:, :, :] * mask_f[:, :, :-1, :, :]
    mask_y = mask_f[:, :, :, 1:, :] * mask_f[:, :, :, :-1, :]
    mask_z = mask_f[:, :, :, :, 1:] * mask_f[:, :, :, :, :-1]

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
    pred = pred.float()  # FIX: force float32
    vel_at_wall = pred[:, :3] * wall_mask_dev.float()
    return torch.mean(vel_at_wall ** 2)


def pressure_stability_loss(pred, mask_dev):
    pred = pred.float()  # FIX: force float32
    p = pred[:, 3:4]
    mask_f = mask_dev.float()
    dp_dx = (p[:, :, 1:, :, :] - p[:, :, :-1, :, :])
    dp_dy = (p[:, :, :, 1:, :] - p[:, :, :, :-1, :])
    dp_dz = (p[:, :, :, :, 1:] - p[:, :, :, :, :-1])
    loss = (
        (dp_dx ** 2 * mask_f[:, :, 1:, :, :]).mean() +
        (dp_dy ** 2 * mask_f[:, :, :, 1:, :]).mean() +
        (dp_dz ** 2 * mask_f[:, :, :, :, 1:]).mean()
    )
    return loss


def smoothness_loss(pred, prev_field, mask_dev):
    pred = pred.float()
    prev_field = prev_field.float()
    diff = (pred[:, :4] - prev_field[:, :4]) * mask_dev.float()
    return torch.mean(diff ** 2)


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
GRID_RES   = 32
MODES      = 8
WIDTH      = 32
NUM_LAYERS = 4

BATCH_SIZE    = 1
ROLLOUT_STEPS = 8
EPOCHS        = 22000
LR            = 3e-4        # FIX: higher initial LR — cosine will decay it properly
LR_MIN        = 1e-6
WEIGHT_DECAY  = 1e-4
NOISE_STD     = 0.01

# Physics loss config — FIX: ramp over first 2000 epochs
PHYS_RAMP_START  = 200
PHYS_RAMP_END    = 2000
LAMBDA_PHYS_MAX  = 0.05
LAMBDA_BC_MAX    = 0.02
LAMBDA_PRES_STAB = 0.02     # FIX: was 0.05, reduced to not dominate

DATA_PATH = "../VelocityData3D"
WALL_PATH = "../VelocityData3D/WallMesh/wall.vtp"
SAVE_DIR  = "../Models"

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

fields      = fields.cpu()
mask        = mask.cpu()
grid_coords = grid_coords.cpu()

T = fields.shape[0]
val_split = int(0.8 * T)
train_fields = fields[:val_split]
val_fields   = fields[val_split:]
print(f"Total timesteps: {T}, training pairs: {val_split - ROLLOUT_STEPS}")


# ═══════════════════════════════════════════════════════════════════════════
#  2. DATASET
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
        field_in = self.fields[idx]
        if self.training_mode:
            if self.noise_std > 0:
                noise = self.noise_std * torch.randn_like(field_in[:4])
                field_in = field_in.clone()
                field_in[:4] = field_in[:4] + noise * self.mask_ch
            if torch.rand(1).item() < 0.3:
                field_in = field_in.clone()
                field_in[:3] = -field_in[:3]

        inp = torch.cat([field_in, self.mask_ch, self.coords], dim=0)
        targets = self.fields[idx + 1 : idx + 1 + self.rollout_steps]
        return inp, targets


def build_loaders(rollout_steps):
    train_ds = TimeStepDataset(
        train_fields, mask, grid_coords,
        rollout_steps=rollout_steps, noise_std=NOISE_STD,
    )
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False)
    val_ds = TimeStepDataset(
        val_fields, mask, grid_coords,
        rollout_steps=rollout_steps, noise_std=0.0,
    )
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False)
    return train_ds, train_dl, val_ds, val_dl


# ═══════════════════════════════════════════════════════════════════════════
#  3. MODEL, OPTIMIZER, SCHEDULER
# ═══════════════════════════════════════════════════════════════════════════
in_ch  = 5 + 1 + 3   # field(5) + mask(1) + coords(3)
out_ch = 5

model = FNO3d(
    modes1=MODES, modes2=MODES, modes3=MODES,
    width=WIDTH,
    in_channels=in_ch,
    out_channels=out_ch,
    num_layers=NUM_LAYERS,
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"FNO3d — {n_params:,} parameters")

# FIX: Single optimizer — eliminates gradient interference
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# FIX: CosineAnnealingLR over full training duration
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=LR_MIN,
)


# ═══════════════════════════════════════════════════════════════════════════
#  4. CHECKPOINT LOADING
# ═══════════════════════════════════════════════════════════════════════════
start_epoch = 0
if os.path.exists(CHECKPOINT_PATH):
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # Load whichever optimizer key exists
    for key in ("optimizer_state_dict", "optimizer_data_state_dict"):
        if key in ckpt:
            try:
                optimizer.load_state_dict(ckpt[key])
            except Exception:
                print(f"  Could not load optimizer state from '{key}', resetting.")
            break

    start_epoch = ckpt.get("epoch", 0) + 1

    # Reset LR to a sane resume value
    resume_lr = 5e-5
    for pg in optimizer.param_groups:
        pg["lr"] = resume_lr
        pg["initial_lr"] = resume_lr

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, EPOCHS - start_epoch), eta_min=LR_MIN,
    )
    print(f"Resuming from epoch {start_epoch}, LR reset to {resume_lr}")
else:
    print("No checkpoint found — starting fresh.")

dataset, train_loader, val_dataset, val_loader = build_loaders(ROLLOUT_STEPS)


# ═══════════════════════════════════════════════════════════════════════════
#  5. HELPER METRICS & MASKS
# ═══════════════════════════════════════════════════════════════════════════
mask_dev = mask.unsqueeze(0).unsqueeze(0).to(device)   # (1,1,res,res,res)
from torch.nn.functional import max_pool3d
_not_mask    = (~mask.bool()).float().unsqueeze(0).unsqueeze(0).to(device)
_dilated     = max_pool3d(_not_mask, kernel_size=3, stride=1, padding=1)
wall_mask_dev = (mask_dev.float() * _dilated).bool()
print(f"Wall voxels: {wall_mask_dev.sum().item()}")

grid_coords_dev = grid_coords.to(device)


def masked_mse(pred, target):
    sq = (pred - target).float() ** 2 * mask_dev.float()
    return sq.sum() / (mask_dev.sum() * pred.shape[1])


@torch.no_grad()
def masked_rel_l2(pred, target):
    d = (pred - target).float() * mask_dev.float()
    t = target.float() * mask_dev.float()
    return torch.sqrt((d ** 2).sum() / ((t ** 2).sum() + 1e-8))


def build_input(field_t):
    return torch.cat([field_t, mask.unsqueeze(0), grid_coords], dim=0).unsqueeze(0)


def get_grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


def make_next_input(pred_field, step_idx, total_steps, batch_size):
    """
    FIX: Replace the time channel in the predicted output with the correct
    ground-truth normalized time before feeding back into the model.
    pred_field: (B, 5, res, res, res) — model output
    Returns:    (B, 9, res, res, res) — next model input
    """
    res = pred_field.shape[-1]
    # Overwrite channel 4 (time) with correct normalized time
    t_val = float(step_idx) / max(1, total_steps - 1)
    pred_corrected = pred_field.clone()
    pred_corrected[:, 4:5] = t_val  # broadcast scalar

    return torch.cat([
        pred_corrected,
        mask_dev.expand(batch_size, -1, -1, -1, -1),
        grid_coords_dev.unsqueeze(0).expand(batch_size, -1, -1, -1, -1),
    ], dim=1)


# ═══════════════════════════════════════════════════════════════════════════
#  6. TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════
print("\n--- Starting FNO Training ---")
t_start  = time.time()
t_epoch  = t_start
best_val_loss = float('inf')

# FIX: uniform step weights early on, increase weight on later steps gradually
step_weights_base = [1.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]

for epoch in range(start_epoch, EPOCHS):
    model.train()
    dataset.training_mode = True

    epoch_loss  = 0.0
    epoch_mse   = 0.0
    epoch_phys  = 0.0
    epoch_bc    = 0.0
    grad_norm_v = 0.0

    # Physics lambda ramp
    if epoch < PHYS_RAMP_START:
        lambda_phys = 0.0
        lambda_bc   = 0.0
    elif epoch < PHYS_RAMP_END:
        progress    = (epoch - PHYS_RAMP_START) / (PHYS_RAMP_END - PHYS_RAMP_START)
        lambda_phys = LAMBDA_PHYS_MAX * progress
        lambda_bc   = LAMBDA_BC_MAX * progress
    else:
        lambda_phys = LAMBDA_PHYS_MAX
        lambda_bc   = LAMBDA_BC_MAX

    for inp, tgts in train_loader:
        inp  = inp.to(device)
        tgts = tgts.to(device)

        optimizer.zero_grad()

        total_loss    = torch.tensor(0.0, device=device)
        current       = inp
        prev_pred     = None
        batch_size    = inp.shape[0]

        # ── Combined forward pass (data + physics in one graph) ────────────
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for s in range(dataset.rollout_steps):
                pred  = model(current)
                tgt_s = tgts[:, s]

                # Data loss (MSE)
                mse_s      = masked_mse(pred, tgt_s)
                w_s        = step_weights_base[s]
                loss_s     = mse_s * w_s
                epoch_mse += mse_s.item()

                # Smoothness regularization
                if prev_pred is not None:
                    loss_s = loss_s + 0.005 * smoothness_loss(pred, prev_pred, mask_dev)
                prev_pred = pred.detach()

                # Physics losses — computed in float32 inside each fn
                phys_val   = fd_physics_loss(pred, mask_dev)
                bc_val     = bc_loss(pred, wall_mask_dev)
                pres_stab  = pressure_stability_loss(pred, mask_dev)

                if not torch.isnan(phys_val):
                    loss_s     = loss_s + lambda_phys * phys_val
                    epoch_phys += phys_val.item()
                if not torch.isnan(bc_val):
                    loss_s     = loss_s + lambda_bc * bc_val
                    epoch_bc   += bc_val.item()
                loss_s = loss_s + LAMBDA_PRES_STAB * pres_stab

                total_loss = total_loss + loss_s

                # FIX: correct time channel in next input
                if s < dataset.rollout_steps - 1:
                    current = make_next_input(
                        pred.detach(), s + 1, T, batch_size
                    )

        # Single backward pass — no retain_graph
        total_loss.backward()
        grad_norm_v = get_grad_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += total_loss.item()

    scheduler.step()

    avg_loss       = epoch_loss  / len(train_loader)
    avg_phys       = epoch_phys  / len(train_loader)
    avg_mse        = epoch_mse   / len(train_loader)

    # ── Periodic logging ─────────────────────────────────────────────────
    if epoch % 10 == 0 and epoch % 100 != 0:
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch} | loss {avg_loss:.4f} | mse {avg_mse:.4f} | "
              f"phys {avg_phys:.5f} | lp {lambda_phys:.4f} | LR {lr_now:.2e}")

    if epoch % 100 == 0:
        model.eval()
        dataset.training_mode = False
        elapsed = time.time() - t_epoch

        # Validation loss
        val_loss = 0.0
        with torch.no_grad():
            for inp_v, tgt_v in val_loader:
                inp_v, tgt_v = inp_v.to(device), tgt_v.to(device)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
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
            print(f"  ✓ New best model saved (val={val_loss:.5f})")

        # One-step accuracy
        total_rel = 0.0
        with torch.no_grad():
            for i in range(T - 1):
                inp_i  = build_input(fields[i]).to(device)
                tgt_i  = fields[i + 1].unsqueeze(0).to(device)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    pred_i = model(inp_i)
                total_rel += masked_rel_l2(pred_i, tgt_i).item()
        acc_1step = (1.0 - total_rel / (T - 1)) * 100.0

        # Autoregressive rollout
        rollout_steps = min(10, T - 1)
        current_r = fields[0].unsqueeze(0).to(device)
        with torch.no_grad():
            for s in range(rollout_steps):
                inp_s = torch.cat([
                    current_r[0], mask.unsqueeze(0).to(device),
                    grid_coords.to(device)
                ], dim=0).unsqueeze(0)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    current_r = model(inp_s)
                # FIX: correct time channel
                current_r[:, 4:5] = float(s + 1) / max(1, T - 1)
        tgt_roll  = fields[rollout_steps].unsqueeze(0).to(device)
        rollout_acc = (1.0 - masked_rel_l2(current_r, tgt_roll).item()) * 100.0

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:5d} | "
            f"MSE {avg_mse:.5f} | "
            f"Phys {avg_phys:.5f} | "
            f"BC {epoch_bc/len(train_loader):.5f} | "
            f"1-step {acc_1step:.1f}% | "
            f"Rollout-{rollout_steps} {rollout_acc:.1f}% | "
            f"Val {val_loss:.5f} | "
            f"lp {lambda_phys:.4f} | "
            f"LR {lr_now:.2e} | "
            f"{elapsed:.0f}s"
        )
        print(f"  Grad norm: {grad_norm_v:.4f}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
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
print(f"Model saved to {SAVE_PATH}")
