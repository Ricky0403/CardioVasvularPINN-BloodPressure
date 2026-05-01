"""
Training script for the 3-D U-ResNet (Multi-Scale Residual Learning).

Paradigm:
  Maps entire 3-D field at time t → field at t+Δt (operator learning).
  Trained with a combination of:
    1. Data loss (masked MSE on velocity + pressure)
    2. Physics loss (Navier-Stokes residuals: continuity + momentum via FD)
    3. BC loss (no-slip at vessel walls)
    4. Regularizers (pressure smoothness, temporal smoothness)
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import time
import gc

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import UResNet3d
from fno_data_loader import FNODataLoader
from physics_loss import fd_physics_loss, bc_loss, pressure_stability_loss

torch.set_float32_matmul_precision("high")


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
GRID_RES   = 32
BASE_WIDTH = 32        # U-ResNet base channel width (doubles each encoder level)
GROUPS     = 8         # GroupNorm groups

BATCH_SIZE = 1
ROLLOUT_STEPS = 8
EPOCHS = 22000
LR = 1e-3    # AdamW with cosine decay handles this well for U-Net architectures
LR_STEP    = 100
LR_GAMMA   = 0.5
WEIGHT_DECAY = 1e-4
NOISE_STD  = 0.01

# Physics loss ramp-up schedule
PHYS_RAMP_START = 200    # pure data learning first
PHYS_RAMP_END   = 1000   # slow ramp over 800 epochs
LAMBDA_PHYS_MAX = 0.01   # physics is a regularizer, not the primary objective
LAMBDA_BC_MAX   = 0.01   # BC same — gentle nudge only
EARLY_STOP_PATIENCE = 800    # stop if val loss doesn't improve for this many epochs
PRES_SMOOTH_WEIGHT  = 0.001  # was 0.01 — too strong, fighting data loss

# Blood viscosity (kinematic, cm²/s — adjust to match your data units)
VISCOSITY = 0.035

DATA_PATH  = "../VelocityData3D"
WALL_PATH  = "../VelocityData3D/WallMesh/wall.vtp"
SAVE_DIR   = "../Models"

CHECKPOINT_PATH = os.path.join(SAVE_DIR, "uresnet_checkpoint.pth")
SAVE_PATH       = os.path.join(SAVE_DIR, "uresnet_model.pth")
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "uresnet_best.pth")
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

fields      = fields.cpu()
mask        = mask.cpu()
grid_coords = grid_coords.cpu()

T = fields.shape[0]
val_split = int(0.8 * T)
train_fields = fields[:val_split]
val_fields   = fields[val_split:]
print(f"Total timesteps: {T}, training pairs: {val_split - ROLLOUT_STEPS}")


# ═══════════════════════════════════════════════════════════════════════════
#  2. DATASET — consecutive timestep pairs with rollout
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

            # NOTE: temporal reversal (flipping velocity signs) was removed because
            # it creates nonphysical samples — reversing velocity without also reversing
            # time and pressure gradients violates Navier-Stokes.

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
in_ch  = 5 + 1 + 3   # field(5: vel+pres+time) + mask + coords
out_ch = 5

model = UResNet3d(
    in_channels=in_ch,
    out_channels=out_ch,
    base_width=BASE_WIDTH,
    groups=GROUPS,
    use_checkpoint=True,   # needed for 8-step rollout on 4GB,
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"UResNet3d — {n_params:,} parameters")

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=2000, eta_min=1e-6,
)


# ═══════════════════════════════════════════════════════════════════════════
#  4. CHECKPOINT LOADING
# ═══════════════════════════════════════════════════════════════════════════
start_epoch = 0
if os.path.exists(CHECKPOINT_PATH):
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    # Only load model weights — optimizer is new (single instead of dual),
    # so we intentionally skip loading optimizer state.
    start_epoch = ckpt.get("epoch", 0) + 1

    resume_lr = 1e-4  # warm-restart LR — lower than initial but not too low
    for pg in optimizer.param_groups:
        pg["lr"] = resume_lr
        pg["initial_lr"] = resume_lr

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=3000, eta_min=1e-7,
    )
    dataset, train_loader, val_dataset, val_loader = build_loaders(ROLLOUT_STEPS)
    print(f"Warm-restarting with model weights from checkpoint, LR={resume_lr}")
else:
    dataset, train_loader, val_dataset, val_loader = build_loaders(ROLLOUT_STEPS)
    print("No checkpoint found — starting fresh.")


# ═══════════════════════════════════════════════════════════════════════════
#  5. HELPER METRICS & MASKS
# ═══════════════════════════════════════════════════════════════════════════
mask_dev = mask.unsqueeze(0).unsqueeze(0).to(device)   # (1, 1, res, res, res)
from torch.nn.functional import max_pool3d
_not_mask = (~mask.bool()).float().unsqueeze(0).unsqueeze(0).to(device)
_dilated  = max_pool3d(_not_mask, kernel_size=3, stride=1, padding=1)
wall_mask_dev = (mask_dev.float() * _dilated).bool()
print(f"Wall voxels: {wall_mask_dev.sum().item()}")


def masked_mse(pred, target):
    sq = (pred - target) ** 2 * mask_dev
    return sq.sum() / (mask_dev.sum() * pred.shape[1])


@torch.no_grad()
def masked_rel_l2(pred, target):
    d = (pred - target) * mask_dev
    t = target * mask_dev
    return torch.sqrt((d ** 2).sum() / ((t ** 2).sum() + 1e-8))


def build_input(field_t):
    return torch.cat([field_t, mask.unsqueeze(0), grid_coords], dim=0).unsqueeze(0)


def smoothness_loss(pred, prev_field, mask_dev):
    diff = (pred[:, :4] - prev_field[:, :4]) * mask_dev
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
print("\n--- Starting U-ResNet Training ---")
t_start = time.time()
t_epoch = t_start
grid_coords_dev = grid_coords.unsqueeze(0).to(device)
mask_inp_dev = mask_dev.to(device)
best_val_loss = float('inf')
epochs_without_improvement = 0
step_weights = [1.0, 1.0, 1.0, 1.1, 1.1, 1.2, 1.2, 1.3]

for epoch in range(start_epoch, EPOCHS):
    model.train()
    dataset.training_mode = True
    epoch_loss = 0.0
    epoch_mse = 0.0
    epoch_phys = 0.0
    epoch_bc = 0.0
    epoch_grad_norm = 0.0

    # Simple linear ramp — no adaptive scheme (cleaner, easier to debug)
    progress = max(0.0, min(1.0, (epoch - PHYS_RAMP_START) / max(1, PHYS_RAMP_END - PHYS_RAMP_START)))
    lambda_phys = LAMBDA_PHYS_MAX * progress
    lambda_bc = LAMBDA_BC_MAX * progress

    for inp, tgts in train_loader:
        inp  = inp.to(device)
        tgts = tgts.to(device)

        # ── Single combined pass (data + physics + BC) ──
        optimizer.zero_grad()
        loss_total = 0.0
        current = inp
        prev_field = inp[:, :5].float()  # field channels for momentum loss ∂u/∂t
        prev_pred = None

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for s in range(dataset.rollout_steps):
                pred = model(current)
                # Enforce zero outside vessel mask
                pred = pred * mask_dev
                tgt_s = tgts[:, s]

                # --- Data loss ---
                mse_unweighted = masked_mse(pred, tgt_s)
                loss_step = mse_unweighted * step_weights[s]
                epoch_mse += mse_unweighted.item()

                if s > 0 and prev_pred is not None:
                    loss_step = loss_step + 0.01 * smoothness_loss(pred, prev_pred, mask_dev)

                # --- Physics loss (now normalized to O(1)) ---
                loss_ns = fd_physics_loss(
                    pred, prev_field, mask_dev, stats,
                    dt=1.0, dx=1.0, viscosity=VISCOSITY,
                )
                if torch.isnan(loss_ns):
                    loss_ns = torch.tensor(0.0, device=device)
                loss_step = loss_step + lambda_phys * loss_ns
                epoch_phys += loss_ns.item()

                # --- BC loss (no-slip at walls) ---
                loss_bc_val = bc_loss(pred, wall_mask_dev)
                if torch.isnan(loss_bc_val):
                    loss_bc_val = torch.tensor(0.0, device=device)
                loss_step = loss_step + lambda_bc * loss_bc_val
                epoch_bc += loss_bc_val.item()

                # --- Pressure smoothness ---
                loss_step = loss_step + PRES_SMOOTH_WEIGHT * pressure_stability_loss(pred, mask_dev)

                loss_total += loss_step

                # Prepare next rollout step
                prev_pred = pred
                prev_field = pred.detach().float()
                if s < dataset.rollout_steps - 1:
                    current = torch.cat([
                        pred.detach(),
                        mask_dev.expand(inp.shape[0], -1, -1, -1, -1),
                        grid_coords_dev.expand(inp.shape[0], -1, -1, -1, -1),
                    ], dim=1)

        loss_total.backward()
        epoch_grad_norm = get_grad_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        epoch_loss += loss_total.item()

    scheduler.step()
    avg_loss = epoch_loss / len(train_loader)

    # ── Periodic logging ──
    if epoch % 10 == 0 and epoch % 100 != 0:
        lr_now = optimizer.param_groups[0]["lr"]
        phys_avg = epoch_phys / max(1, len(train_loader))
        bc_avg = epoch_bc / max(1, len(train_loader))
        print(f"  Epoch {epoch} | loss {avg_loss:.4f} | phys {phys_avg:.4f} | BC {bc_avg:.4f} | lp {lambda_phys:.3f} | LR {lr_now:.1e}")

    # ── Detailed evaluation every 100 epochs ──
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
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "stats": stats, "mask": mask, "grid_coords": grid_coords,
            }, BEST_MODEL_PATH)
            print(f"  New best model saved (val={val_loss:.5f})")
        else:
            epochs_without_improvement += 100
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch} — no improvement for {EARLY_STOP_PATIENCE} epochs")
                break

        # One-step accuracy
        total_rel = 0.0
        with torch.no_grad():
            for i in range(T - 1):
                inp_i = build_input(fields[i]).to(device)
                tgt_i = fields[i + 1].unsqueeze(0).to(device)
                pred_i = model(inp_i)
                total_rel += masked_rel_l2(pred_i, tgt_i).item()
        avg_rel = total_rel / (T - 1)
        acc_1step = (1.0 - avg_rel) * 100.0

        # Autoregressive rollout
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

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:4d} | "
            f"MSE {epoch_mse/len(train_loader):.5f} | "
            f"Phys {epoch_phys/len(train_loader):.5f} | "
            f"BC {epoch_bc/len(train_loader):.5f} | "
            f"1-step {acc_1step:.1f}% | "
            f"Rollout-{rollout_steps} {rollout_acc:.1f}% | "
            f"Val {val_loss:.5f} | "
            f"lp {lambda_phys:.3f} lb {lambda_bc:.3f} | "
            f"LR {lr_now:.1e} | "
            f"{elapsed:.0f}s"
        )
        print(f"  Grad norm: {epoch_grad_norm:.4f}")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
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