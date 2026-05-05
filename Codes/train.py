"""
FNO Training Script v4
Priority: recover 1-step accuracy to 86%+
Key changes:
  - Phase 1 (epochs 0-500):   freeze backbone, train output head only, no BC
  - Phase 2 (epochs 500-2000): unfreeze all, low BC, strong data loss
  - Phase 3 (2000+):          full physics curriculum
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


# ═══════════════════════════════════════════════════════════════════════════
#  LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def fd_physics_loss(pred, mask_dev, dx=1.0):
    pred   = pred.float()
    mask_f = mask_dev.float()
    u, v, w = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

    du_dx = (u[:, :, 1:, :, :] - u[:, :, :-1, :, :]) / dx
    dv_dy = (v[:, :, :, 1:, :] - v[:, :, :, :-1, :]) / dx
    dw_dz = (w[:, :, :, :, 1:] - w[:, :, :, :, :-1]) / dx

    mx = mask_f[:, :, 1:, :, :] * mask_f[:, :, :-1, :, :]
    my = mask_f[:, :, :, 1:, :] * mask_f[:, :, :, :-1, :]
    mz = mask_f[:, :, :, :, 1:] * mask_f[:, :, :, :, :-1]

    sx = min(du_dx.shape[2], dv_dy.shape[2], dw_dz.shape[2])
    sy = min(du_dx.shape[3], dv_dy.shape[3], dw_dz.shape[3])
    sz = min(du_dx.shape[4], dv_dy.shape[4], dw_dz.shape[4])

    div      = (du_dx[:, :, :sx, :sy, :sz] +
                dv_dy[:, :, :sx, :sy, :sz] +
                dw_dz[:, :, :sx, :sy, :sz])
    mask_int = (mx[:, :, :sx, :sy, :sz] *
                my[:, :, :sx, :sy, :sz] *
                mz[:, :, :sx, :sy, :sz])

    n = mask_int.sum().clamp(min=1)
    return (div ** 2 * mask_int).sum() / n


def bc_loss(pred, wall_mask_dev):
    pred = pred.float()
    vel  = pred[:, :3] * wall_mask_dev.float()
    return (vel ** 2).mean()


def pressure_stability_loss(pred, mask_dev):
    pred   = pred.float()
    mask_f = mask_dev.float()
    p      = pred[:, 3:4]
    loss   = (((p[:, :, 1:, :, :] - p[:, :, :-1, :, :]) ** 2) * mask_f[:, :, 1:, :, :]).mean()
    loss  += (((p[:, :, :, 1:, :] - p[:, :, :, :-1, :]) ** 2) * mask_f[:, :, :, 1:, :]).mean()
    loss  += (((p[:, :, :, :, 1:] - p[:, :, :, :, :-1]) ** 2) * mask_f[:, :, :, :, 1:]).mean()
    return loss


def smoothness_loss(pred, prev_field, mask_dev):
    diff = (pred[:, :4].float() - prev_field[:, :4].float()) * mask_dev.float()
    return diff.pow(2).mean()


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
GRID_RES   = 32
MODES      = 8
WIDTH      = 32
NUM_LAYERS = 4

BATCH_SIZE = 1
EPOCHS     = 22000
NOISE_STD  = 0.005   # reduced noise during recovery

PRES_WEIGHT = 1.0   # Phase 1: equal weight
# Will be set to 3.0 in Phase 2+ via get_phase logic below

# ── Training phases ────────────────────────────────────────────────────────
# Phase 1: head-only warmup — fix backbone, train proj1+proj2 aggressively
PHASE1_END      = 500    # epochs of head-only training
PHASE1_LR       = 5e-4   # high LR for fresh output head
PHASE1_ROLLOUT  = 1      # short rollout — focus purely on 1-step quality

# Phase 2: full fine-tune, data-priority, weak physics
PHASE2_END      = 3000
PHASE2_LR       = 1e-4
PHASE2_ROLLOUT  = 4
PHASE2_LAMBDA_PHYS = 0.05
PHASE2_LAMBDA_BC   = 0.005   # very weak BC until 1-step > 82%
PHASE2_LAMBDA_PRES = 0.02

# Phase 3: full training with strong physics
PHASE3_LR          = 5e-5
PHASE3_ROLLOUT     = 8
PHASE3_LAMBDA_PHYS = 0.20
PHASE3_LAMBDA_BC   = 0.08
PHASE3_LAMBDA_PRES = 0.05

LR_MIN       = 1e-7
WEIGHT_DECAY = 1e-4
T_RESTART    = 1500    # warm restart period per phase

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
#  1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
loader = FNODataLoader(DATA_PATH, wall_file_path=WALL_PATH, resolution=GRID_RES)
fields, mask, grid_coords, stats = loader.load()
del loader
import gc; gc.collect()
torch.cuda.empty_cache()

fields      = fields.cpu()
mask        = mask.cpu()
grid_coords = grid_coords.cpu()

T         = fields.shape[0]
val_split = int(0.8 * T)
train_fields = fields[:val_split]
val_fields   = fields[val_split:]
print(f"Total timesteps: {T} | train: {val_split} | val: {T - val_split}")


# ═══════════════════════════════════════════════════════════════════════════
#  2. DATASET
# ═══════════════════════════════════════════════════════════════════════════
class TimeStepDataset(Dataset):
    def __init__(self, fields, mask, coords,
                 rollout_steps=2, noise_std=0.0):
        self.fields        = fields
        self.mask_ch       = mask.unsqueeze(0)
        self.coords        = coords
        self.rollout_steps = rollout_steps
        self.noise_std     = noise_std
        self.n_pairs       = max(1, fields.shape[0] - rollout_steps)
        self.training_mode = True

    def __len__(self):
        return self.n_pairs

    def __getitem__(self, idx):
        field_in = self.fields[idx].clone()

        if self.training_mode:
            if self.noise_std > 0:
                noise = self.noise_std * torch.randn_like(field_in[:4])
                field_in[:4] += noise * self.mask_ch

        inp     = torch.cat([field_in, self.mask_ch, self.coords], dim=0)
        targets = self.fields[idx + 1 : idx + 1 + self.rollout_steps, :4]
        return inp, targets


def build_loaders(rollout_steps):
    train_ds = TimeStepDataset(
        train_fields, mask, grid_coords,
        rollout_steps=rollout_steps, noise_std=NOISE_STD,
    )
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True, pin_memory=False)
    val_ds = TimeStepDataset(
        val_fields, mask, grid_coords,
        rollout_steps=rollout_steps, noise_std=0.0,
    )
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE,
                        shuffle=False, pin_memory=False)
    return train_ds, train_dl, val_ds, val_dl


# ═══════════════════════════════════════════════════════════════════════════
#  3. MODEL  (out_ch=4)
# ═══════════════════════════════════════════════════════════════════════════
in_ch  = 9   # vel(3)+pres(1)+time(1) + mask(1) + coords(3)
out_ch = 4   # vel(3)+pres(1)

model = FNO3d(
    modes1=MODES, modes2=MODES, modes3=MODES,
    width=WIDTH,
    in_channels=in_ch,
    out_channels=out_ch,
    num_layers=NUM_LAYERS,
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"FNO3d — {n_params:,} parameters")


# ═══════════════════════════════════════════════════════════════════════════
#  4. CHECKPOINT — Smart Load (Resume vs. Restart)
# ═══════════════════════════════════════════════════════════════════════════
start_epoch   = 0

if os.path.exists(CHECKPOINT_PATH):
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, weights_only=False, map_location=device)

    # SCENARIO A: Resuming an interrupted training run (Safe Resume)
    if "phase" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_1step  = ckpt.get("best_1step", 0.0) # Recover the best accuracy score
        print(f"  [✓] Resuming existing training from Epoch {start_epoch} (Phase {ckpt['phase']})")

    # SCENARIO B: Loading an old baseline to start Phase 1 from scratch (Transfer Learning)
    else:
        old_sd = ckpt["model_state_dict"]
        new_sd = model.state_dict()
        matched, skipped = {}, []
        
        for k, v in old_sd.items():
            if k in new_sd and new_sd[k].shape == v.shape:
                matched[k] = v
            else:
                skipped.append(k)
        new_sd.update(matched)
        model.load_state_dict(new_sd)
        print(f"  Loaded {len(matched)} tensors | skipped: {skipped}")

        # Wipe the output head for a fresh start
        if hasattr(model, 'reset_output_head'):
            model.reset_output_head()
        else:
            torch.nn.init.xavier_uniform_(model.proj1.weight)
            torch.nn.init.zeros_(model.proj1.bias)
            torch.nn.init.xavier_uniform_(model.proj2.weight)
            torch.nn.init.zeros_(model.proj2.bias)

        start_epoch = 0
        print(f"  [!] Wiped output head. Starting phased retraining from Epoch 0")
else:
    print("No checkpoint — starting fresh.")


# ═══════════════════════════════════════════════════════════════════════════
#  5. PHASE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════
def get_phase(epoch):
    if epoch < PHASE1_END:
        return 1
    elif epoch < PHASE2_END:
        return 2
    else:
        return 3


def setup_phase(phase, model, current_epoch):
    """Configure optimizer, scheduler, dataset for the given phase."""
    global PRES_WEIGHT
    if phase == 1:
        print(f"\n{'='*60}")
        print(f"PHASE 1 — Head-only warmup (epochs 0–{PHASE1_END})")
        print(f"  LR={PHASE1_LR}, rollout={PHASE1_ROLLOUT}, BC=OFF")
        print(f"{'='*60}")
        # Freeze backbone — only train output projection
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("proj")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable params: {trainable:,} (proj1+proj2 only)")

        optimizer  = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=PHASE1_LR, weight_decay=WEIGHT_DECAY,
        )
        scheduler  = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=PHASE1_END, eta_min=LR_MIN,
        )
        ds, dl, vds, vdl = build_loaders(PHASE1_ROLLOUT)
        lambdas = dict(phys=0.0, bc=0.0, pres=0.0, smooth=0.0)
        return optimizer, scheduler, ds, dl, vds, vdl, lambdas

    elif phase == 2:
        print(f"\n{'='*60}")
        print(f"PHASE 2 — Full fine-tune, data priority (epochs {PHASE1_END}–{PHASE2_END})")
        print(f"  LR={PHASE2_LR}, rollout={PHASE2_ROLLOUT}, BC={PHASE2_LAMBDA_BC}")
        print(f"{'='*60}")
        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters())
        print(f"  Trainable params: {trainable:,} (all layers)")

        optimizer  = optim.AdamW(
            model.parameters(), lr=PHASE2_LR, weight_decay=WEIGHT_DECAY,
        )
        scheduler  = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_RESTART, eta_min=LR_MIN,
        )
        ds, dl, vds, vdl = build_loaders(PHASE2_ROLLOUT)
        lambdas = dict(
            phys=PHASE2_LAMBDA_PHYS, bc=PHASE2_LAMBDA_BC,
            pres=PHASE2_LAMBDA_PRES, smooth=0.003,
        )
        PRES_WEIGHT = 3.0
        return optimizer, scheduler, ds, dl, vds, vdl, lambdas

    else:  # phase 3
        print(f"\n{'='*60}")
        print(f"PHASE 3 — Full physics curriculum (epoch {PHASE2_END}+)")
        print(f"  LR={PHASE3_LR}, rollout={PHASE3_ROLLOUT}, BC={PHASE3_LAMBDA_BC}")
        print(f"{'='*60}")
        for param in model.parameters():
            param.requires_grad = True

        optimizer  = optim.AdamW(
            model.parameters(), lr=PHASE3_LR, weight_decay=WEIGHT_DECAY,
        )
        scheduler  = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_RESTART, eta_min=LR_MIN,
        )
        ds, dl, vds, vdl = build_loaders(PHASE3_ROLLOUT)
        lambdas = dict(
            phys=PHASE3_LAMBDA_PHYS, bc=PHASE3_LAMBDA_BC,
            pres=PHASE3_LAMBDA_PRES, smooth=0.005,
        )
        PRES_WEIGHT = 3.0
        return optimizer, scheduler, ds, dl, vds, vdl, lambdas


# ═══════════════════════════════════════════════════════════════════════════
#  6. MASKS & METRICS
# ═══════════════════════════════════════════════════════════════════════════
mask_dev = mask.unsqueeze(0).unsqueeze(0).to(device)

from torch.nn.functional import max_pool3d
_not_mask     = (~mask.bool()).float().unsqueeze(0).unsqueeze(0).to(device)
_dilated      = max_pool3d(_not_mask, kernel_size=3, stride=1, padding=1)
wall_mask_dev = (mask_dev.float() * _dilated).bool()
print(f"Wall voxels: {wall_mask_dev.sum().item()}")

grid_coords_dev = grid_coords.to(device)


def masked_mse(pred, target):
    """MSE weighted by local velocity magnitude — high-flow regions penalised more."""
    sq = (pred.float() - target.float()) ** 2 * mask_dev.float()

    # Weight velocity channels by local speed
    vel_mag   = (target[:, :3].float() ** 2).sum(dim=1, keepdim=True).sqrt()
    vel_weight = (0.5 + vel_mag / (vel_mag.max() + 1e-8)) * mask_dev.float()

    weighted = sq.clone()
    weighted[:, :3] = weighted[:, :3] * vel_weight   # speed-weighted velocity loss
    weighted[:, 3:4] = weighted[:, 3:4] * PRES_WEIGHT  # see Change 11 below

    return weighted.sum() / (mask_dev.sum() * pred.shape[1] + 1e-8)


@torch.no_grad()
def masked_rel_l2(pred, target):
    d = (pred.float() - target.float()) * mask_dev.float()
    t = target.float() * mask_dev.float()
    return torch.sqrt((d ** 2).sum() / ((t ** 2).sum() + 1e-8))


def build_input(field_t, t_idx):
    f      = field_t.clone()
    f[4]   = float(t_idx) / max(1, T - 1)
    return torch.cat([f, mask.unsqueeze(0), grid_coords], dim=0).unsqueeze(0)


def make_next_input(pred_4ch, t_idx, batch_size):
    t_val   = float(t_idx) / max(1, T - 1)
    time_ch = torch.full(
        (batch_size, 1, *pred_4ch.shape[2:]),
        t_val, dtype=pred_4ch.dtype, device=pred_4ch.device,
    )
    field_5ch = torch.cat([pred_4ch, time_ch], dim=1)
    return torch.cat([
        field_5ch,
        mask_dev.expand(batch_size, -1, -1, -1, -1),
        grid_coords_dev.unsqueeze(0).expand(batch_size, -1, -1, -1, -1),
    ], dim=1)


def get_grad_norm(model):
    total = sum(
        p.grad.data.norm(2).item() ** 2
        for p in model.parameters()
        if p.grad is not None
    )
    return total ** 0.5


# ═══════════════════════════════════════════════════════════════════════════
#  7. TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════
print("\n--- Starting Phased FNO Training ---")
t_start       = time.time()
t_epoch       = t_start
best_val_loss = float("inf")
best_1step    = 0.0

# Initialize phase 1
current_phase                                           = 1
optimizer, scheduler, dataset, train_loader, \
    val_dataset, val_loader, lambdas = setup_phase(1, model, 0)

for epoch in range(start_epoch, EPOCHS):

    # ── Phase transitions ────────────────────────────────────────────────
    new_phase = get_phase(epoch)
    if new_phase != current_phase:
        current_phase = new_phase
        optimizer, scheduler, dataset, train_loader, \
            val_dataset, val_loader, lambdas = setup_phase(
                current_phase, model, epoch)

    model.train()
    dataset.training_mode = True

    epoch_loss  = 0.0
    epoch_mse   = 0.0
    epoch_phys  = 0.0
    epoch_bc    = 0.0
    grad_norm_v = 0.0

    for inp, tgts in train_loader:
        inp  = inp.to(device)
        tgts = tgts.to(device)
        B    = inp.shape[0]

        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)
        current    = inp
        prev_pred  = None

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for s in range(dataset.rollout_steps):
                pred  = model(current)
                tgt_s = tgts[:, s]

                # Data loss — pure MSE, no step weighting in phase 1
                mse_s  = masked_mse(pred, tgt_s)
                step_w = 2.0 if s == 0 else 1.0   # front-weight step 1
                loss_s = mse_s * step_w
                epoch_mse += mse_s.item()

                # Smoothness
                if prev_pred is not None and lambdas["smooth"] > 0:
                    loss_s = loss_s + lambdas["smooth"] * smoothness_loss(
                        pred, prev_pred, mask_dev)
                prev_pred = pred.detach()

                # Physics (skipped in phase 1)
                if lambdas["phys"] > 0:
                    phys_v = fd_physics_loss(pred, mask_dev)
                    if not torch.isnan(phys_v):
                        loss_s      = loss_s + lambdas["phys"] * phys_v
                        epoch_phys += phys_v.item()

                if lambdas["bc"] > 0:
                    bc_v = bc_loss(pred, wall_mask_dev)
                    if not torch.isnan(bc_v):
                        loss_s    = loss_s + lambdas["bc"] * bc_v
                        epoch_bc += bc_v.item()

                if lambdas["pres"] > 0:
                    pres_v = pressure_stability_loss(pred, mask_dev)
                    loss_s = loss_s + lambdas["pres"] * pres_v

                total_loss = total_loss + loss_s

                if s < dataset.rollout_steps - 1:
                    current = make_next_input(pred.detach(), s + 1, B)

        total_loss.backward()
        grad_norm_v = get_grad_norm(model)

        # Tighter clipping in phase 1 (head-only, small gradients expected)
        clip = 0.5 if current_phase == 1 else 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
        optimizer.step()

        epoch_loss += total_loss.item()

    scheduler.step()

    avg_loss = epoch_loss / len(train_loader)
    avg_mse  = epoch_mse  / len(train_loader)
    avg_phys = epoch_phys / len(train_loader)
    avg_bc   = epoch_bc   / len(train_loader)

    # ── Logging ──────────────────────────────────────────────────────────
    if epoch % 10 == 0 and epoch % 100 != 0:
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch} [P{current_phase}] | loss {avg_loss:.4f} | "
              f"mse {avg_mse:.4f} | phys {avg_phys:.4f} | bc {avg_bc:.4f} | "
              f"LR {lr_now:.2e}")

    if epoch % 100 == 0:
        model.eval()
        dataset.training_mode = False
        elapsed = time.time() - t_epoch

        # Validation loss
        val_loss = 0.0
        with torch.no_grad():
            for inp_v, tgt_v in val_loader:
                inp_v, tgt_v = inp_v.to(device), tgt_v.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred_v = model(inp_v)
                val_loss += masked_mse(pred_v, tgt_v[:, 0]).item()
        val_loss /= max(1, len(val_loader))

        # One-step accuracy — PRIMARY METRIC
        total_rel = 0.0
        with torch.no_grad():
            for i in range(T - 1):
                inp_i  = build_input(fields[i], i).to(device)
                tgt_i  = fields[i + 1, :4].unsqueeze(0).to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred_i = model(inp_i)
                total_rel += masked_rel_l2(pred_i, tgt_i).item()
        acc_1step = (1.0 - total_rel / (T - 1)) * 100.0

        # Save best by 1-step accuracy (not val loss)
        if acc_1step > best_1step:
            best_1step = acc_1step
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "stats": stats, "mask": mask, "grid_coords": grid_coords,
                "out_channels": out_ch,
                "best_1step": best_1step,
            }, BEST_MODEL_PATH)
            print(f"  ✓ Best 1-step model saved ({acc_1step:.1f}%)")

        # Also track val loss best separately
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Autoregressive rollout
        rollout_n = min(10, T - 1)
        current_r = build_input(fields[0], 0).to(device)
        with torch.no_grad():
            for s in range(rollout_n):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred_r = model(current_r)
                current_r = make_next_input(pred_r, s + 1, 1)
        tgt_roll    = fields[rollout_n, :4].unsqueeze(0).to(device)
        rollout_acc = (1.0 - masked_rel_l2(pred_r, tgt_roll).item()) * 100.0

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:5d} [Phase {current_phase}] | "
            f"MSE {avg_mse:.5f} | Phys {avg_phys:.4f} | BC {avg_bc:.4f} | "
            f"1-step {acc_1step:.1f}% | Rollout-{rollout_n} {rollout_acc:.1f}% | "
            f"Val {val_loss:.5f} | BestAcc {best_1step:.1f}% | "
            f"LR {lr_now:.2e} | {elapsed:.0f}s"
        )
        print(f"  Grad norm: {grad_norm_v:.4f}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": avg_loss,
            "stats": stats, "mask": mask, "grid_coords": grid_coords,
            "out_channels": out_ch,
            "phase": current_phase,
        }, CHECKPOINT_PATH)
        t_epoch = time.time()


# ═══════════════════════════════════════════════════════════════════════════
#  8. FINAL SAVE
# ═══════════════════════════════════════════════════════════════════════════
total_mins = (time.time() - t_start) / 60
print(f"\nTraining complete in {total_mins:.2f} min.")
torch.save({
    "model_state_dict": model.state_dict(),
    "stats": stats, "mask": mask, "grid_coords": grid_coords,
    "out_channels": out_ch,
}, SAVE_PATH)
print(f"Saved → {SAVE_PATH}")
