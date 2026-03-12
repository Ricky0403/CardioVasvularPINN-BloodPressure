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
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import FNO3d
from fno_data_loader import FNODataLoader

torch.set_float32_matmul_precision("high")


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
GRID_RES   = 32        # Voxel resolution per spatial dimension
MODES      = 8         # Fourier modes to keep per dim (must be ≤ GRID_RES // 2)
WIDTH      = 32        # Hidden channel width  (dv in the paper)
NUM_LAYERS = 4         # Number of Fourier layers  (paper: 4)

BATCH_SIZE = 8
EPOCHS     = 1000
LR         = 1e-3
LR_STEP    = 100       # Halve the LR every LR_STEP epochs (paper: 100)
LR_GAMMA   = 0.5
WEIGHT_DECAY = 1e-4
NOISE_STD  = 0.01      # Gaussian noise injected into inputs (regularization)

DATA_PATH  = "../VelocityData3D"
WALL_PATH  = "../VelocityData3D/WallMesh/wall.vtp"
SAVE_DIR   = "../Models"

CHECKPOINT_PATH = os.path.join(SAVE_DIR, "fno_checkpoint.pth")
SAVE_PATH       = os.path.join(SAVE_DIR, "fno_model.pth")
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ═══════════════════════════════════════════════════════════════════════════
#  1. LOAD & VOXELIZE DATA
# ═══════════════════════════════════════════════════════════════════════════
loader = FNODataLoader(DATA_PATH, wall_file_path=WALL_PATH, resolution=GRID_RES)
fields, mask, grid_coords, stats = loader.load()

# fields      : (T, 4, res, res, res)  — standardized velocity(3) + pressure(1)
# mask        : (res, res, res)         — binary vessel mask
# grid_coords : (3, res, res, res)      — normalised spatial coordinates [0,1]

T = fields.shape[0]
print(f"Total timesteps: {T}, training pairs: {T - 1}")


# ═══════════════════════════════════════════════════════════════════════════
#  2. DATASET  — consecutive timestep pairs
# ═══════════════════════════════════════════════════════════════════════════
class TimeStepDataset(Dataset):
    """
    Each sample is a pair  (field[t], field[t+1]).
    Input channels:  velocity(3) + pressure(1) + mask(1) + coords(3) = 8
    Target channels: velocity(3) + pressure(1) = 4
    """

    def __init__(self, fields, mask, coords, noise_std=0.0):
        self.fields    = fields
        self.mask_ch   = mask.unsqueeze(0)     # (1, res, res, res)
        self.coords    = coords                # (3, res, res, res)
        self.noise_std = noise_std
        self.n_pairs   = fields.shape[0] - 1

    def __len__(self):
        return self.n_pairs

    def __getitem__(self, idx):
        field_in  = self.fields[idx]         # (4, res³)
        field_out = self.fields[idx + 1]     # (4, res³)

        # Noise augmentation (only on the field channels, only inside vessel)
        if self.noise_std > 0 and self.training_mode:
            noise = self.noise_std * torch.randn_like(field_in)
            field_in = field_in + noise * self.mask_ch

        inp = torch.cat([field_in, self.mask_ch, self.coords], dim=0)  # (8, res³)
        return inp, field_out

    # Called from the training loop to toggle noise on/off
    training_mode = True


dataset = TimeStepDataset(fields, mask, grid_coords, noise_std=NOISE_STD)
train_loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True,
)


# ═══════════════════════════════════════════════════════════════════════════
#  3. MODEL, OPTIMIZER, SCHEDULER
# ═══════════════════════════════════════════════════════════════════════════
in_ch  = 4 + 1 + 3   # field + mask + coords
out_ch = 4            # velocity(3) + pressure(1)

model = FNO3d(
    modes1=MODES, modes2=MODES, modes3=MODES,
    width=WIDTH,
    in_channels=in_ch,
    out_channels=out_ch,
    num_layers=NUM_LAYERS,
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"FNO3d — {n_params:,} parameters")

optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma=LR_GAMMA)


# ═══════════════════════════════════════════════════════════════════════════
#  4. CHECKPOINT LOADING
# ═══════════════════════════════════════════════════════════════════════════
start_epoch = 0
if os.path.exists(CHECKPOINT_PATH):
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    print(f"Resuming from epoch {start_epoch}")
else:
    print("No checkpoint found — starting fresh.")


# ═══════════════════════════════════════════════════════════════════════════
#  5. HELPER METRICS
# ═══════════════════════════════════════════════════════════════════════════
mask_dev = mask.unsqueeze(0).unsqueeze(0).to(device)   # (1, 1, res, res, res)


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


# ═══════════════════════════════════════════════════════════════════════════
#  6. TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════
print("\n--- Starting FNO Training ---")
t_start = time.time()
t_epoch = t_start

for epoch in range(start_epoch, EPOCHS):
    model.train()
    dataset.training_mode = True
    epoch_loss = 0.0

    for inp, tgt in train_loader:
        inp, tgt = inp.to(device), tgt.to(device)

        optimizer.zero_grad()
        pred = model(inp)
        loss = masked_mse(pred, tgt)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    scheduler.step()
    avg_loss = epoch_loss / len(train_loader)

    # ── Periodic evaluation ──────────────────────────────────────────────
    if epoch % 100 == 0:
        model.eval()
        dataset.training_mode = False
        elapsed = time.time() - t_epoch

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

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:4d} | Loss {avg_loss:.6f} | "
            f"1-step Acc {acc_1step:.2f}% | "
            f"Rollout-{rollout_steps} Acc {rollout_acc:.2f}% | "
            f"LR {lr_now:.1e} | {elapsed:.1f}s"
        )

        # Save checkpoint
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
