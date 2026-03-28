"""
Final evaluation script for the Hierarchical FNO.

Runs a full autoregressive rollout over all timesteps and reports:
  - Per-step relative L2 error and accuracy
  - Separate velocity vs pressure errors
  - Rollout stability statistics

Changelog:
  [FIX] Switched from FNO3d to HFNO3d to match the new training architecture.
  [FIX] Data loaded at the same resolution as training (64) — the previous
        script loaded at resolution=32 while the checkpoint mask was 64³,
        causing a silent field/mask shape mismatch during evaluation.
  [FIX] mask_dev is reconstructed from the checkpoint mask (correct res)
        instead of a separately-loaded mask at a different resolution.
"""

import os

import torch

from model import HFNO3d
from fno_data_loader import FNODataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH       = "../VelocityData3D"
WALL_PATH       = "../VelocityData3D/WallMesh/wall.vtp"
BEST_MODEL_PATH = "../Models/fno_best.pth"
FALLBACK_PATH   = "../Models/fno_checkpoint.pth"

MODEL_PATH = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else FALLBACK_PATH
print(f"Loading model from: {MODEL_PATH}")

ckpt        = torch.load(MODEL_PATH, map_location=device, weights_only=False)
mask        = ckpt["mask"].to(device)         # (res, res, res) — correct resolution
grid_coords = ckpt["grid_coords"].to(device)  # (3, res, res, res)
stats       = ckpt["stats"]

# Infer the resolution from the saved mask
res = mask.shape[0]
print(f"Checkpoint resolution: {res}³")

model = HFNO3d(
    modes=8,
    width=32,
    in_channels=9,
    out_channels=5,
    num_fno_layers=4,
).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

mask_dev = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, res, res, res)

# Load evaluation data at the SAME resolution as the checkpoint
loader = FNODataLoader(DATA_PATH, wall_file_path=WALL_PATH, resolution=res)
fields, _, _, _ = loader.load()
del loader


def masked_rel_l2(pred, target):
    d = (pred - target) * mask_dev
    t = target * mask_dev
    return torch.sqrt((d ** 2).sum() / ((t ** 2).sum() + 1e-8))


# ── Full autoregressive rollout over all timesteps ──────────────────────────
n_steps = len(fields) - 1
print(f"\n{'Step':>5} | {'Rel L2':>8} | {'Acc':>8} | {'Vel err':>9} | {'Pres err':>9}")
print("-" * 50)

current  = fields[0].unsqueeze(0).to(device)
max_err  = 0.0
acc_list  = []
pres_list = []

with torch.no_grad():
    for s in range(n_steps):
        inp = torch.cat([
            current[0],
            mask.unsqueeze(0),
            grid_coords,
        ], dim=0).unsqueeze(0)

        current = model(inp)
        tgt     = fields[s + 1].unsqueeze(0).to(device)

        err      = masked_rel_l2(current, tgt).item()
        vel_err  = masked_rel_l2(current[:, :3], tgt[:, :3]).item()
        pres_err = masked_rel_l2(current[:, 3:4], tgt[:, 3:4]).item()

        max_err = max(max_err, err)
        acc_list.append((1.0 - err) * 100.0)
        pres_list.append(pres_err)

        print(f"{s + 1:>5} | {err:>8.4f} | {acc_list[-1]:>7.1f}% | "
              f"{vel_err:>9.4f} | {pres_err:>9.4f}")

# ── Summary statistics ───────────────────────────────────────────────────────
stable_steps   = sum(1 for a in acc_list if a > 70.0)
best_step      = acc_list.index(max(acc_list)) + 1
first_unstable = next((i + 1 for i, a in enumerate(acc_list) if a < 70.0), n_steps + 1)
pres_explode   = next((i + 1 for i, p in enumerate(pres_list) if p > 1.0), n_steps + 1)

print(f"\nStable steps (>70% acc):  {stable_steps}/{n_steps}")
print(f"Best accuracy:            {max(acc_list):.1f}% at step {best_step}")
print(f"First step below 70%:     step {first_unstable}")
print(f"Pressure explosion step:  step {pres_explode}")
print(f"Mean accuracy:            {sum(acc_list)/len(acc_list):.1f}%")