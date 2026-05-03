import os

import torch

from model import UResNet3d
from fno_data_loader import FNODataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "../VelocityData3D"
WALL_PATH = "../VelocityData3D/WallMesh/wall.vtp"
BEST_MODEL_PATH = "../Models/uresnet_best.pth"
FALLBACK_MODEL_PATH = "../Models/uresnet_checkpoint.pth"

if os.path.exists(BEST_MODEL_PATH):
    MODEL_PATH = BEST_MODEL_PATH
else:
    MODEL_PATH = FALLBACK_MODEL_PATH

ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
mask = ckpt["mask"].to(device)
grid_coords = ckpt["grid_coords"].to(device)
stats = ckpt["stats"]

model = UResNet3d(
    in_channels=9,
    out_channels=5,
    base_width=32,
    groups=8,
    use_checkpoint=False,
).to(device)

# ── Fix legacy state dict: conv2.0.* → conv2.* ──
old_sd = ckpt["model_state_dict"]
new_sd = {}
for k, v in old_sd.items():
    new_k = k.replace(".conv2.0.", ".conv2.")
    new_sd[new_k] = v
ckpt["model_state_dict"] = new_sd

model.load_state_dict(ckpt["model_state_dict"])
model.eval()   # Dropout3d is automatically inactive during eval()

mask_dev = mask.unsqueeze(0).unsqueeze(0).to(device)

loader = FNODataLoader(DATA_PATH, wall_file_path=WALL_PATH, resolution=32)
fields, _, _, _ = loader.load()


def masked_rel_l2(pred, target):
    d = (pred - target) * mask_dev
    t = target * mask_dev
    return torch.sqrt((d**2).sum() / ((t**2).sum() + 1e-8))


# Full autoregressive rollout
print(f"\n{'Step':>5} | {'Rel L2':>8} | {'Acc':>8} | {'Vel err':>9} | {'Pres err':>9}")
print("-" * 50)

current = fields[0].unsqueeze(0).to(device)
max_err = 0.0
acc_list = []
pres_list = []
with torch.no_grad():
    for s in range(len(fields) - 1):
        inp = torch.cat([
            current[0],
            mask.unsqueeze(0),
            grid_coords,
        ], dim=0).unsqueeze(0)
        current = model(inp)
        tgt = fields[s + 1].unsqueeze(0).to(device)

        err = masked_rel_l2(current, tgt).item()
        max_err = max(max_err, err)

        vel_err = masked_rel_l2(current[:, :3], tgt[:, :3]).item()
        pres_err = masked_rel_l2(current[:, 3:4], tgt[:, 3:4]).item()
        acc_list.append((1 - err) * 100)
        pres_list.append(pres_err)

        print(f"{s + 1:>5} | {err:>8.4f} | {(1 - err) * 100:>7.1f}% | {vel_err:>9.4f} | {pres_err:>9.4f}")

    stable_steps = sum(1 for a in acc_list if a > 70.0)
    print(f"\nStable steps (>70% acc): {stable_steps}/{len(fields)-1}")
    print(f"Best accuracy:           {max(acc_list):.1f}% at step {acc_list.index(max(acc_list))+1}")
    print(f"First step below 70%:    step {next((i+1 for i,a in enumerate(acc_list) if a < 70.0), len(acc_list)+1)}")
    print(f"Pressure explosion step: step {next((i+1 for i,p in enumerate(pres_list) if p > 1.0), len(pres_list)+1)}")