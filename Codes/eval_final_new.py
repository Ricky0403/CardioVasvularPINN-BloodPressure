import os
import torch
from model import FNO3d
from fno_data_loader import FNODataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH       = "../VelocityData3D"
WALL_PATH       = "../VelocityData3D/WallMesh/wall.vtp"
BEST_MODEL_PATH = "../Models/fno_best.pth"
FALLBACK_PATH   = "../Models/fno_checkpoint.pth"

MODEL_PATH = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else FALLBACK_PATH
ckpt       = torch.load(MODEL_PATH, map_location=device, weights_only=False)

mask        = ckpt["mask"].to(device)
grid_coords = ckpt["grid_coords"].to(device)
stats       = ckpt["stats"]
out_ch      = ckpt.get("out_channels", 4)   # backward compat

model = FNO3d(
    modes1=8, modes2=8, modes3=8,
    width=32,
    in_channels=9,
    out_channels=out_ch,
    num_layers=4,
).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

mask_dev = mask.unsqueeze(0).unsqueeze(0).to(device)

loader = FNODataLoader(DATA_PATH, wall_file_path=WALL_PATH, resolution=32)
fields, _, _, _ = loader.load()
T = len(fields)


def masked_rel_l2(pred, target):
    d = (pred - target) * mask_dev
    t = target * mask_dev
    return torch.sqrt((d**2).sum() / ((t**2).sum() + 1e-8))


def make_next_input(pred_4ch, t_idx):
    t_val   = float(t_idx) / max(1, T - 1)
    time_ch = torch.full(
        (1, 1, *pred_4ch.shape[2:]), t_val,
        dtype=pred_4ch.dtype, device=pred_4ch.device,
    )
    field_5ch = torch.cat([pred_4ch, time_ch], dim=1)
    return torch.cat([
        field_5ch,
        mask_dev,
        grid_coords.unsqueeze(0),
    ], dim=1)


print(f"\n{'Step':>5} | {'Rel L2':>8} | {'Acc':>8} | {'Vel err':>9} | {'Pres err':>9}")
print("-" * 50)

# Build first input with correct time
f0    = fields[0].clone()
f0[4] = 0.0   # t=0
inp   = torch.cat([f0, mask.unsqueeze(0), grid_coords], dim=0).unsqueeze(0).to(device)

acc_list  = []
pres_list = []
max_err   = 0.0

with torch.no_grad():
    current = model(inp)   # (1, 4, res, res, res)
    tgt     = fields[0, :4].unsqueeze(0).to(device)   # warm-up step

    for s in range(T - 1):
        inp     = make_next_input(current, s + 1)
        current = model(inp)
        tgt     = fields[s + 1, :4].unsqueeze(0).to(device)

        err      = masked_rel_l2(current, tgt).item()
        vel_err  = masked_rel_l2(current[:, :3], tgt[:, :3]).item()
        pres_err = masked_rel_l2(current[:, 3:4], tgt[:, 3:4]).item()

        max_err = max(max_err, err)
        acc_list.append((1 - err) * 100)
        pres_list.append(pres_err)

        print(f"{s+1:>5} | {err:>8.4f} | {(1-err)*100:>7.1f}% | "
              f"{vel_err:>9.4f} | {pres_err:>9.4f}")

stable_steps = sum(1 for a in acc_list if a > 70.0)
print(f"\nStable steps (>70% acc): {stable_steps}/{T-1}")
print(f"Best accuracy:           {max(acc_list):.1f}% at step {acc_list.index(max(acc_list))+1}")
print(f"First below 70%:         step {next((i+1 for i,a in enumerate(acc_list) if a<70), T)}")
print(f"Pressure explosion step: step {next((i+1 for i,p in enumerate(pres_list) if p>1.0), T)}")
