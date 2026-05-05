import os
import csv
import torch
from model import UResNet3d
from fno_data_loader import FNODataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "../VelocityData3D"
WALL_PATH = "../VelocityData3D/WallMesh/wall.vtp"
BEST_MODEL_PATH = "../Models/uresnet_best.pth"
FALLBACK_MODEL_PATH = "../Models/uresnet_checkpoint.pth"

# ── Ensure a Results directory exists for the CSV ──
RESULTS_DIR = "../Results"
os.makedirs(RESULTS_DIR, exist_ok=True)
CSV_OUTPUT_PATH = os.path.join(RESULTS_DIR, "res-net_eval_final_metrics.csv")

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
new_sd = {k.replace(".conv2.0.", ".conv2."): v for k, v in old_sd.items()}
ckpt["model_state_dict"] = new_sd

model.load_state_dict(ckpt["model_state_dict"])
model.eval()

mask_dev = mask.unsqueeze(0).unsqueeze(0).to(device)

loader = FNODataLoader(DATA_PATH, wall_file_path=WALL_PATH, resolution=32)
fields, _, _, _ = loader.load()


def masked_rel_l2(pred, target):
    d = (pred - target) * mask_dev
    t = target * mask_dev
    return torch.sqrt((d**2).sum() / ((t**2).sum() + 1e-8))


def pressure_drop(field):
    """
    Compute ΔP = P_inlet - P_outlet in standardised units.

    Strategy: inlet = voxels with the highest mean pressure (upstream face),
    outlet = voxels with the lowest mean pressure (downstream face).
    We approximate this by taking the mean pressure of the top-5% and
    bottom-5% pressure voxels inside the vessel mask.

    Returns a scalar tensor.
    """
    # field: (1, 5, X, Y, Z)  — channel 3 is pressure (standardised)
    p = field[0, 3]                          # (X, Y, Z)
    m = mask_dev[0, 0].bool()               # (X, Y, Z)
    p_vessel = p[m]                          # only inside vessel

    k = max(1, int(0.05 * p_vessel.numel()))
    p_inlet  = p_vessel.topk(k).values.mean()
    p_outlet = p_vessel.topk(k, largest=False).values.mean()
    return (p_inlet - p_outlet).item()


# Full autoregressive rollout
header_string = f"\n{'Step':>5} | {'Rel L2':>8} | {'Acc':>8} | {'Vel err':>9} | {'Pres err':>9} | {'ΔP (pred)':>10} | {'ΔP (true)':>10} | {'ΔP err%':>8}"
print(header_string)
print("-" * 88)

current = fields[0].unsqueeze(0).to(device)
max_err = 0.0
acc_list = []
pres_list = []
dp_list = []

# ── Open CSV file and write header ──
with open(CSV_OUTPUT_PATH, mode='w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Step', 'Rel L2', 'Acc', 'Vel err', 'Pres err', 'ΔP (pred)', 'ΔP (true)', 'ΔP err%'])

    with torch.no_grad():
        for s in range(len(fields) - 1):
            inp = torch.cat([
                current[0],
                mask.unsqueeze(0),
                grid_coords,
            ], dim=0).unsqueeze(0)
            current = model(inp)
            tgt = fields[s + 1].unsqueeze(0).to(device)

            err      = masked_rel_l2(current, tgt).item()
            vel_err  = masked_rel_l2(current[:, :3], tgt[:, :3]).item()
            pres_err = masked_rel_l2(current[:, 3:4], tgt[:, 3:4]).item()

            dp_pred = pressure_drop(current)
            dp_true = pressure_drop(tgt)
            dp_err_pct = abs(dp_pred - dp_true) / (abs(dp_true) + 1e-8) * 100.0

            max_err = max(max_err, err)
            acc_list.append((1 - err) * 100)
            pres_list.append(pres_err)
            dp_list.append((dp_pred, dp_true, dp_err_pct))

            # Terminal print
            print(
                f"{s+1:>5} | {err:>8.4f} | {(1-err)*100:>7.1f}% | "
                f"{vel_err:>9.4f} | {pres_err:>9.4f} | "
                f"{dp_pred:>10.4f} | {dp_true:>10.4f} | {dp_err_pct:>7.1f}%"
            )
            
            # CSV row write
            csv_writer.writerow([
                s + 1,
                f"{err:.4f}",
                f"{(1-err)*100:.1f}%",
                f"{vel_err:.4f}",
                f"{pres_err:.4f}",
                f"{dp_pred:.4f}",
                f"{dp_true:.4f}",
                f"{dp_err_pct:.1f}%"
            ])

print(f"\nCSV successfully saved to: {CSV_OUTPUT_PATH}")

# ── Summary ──
stable_steps = sum(1 for a in acc_list if a > 70.0)
avg_dp_err   = sum(d[2] for d in dp_list) / len(dp_list)
best_dp_err  = min(d[2] for d in dp_list)

print(f"\nStable steps (>70% acc):    {stable_steps}/{len(fields)-1}")
print(f"Best accuracy:              {max(acc_list):.1f}% at step {acc_list.index(max(acc_list))+1}")
print(f"First step below 70%:       step {next((i+1 for i,a in enumerate(acc_list) if a < 70.0), len(acc_list)+1)}")
print(f"Pressure explosion step:    step {next((i+1 for i,p in enumerate(pres_list) if p > 1.0), len(pres_list)+1)}")
print(f"Avg ΔP error across steps:  {avg_dp_err:.1f}%")
print(f"Best ΔP error:              {best_dp_err:.1f}% at step {min(range(len(dp_list)), key=lambda i: dp_list[i][2])+1}")