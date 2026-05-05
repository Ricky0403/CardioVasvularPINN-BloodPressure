import os
import csv
import torch
from model import FNO3d
from fno_data_loader import FNODataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "../VelocityData3D"
WALL_PATH = "../VelocityData3D/WallMesh/wall.vtp"
BEST_MODEL_PATH = "../Models/fno_best.pth"
FALLBACK_MODEL_PATH = "../Models/fno_checkpoint.pth"

# ── Ensure a Results directory exists for the CSV ──
RESULTS_DIR = "../Results"
os.makedirs(RESULTS_DIR, exist_ok=True)
CSV_OUTPUT_PATH = os.path.join(RESULTS_DIR, "fno_eval_final_metrics.csv")

if os.path.exists(BEST_MODEL_PATH):
    MODEL_PATH = BEST_MODEL_PATH
else:
    MODEL_PATH = FALLBACK_MODEL_PATH

ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
mask        = ckpt["mask"].to(device)
grid_coords = ckpt["grid_coords"].to(device)
stats       = ckpt["stats"]

# FNO outputs 4 channels (vel×3 + pressure), no time channel in output
out_ch = ckpt.get("out_channels", 4)

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

T = fields.shape[0]


def masked_rel_l2(pred, target):
    d = (pred.float() - target.float()) * mask_dev.float()
    t = target.float() * mask_dev.float()
    return torch.sqrt((d**2).sum() / ((t**2).sum() + 1e-8))


def pressure_drop(field_4ch):
    """
    ΔP = mean pressure of top-5% voxels (inlet) minus mean of bottom-5% (outlet),
    computed inside the vessel mask. Returns a scalar in standardised units.

    field_4ch: (1, 4, X, Y, Z) — channels are vel×3 + pressure (channel 3)
    """
    p = field_4ch[0, 3]            # (X, Y, Z)
    m = mask_dev[0, 0].bool()      # (X, Y, Z)
    p_vessel = p[m]

    k = max(1, int(0.05 * p_vessel.numel()))
    p_inlet  = p_vessel.topk(k).values.mean()
    p_outlet = p_vessel.topk(k, largest=False).values.mean()
    return (p_inlet - p_outlet).item()


def make_next_input(pred_4ch, t_idx):
    """Build the 9-channel input for the next FNO step."""
    t_val   = float(t_idx) / max(1, T - 1)
    time_ch = torch.full(
        (1, 1, *pred_4ch.shape[2:]),
        t_val, dtype=pred_4ch.dtype, device=pred_4ch.device,
    )
    field_5ch = torch.cat([pred_4ch, time_ch], dim=1)
    return torch.cat([
        field_5ch,
        mask_dev.expand(1, -1, -1, -1, -1),
        grid_coords.unsqueeze(0).expand(1, -1, -1, -1, -1),
    ], dim=1)


def build_input(field_t, t_idx):
    f = field_t.clone()
    f[4] = float(t_idx) / max(1, T - 1)
    
    # FIX: Move each individual piece to 'device' BEFORE concatenating
    return torch.cat([
        f.to(device), 
        mask.unsqueeze(0).to(device), 
        grid_coords.to(device)
    ], dim=0).unsqueeze(0)


# ── Full autoregressive rollout ──────────────────────────────────────────
print(f"\n{'Step':>5} | {'Rel L2':>8} | {'Acc':>8} | {'Vel err':>9} | "
      f"{'Pres err':>9} | {'ΔP pred':>9} | {'ΔP true':>9} | {'ΔP err%':>8}")
print("-" * 96)

current  = build_input(fields[0], 0)
max_err  = 0.0
acc_list = []
pres_list = []
dp_list  = []

# ── Open CSV file and write header ──
with open(CSV_OUTPUT_PATH, mode='w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Step', 'Rel L2', 'Acc', 'Vel err', 'Pres err', 'ΔP (pred)', 'ΔP (true)', 'ΔP err%'])

    with torch.no_grad():
        for s in range(T - 1):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(current)

            # Ground truth has 5 channels (vel+pres+time); compare first 4 only
            tgt      = fields[s + 1, :4].unsqueeze(0).to(device)
            pred_f32 = pred.float()
            tgt_f32  = tgt.float()

            err      = masked_rel_l2(pred_f32, tgt_f32).item()
            vel_err  = masked_rel_l2(pred_f32[:, :3], tgt_f32[:, :3]).item()
            pres_err = masked_rel_l2(pred_f32[:, 3:4], tgt_f32[:, 3:4]).item()

            dp_pred    = pressure_drop(pred_f32)
            dp_true    = pressure_drop(tgt_f32)
            dp_err_pct = abs(dp_pred - dp_true) / (abs(dp_true) + 1e-8) * 100.0

            max_err = max(max_err, err)
            acc_list.append((1 - err) * 100)
            pres_list.append(pres_err)
            dp_list.append((dp_pred, dp_true, dp_err_pct))

            # Terminal print
            print(
                f"{s+1:>5} | {err:>8.4f} | {(1-err)*100:>7.1f}% | "
                f"{vel_err:>9.4f} | {pres_err:>9.4f} | "
                f"{dp_pred:>9.4f} | {dp_true:>9.4f} | {dp_err_pct:>7.1f}%"
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

            # Prepare next input from model prediction
            current = make_next_input(pred_f32, s + 1)

print(f"\nCSV successfully saved to: {CSV_OUTPUT_PATH}")

# ── Summary ─────────────────────────────────────────────────────────────
stable_steps = sum(1 for a in acc_list if a > 70.0)
avg_dp_err   = sum(d[2] for d in dp_list) / len(dp_list)
best_dp_idx  = min(range(len(dp_list)), key=lambda i: dp_list[i][2])

print(f"\nStable steps (>70% acc):    {stable_steps}/{T-1}")
print(f"Best accuracy:              {max(acc_list):.1f}% at step {acc_list.index(max(acc_list))+1}")
print(f"First step below 70%:       step {next((i+1 for i,a in enumerate(acc_list) if a < 70.0), T)}")
print(f"Pressure explosion step:    step {next((i+1 for i,p in enumerate(pres_list) if p > 1.0), T)}")
print(f"Avg ΔP error across steps:  {avg_dp_err:.1f}%")
print(f"Best ΔP error:              {dp_list[best_dp_idx][2]:.1f}% at step {best_dp_idx+1}")