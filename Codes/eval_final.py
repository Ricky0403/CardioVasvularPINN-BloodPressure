import os
import csv
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

from data_loader  import DataLoader as PINN_DataLoader
from model        import PINNModel  as PINN
from normalizer   import MinMaxNormalizer

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH        = "../VelocityData3D"
WALL_PATH        = os.path.join(DATA_PATH, "WallMesh", "wall.vtp")
BEST_MODEL_PATH  = "../Models/pinn_final.pth"
FALLBACK_PATH    = "../Models/pinn_checkpoint.pth"
MODEL_PATH       = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else FALLBACK_PATH

RESULTS_DIR      = "../Results"
os.makedirs(RESULTS_DIR, exist_ok=True)

BATCH_SIZE       = 8_000
PHYSICS_SAMPLES  = 5_000
SEED             = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}", flush=True)
print(f"Model  : {MODEL_PATH}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
def rel_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (torch.norm(pred - target) / (torch.norm(target) + 1e-12)).item()


# ─────────────────────────────────────────────────────────────────────────────
# PREDICT ALL POINTS FOR ONE TIMESTEP  (batched, no grad)
# ─────────────────────────────────────────────────────────────────────────────
def predict_timestep(model, x: torch.Tensor) -> torch.Tensor:
    parts = []
    with torch.no_grad():
        for i in range(0, x.shape[0], BATCH_SIZE):
            parts.append(model(x[i : i + BATCH_SIZE]))
    return torch.cat(parts, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS RESIDUALS  (autograd — OUTSIDE torch.no_grad())
# ─────────────────────────────────────────────────────────────────────────────
def compute_physics_residuals(model, x_norm, scales):
    s_u, s_v, s_w, s_p = scales['u'], scales['v'], scales['w'], scales['p']
    s_x, s_y, s_z, s_t = scales['x'], scales['y'], scales['z'], scales['t']
    min_u, min_v, min_w = scales['min_u'], scales['min_v'], scales['min_w']

    x    = x_norm.clone().detach().requires_grad_(True)
    pred = model(x)

    u_n, v_n, w_n, p_n = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3], pred[:, 3:4]
    ones = torch.ones_like(u_n)

    def grad1(y):
        return torch.autograd.grad(y, x, grad_outputs=ones,
                                   create_graph=True, retain_graph=True)[0]

    u_g = grad1(u_n);  v_g = grad1(v_n)
    w_g = grad1(w_n);  p_g = grad1(p_n)

    u_x = u_g[:, 0:1]*(s_u/s_x);  u_y = u_g[:, 1:2]*(s_u/s_y)
    u_z = u_g[:, 2:3]*(s_u/s_z);  u_t = u_g[:, 3:4]*(s_u/s_t)

    v_x = v_g[:, 0:1]*(s_v/s_x);  v_y = v_g[:, 1:2]*(s_v/s_y)
    v_z = v_g[:, 2:3]*(s_v/s_z);  v_t = v_g[:, 3:4]*(s_v/s_t)

    w_x = w_g[:, 0:1]*(s_w/s_x);  w_y = w_g[:, 1:2]*(s_w/s_y)
    w_z = w_g[:, 2:3]*(s_w/s_z);  w_t = w_g[:, 3:4]*(s_w/s_t)

    p_x = p_g[:, 0:1]*(s_p/s_x)
    p_y = p_g[:, 1:2]*(s_p/s_y)
    p_z = p_g[:, 2:3]*(s_p/s_z)

    def grad2_diag(raw_g, col, sn, sd):
        g2 = torch.autograd.grad(
            raw_g[:, col:col+1], x,
            grad_outputs=torch.ones_like(raw_g[:, col:col+1]),
            create_graph=False, retain_graph=True)[0]
        return g2[:, col:col+1] * (sn / sd**2)

    u_xx = grad2_diag(u_g, 0, s_u, s_x);  u_yy = grad2_diag(u_g, 1, s_u, s_y)
    u_zz = grad2_diag(u_g, 2, s_u, s_z)

    v_xx = grad2_diag(v_g, 0, s_v, s_x);  v_yy = grad2_diag(v_g, 1, s_v, s_y)
    v_zz = grad2_diag(v_g, 2, s_v, s_z)

    w_xx = grad2_diag(w_g, 0, s_w, s_x);  w_yy = grad2_diag(w_g, 1, s_w, s_y)
    w_zz = grad2_diag(w_g, 2, s_w, s_z)

    u_real = (u_n + 1.0)*s_u + min_u
    v_real = (v_n + 1.0)*s_v + min_v
    w_real = (w_n + 1.0)*s_w + min_w

    visc = F.softplus(model.viscosity)

    f_u = u_t + (u_real*u_x + v_real*u_y + w_real*u_z) + p_x - visc*(u_xx+u_yy+u_zz)
    f_v = v_t + (u_real*v_x + v_real*v_y + w_real*v_z) + p_y - visc*(v_xx+v_yy+v_zz)
    f_w = w_t + (u_real*w_x + v_real*w_y + w_real*w_z) + p_z - visc*(w_xx+w_yy+w_zz)
    f_c = u_x + v_y + w_z

    return f_u.detach(), f_v.detach(), f_w.detach(), f_c.detach()


# ─────────────────────────────────────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
_BG   = "#0d1117"
_GRID = "#1e2530"
_GOLD = "#FFD700"

def _dark_axes(ax, title):
    ax.set_facecolor(_BG)
    ax.tick_params(colors="#aaaaaa", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.set_title(title, color="#ffffff", fontsize=12, fontweight="bold", pad=12)
    ax.grid(color=_GRID, linewidth=0.7, zorder=1)


def plot_predictive_error(steps, vel_errs, pres_errs, pres_explode, out_path):
    """Graph 1 — Velocity & Pressure Rel-L2 error per timestep."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor(_BG)

    ax.plot(steps, vel_errs,  color="#4C9BE8", lw=2.2, label="Velocity Rel-L2",  zorder=3)
    ax.plot(steps, pres_errs, color="#E84C4C", lw=2.2, label="Pressure Rel-L2",  zorder=3)

    # 70 % accuracy threshold  →  rel-l2 = 0.30
    ax.axhline(0.30, color="#666666", lw=1.0, ls=":", alpha=0.7,
               label="70% accuracy threshold (0.30)", zorder=2)

    # Pressure explosion marker
    if 1 <= pres_explode <= len(steps):
        ex_y = pres_errs[pres_explode - 1]
        ax.scatter([pres_explode], [ex_y], s=130, color=_GOLD,
                   zorder=5, edgecolors=_BG, linewidths=1.5)
        ax.axvline(pres_explode, color=_GOLD, lw=1.0, ls="--", alpha=0.45, zorder=2)
        ax.annotate(
            f"Pressure Explosion\n(Step {pres_explode})",
            xy=(pres_explode, ex_y),
            xytext=(min(pres_explode + 3, len(steps) - 2), ex_y + 0.12),
            arrowprops=dict(arrowstyle="->", color=_GOLD, lw=1.4),
            color=_GOLD, fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#1a1f2b", ec=_GOLD, lw=1.1),
        )

    _dark_axes(ax, "Graph 1 · Predictive Error Curve — Data vs. Model")
    ax.set_xlabel("Rollout Step", color="#cccccc", fontsize=11)
    ax.set_ylabel("Relative L2 Error",  color="#cccccc", fontsize=11)
    ax.set_xlim(1, len(steps));  ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, facecolor="#1a1f2b", edgecolor="#444444",
              labelcolor="#dddddd", loc="upper left")

    fig.tight_layout(pad=1.8)
    fig.savefig(out_path, dpi=150, facecolor=_BG)
    plt.close(fig)
    print(f"  Saved → {out_path}", flush=True)


def plot_physics_compliance(steps, rms_fu, rms_fv, rms_fw, rms_fc,
                             pres_explode, out_path):
    """Graph 2 — Per-step Navier-Stokes residuals on a log scale."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor(_BG)

    for label, data, col in [
        ("f_u  (x-momentum)", rms_fu, "#4C9BE8"),
        ("f_v  (y-momentum)", rms_fv, "#56C596"),
        ("f_w  (z-momentum)", rms_fw, "#C578E8"),
        ("f_c  (continuity)", rms_fc, "#E84C4C"),
    ]:
        ax.plot(steps, data, color=col, lw=1.9, label=label, zorder=3)

    if 1 <= pres_explode <= len(steps):
        ax.axvline(pres_explode, color=_GOLD, lw=1.2, ls="--", alpha=0.55,
                   label=f"Pressure Explosion (step {pres_explode})", zorder=2)

    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(LogLocator(subs="all"))
    ax.grid(which="minor", color=_GRID, linewidth=0.4, zorder=1)

    _dark_axes(ax, "Graph 2 · Physical Compliance Curve — Navier-Stokes Residuals")
    ax.set_xlabel("Rollout Step",                      color="#cccccc", fontsize=11)
    ax.set_ylabel("RMS Physics Residual  (log scale)", color="#cccccc", fontsize=11)
    ax.set_xlim(1, len(steps))
    ax.legend(fontsize=9, facecolor="#1a1f2b", edgecolor="#444444",
              labelcolor="#dddddd", loc="upper left")

    fig.tight_layout(pad=1.8)
    fig.savefig(out_path, dpi=150, facecolor=_BG)
    plt.close(fig)
    print(f"  Saved → {out_path}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def evaluate():
    torch.manual_seed(SEED)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("Loading data …", flush=True)
    loader = PINN_DataLoader(DATA_PATH, WALL_PATH)
    coords_t, vel, pres, wss, b_mask = loader.load(time_step=0.2)

    # ── 2. Normalise ──────────────────────────────────────────────────────────
    norm_coords = MinMaxNormalizer(coords_t, method='column-wise', device=DEVICE)
    norm_vel    = MinMaxNormalizer(vel,      method='global',      device=DEVICE)
    norm_pres   = MinMaxNormalizer(pres,     method='global',      device=DEVICE)

    X_all  = norm_coords.encode(coords_t).to(DEVICE)
    Yv_all = norm_vel.encode(vel).to(DEVICE)
    Yp_all = norm_pres.encode(pres).to(DEVICE)

    def _range(n, dim=None):
        d = n.max - n.min
        return ((d[dim] if n.method == 'column-wise' else d) / 2.0).to(DEVICE)

    def _min(n, dim=None):
        return (n.min[dim] if n.method == 'column-wise' else n.min).to(DEVICE)

    scales = {
        'x': _range(norm_coords, 0), 'y': _range(norm_coords, 1),
        'z': _range(norm_coords, 2), 't': _range(norm_coords, 3),
        'u': _range(norm_vel),       'v': _range(norm_vel),
        'w': _range(norm_vel),       'p': _range(norm_pres),
        'min_u': _min(norm_vel),     'min_v': _min(norm_vel),
        'min_w': _min(norm_vel),
    }

    # ── 3. Group rows by timestep ─────────────────────────────────────────────
    t_values, t_inverse = torch.unique(coords_t[:, 3], sorted=True,
                                        return_inverse=True)
    n_steps    = len(t_values)
    ts_indices = [torch.where(t_inverse == i)[0] for i in range(n_steps)]

    # ── 4. Load model ─────────────────────────────────────────────────────────
    model = PINN(layers=[5, 64, 64, 64, 64, 64, 64, 64, 4],
                 activation=nn.SiLU()).to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No model weights at '{MODEL_PATH}'")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Model loaded  (viscosity = {F.softplus(model.viscosity).item():.6f})\n",
          flush=True)

    # ── 5. Per-timestep predictions ───────────────────────────────────────────
    print(f" {'Step':>4} | {'Rel L2':>8} | {'Acc':>8} | {'Vel err':>9} | {'Pres err':>9}",
          flush=True)
    print("-" * 50, flush=True)

    acc_list    = []
    pres_list   = []
    vel_list    = []
    rel_l2_list = []
    csv_rows    = []

    for s, idx in enumerate(ts_indices):
        x_s  = X_all[idx]
        yv_s = Yv_all[idx]
        yp_s = Yp_all[idx]

        pred      = predict_timestep(model, x_s)
        vp, pp    = pred[:, 0:3], pred[:, 3:4]

        err      = rel_l2(torch.cat([vp, pp], 1), torch.cat([yv_s, yp_s], 1))
        vel_err  = rel_l2(vp, yv_s)
        pres_err = rel_l2(pp, yp_s)
        acc      = (1.0 - err) * 100.0

        acc_list.append(acc);      pres_list.append(pres_err)
        vel_list.append(vel_err);  rel_l2_list.append(err)
        csv_rows.append({"step": s+1, "rel_l2": round(err, 4),
                         "acc_pct": round(acc, 2), "vel_err": round(vel_err, 4),
                         "pres_err": round(pres_err, 4)})

        print(f" {s+1:>4} | {err:>8.4f} | {acc:>7.1f}% | {vel_err:>9.4f} | {pres_err:>9.4f}",
              flush=True)

    # ── 6. Per-step physics residuals (for Graph 2) ───────────────────────────
    print("\nComputing per-step physics residuals for Graph 2 …", flush=True)
    PHYS_PER_STEP = max(200, PHYSICS_SAMPLES // n_steps)
    rms_fu_list, rms_fv_list, rms_fw_list, rms_fc_list = [], [], [], []
    physics_ok = True

    try:
        for s, idx in enumerate(ts_indices):
            n_pts  = min(PHYS_PER_STEP, len(idx))
            perm   = torch.randperm(len(idx))[:n_pts]
            x_p    = X_all[idx[perm]]
            f_u, f_v, f_w, f_c = compute_physics_residuals(model, x_p, scales)
            rms_fu_list.append(f_u.pow(2).mean().sqrt().item())
            rms_fv_list.append(f_v.pow(2).mean().sqrt().item())
            rms_fw_list.append(f_w.pow(2).mean().sqrt().item())
            rms_fc_list.append(f_c.pow(2).mean().sqrt().item())
            csv_rows[s].update({
                "rms_fu": round(rms_fu_list[-1], 6),
                "rms_fv": round(rms_fv_list[-1], 6),
                "rms_fw": round(rms_fw_list[-1], 6),
                "rms_fc": round(rms_fc_list[-1], 6),
            })

        # Overall pooled summary
        perm_all = torch.randperm(X_all.shape[0])[:PHYSICS_SAMPLES]
        f_u, f_v, f_w, f_c = compute_physics_residuals(model, X_all[perm_all], scales)
        mae_fu = f_u.abs().mean().item();  rms_fu = f_u.pow(2).mean().sqrt().item()
        mae_fv = f_v.abs().mean().item();  rms_fv = f_v.pow(2).mean().sqrt().item()
        mae_fw = f_w.abs().mean().item();  rms_fw = f_w.pow(2).mean().sqrt().item()
        mae_fc = f_c.abs().mean().item();  rms_fc = f_c.pow(2).mean().sqrt().item()
        composite_rms = math.sqrt((rms_fu**2 + rms_fv**2 + rms_fw**2 + rms_fc**2) / 4.0)

    except Exception as e:
        print(f"  WARNING: physics residuals failed — {e}", flush=True)
        physics_ok = False

    # ── 7. Summary ────────────────────────────────────────────────────────────
    stable_steps = sum(1 for a in acc_list if a > 70.0)
    best_acc     = max(acc_list)
    best_step    = acc_list.index(best_acc) + 1
    first_bad    = next((i+1 for i, a in enumerate(acc_list) if a < 70.0), n_steps + 1)
    pres_explode = next((i+1 for i, p in enumerate(pres_list) if p > 1.0),  n_steps + 1)

    print(f"\n=== SUMMARY ===", flush=True)
    print(f"Stable steps (>70% acc): {stable_steps}/{n_steps}", flush=True)
    print(f"Best accuracy:           {best_acc:.1f}% at step {best_step}", flush=True)
    print(f"First step below 70%:    step {first_bad}", flush=True)
    print(f"Pressure explosion step: step {pres_explode}", flush=True)

    if physics_ok:
        print(f"\nPhysics residuals  [{PHYSICS_SAMPLES} collocation points]", flush=True)
        print(f"  {'Equation':<20}  {'Mean |Res|':>12}  {'RMS':>12}", flush=True)
        print(f"  {'─'*48}", flush=True)
        for lbl, mae, rms_ in [
            ("f_u  (x-momentum)", mae_fu, rms_fu),
            ("f_v  (y-momentum)", mae_fv, rms_fv),
            ("f_w  (z-momentum)", mae_fw, rms_fw),
            ("f_c  (continuity)", mae_fc, rms_fc),
        ]:
            print(f"  {lbl:<20}  {mae:>12.6f}  {rms_:>12.6f}", flush=True)
        print(f"  {'─'*48}", flush=True)
        print(f"  {'Composite PDE RMS':<20}  {composite_rms:>12.6f}", flush=True)

    mu = F.softplus(model.viscosity).item()
    print(f"\nLearned viscosity : {mu:.6f}  Pa·s", flush=True)
    print(f"(blood plasma ref : 0.0027 – 0.0050 Pa·s)\n", flush=True)

    # ── 8. Save to ../Results ─────────────────────────────────────────────────
    steps = list(range(1, n_steps + 1))
    print(f"Saving results to {RESULTS_DIR} …", flush=True)

    plot_predictive_error(
        steps, vel_list, pres_list, pres_explode,
        os.path.join(RESULTS_DIR, "graph1_predictive_error.png"),
    )

    if physics_ok and rms_fu_list:
        plot_physics_compliance(
            steps, rms_fu_list, rms_fv_list, rms_fw_list, rms_fc_list,
            pres_explode,
            os.path.join(RESULTS_DIR, "graph2_physics_compliance.png"),
        )

    csv_fields = ["step", "rel_l2", "acc_pct", "vel_err", "pres_err"]
    if physics_ok:
        csv_fields += ["rms_fu", "rms_fv", "rms_fw", "rms_fc"]
    csv_path = os.path.join(RESULTS_DIR, "pinn_results_table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader();  w.writerows(csv_rows)
    print(f"  Saved → {csv_path}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    evaluate()