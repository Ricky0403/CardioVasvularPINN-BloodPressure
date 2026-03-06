import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F
import time

# Import custom modules
from data_loader import DataLoader as PINN_DataLoader
from SIREN_model import SIREN_PINN 
from physics_loss import get_physics_loss, get_wss_loss
from normalizer import MinMaxNormalizer

torch.set_float32_matmul_precision('high')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def calculate_metrics(prediction, target):
    with torch.no_grad():
        mse = F.mse_loss(prediction, target)
        error_norm = torch.norm(prediction - target)
        target_norm = torch.norm(target)
        relative_error = error_norm / (target_norm + 1e-8)
        accuracy = (1.0 - relative_error.item()) * 100.0
    return mse.item(), accuracy

# --- HYPERPARAMETERS ---
Total_Duration = 1.0
EPOCHS = 10000
PRETRAIN_EPOCHS = 3000 
BATCH_SIZE = 15000
LEARNING_RATE = 1e-3

SAVE_DIR = "../Models"
CHECKPOINT_PATH = os.path.join(SAVE_DIR, "pinn_checkpoint.pth")
FINAL_MODEL_PATH = os.path.join(SAVE_DIR, "pinn_final.pth")
os.makedirs(SAVE_DIR, exist_ok=True)

print("Extracting coordinates, velocities, WSS, and computing KDTree masks...")
loader = PINN_DataLoader(folder_path="../VelocityData3D", wall_file_path="../VelocityData3D/WallMesh/wall.vtp")
dt = Total_Duration / max(1, len(loader.files))

# Expecting 5 outputs from your updated DataLoader
coords_t, vel, pres, wss, b_mask = loader.load(time_step=dt)

spatial_normalizer = MinMaxNormalizer(coords_t, method='column-wise', device=DEVICE)
velocity_normalizer = MinMaxNormalizer(vel, method='global', device=DEVICE)
pressure_normalizer = MinMaxNormalizer(pres, method='global', device=DEVICE)
wss_normalizer = MinMaxNormalizer(wss, method='global', device=DEVICE)

X_norm = spatial_normalizer.encode(coords_t).to(DEVICE)
U_norm = velocity_normalizer.encode(vel).to(DEVICE)
P_norm = pressure_normalizer.encode(pres).to(DEVICE)
WSS_norm = wss_normalizer.encode(wss).to(DEVICE)

# Ensure mask is boolean and pushed to GPU
B_mask = b_mask.clone().detach().to(dtype=torch.bool, device=DEVICE)

scales = {
    'x': (spatial_normalizer.max[0] - spatial_normalizer.min[0]) / 2.0,
    'y': (spatial_normalizer.max[1] - spatial_normalizer.min[1]) / 2.0,
    'z': (spatial_normalizer.max[2] - spatial_normalizer.min[2]) / 2.0,
    't': (spatial_normalizer.max[3] - spatial_normalizer.min[3]) / 2.0,
    'd': (spatial_normalizer.max[4] - spatial_normalizer.min[4]) / 2.0,
    'u': (velocity_normalizer.max - velocity_normalizer.min) / 2.0,
    'v': (velocity_normalizer.max - velocity_normalizer.min) / 2.0,
    'w': (velocity_normalizer.max - velocity_normalizer.min) / 2.0,
    'p': (pressure_normalizer.max - pressure_normalizer.min) / 2.0
}

for key in scales:
    scales[key] = scales[key].to(DEVICE) if isinstance(scales[key], torch.Tensor) else torch.tensor(scales[key]).to(DEVICE)

dataset = TensorDataset(X_norm, U_norm, P_norm, WSS_norm, B_mask)
total_points = len(dataset)

train_size = int(0.80 * total_points)
val_size = int(0.10 * total_points)
test_size = total_points - train_size - val_size 

print(f"Data Split -> Train: {train_size:,} | Val: {val_size:,} | Test: {test_size:,}")

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = SIREN_PINN(in_features=5, hidden_features=128, out_features=4).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# VRAM OPTIMIZATION: Pre-allocate single gradient tracker
global_ones_tensor = torch.ones((BATCH_SIZE, 1), device=DEVICE, requires_grad=False)

start_epoch = 0
if os.path.exists(CHECKPOINT_PATH):
    print(f"Found checkpoint at {CHECKPOINT_PATH}. Loading...")
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming from Epoch {start_epoch}")
else:
    print("No checkpoint found. Starting fresh.")

# --- CURRICULUM SETUP ---
if start_epoch < PRETRAIN_EPOCHS:
    print("Starting in Phase 1. Viscosity frozen.")
    model.freeze_viscosity()
else:
    print("Starting in Phase 2. Viscosity unfrozen.")
    model.unfreeze_viscosity()

print("Starting Training...")
start_time = time.time()
running_time = start_time

for epoch in range(start_epoch, EPOCHS + 1):
    epoch_loss = 0

    if epoch == PRETRAIN_EPOCHS:
        print("Transitioning to Phase 2: Physics-Informed. Viscosity unfrozen.")
        model.unfreeze_viscosity()

    for x_batch, u_batch, p_batch, wss_batch, mask_batch in train_loader:
        optimizer.zero_grad()
        
        # Flatten mask batch safely for indexing
        mask_batch = mask_batch.squeeze()
        
        if epoch < PRETRAIN_EPOCHS:
            # --- PHASE 1: Geometry Setup ---
            predictions = model(x_batch)
            pred_vel = predictions[:, 0:3]
            pred_pres = predictions[:, 3:4]
            
            loss_vel = F.mse_loss(pred_vel, u_batch)
            
            loss_pres_boundary = 0.0
            if mask_batch.any():
                loss_pres_boundary = F.mse_loss(pred_pres[mask_batch], p_batch[mask_batch])
                
            loss = loss_vel + loss_pres_boundary
            
        else:
            # --- PHASE 2: Navier-Stokes Physics ---
            x_boundary = x_batch[mask_batch].clone().detach().requires_grad_(True)
            x_interior = x_batch[~mask_batch].clone().detach().requires_grad_(True)
            
            mu_positive = F.softplus(model.viscosity)
            
            loss_vel, loss_pde, loss_vel_bnd, loss_pres_boundary, loss_wss = 0.0, 0.0, 0.0, 0.0, 0.0
            
            # Interior Calculus
            if (~mask_batch).any():
                pred_interior = model(x_interior)
                loss_vel = F.mse_loss(pred_interior[:, 0:3], u_batch[~mask_batch])
                
                b_ones_int = global_ones_tensor[:x_interior.shape[0]]
                loss_pde = get_physics_loss(pred_interior, x_interior, mu_positive, b_ones_int, scales)
            
            # Boundary Calculus (WSS)
            if mask_batch.any():
                pred_boundary = model(x_boundary)
                loss_vel_bnd = F.mse_loss(pred_boundary[:, 0:3], u_batch[mask_batch])
                loss_pres_boundary = F.mse_loss(pred_boundary[:, 3:4], p_batch[mask_batch])
                
                wss_target_real = wss_normalizer.decode(wss_batch[mask_batch])
                b_ones_bnd = global_ones_tensor[:x_boundary.shape[0]]
                loss_wss = get_wss_loss(pred_boundary, x_boundary, wss_target_real, mu_positive, scales, b_ones_bnd)
            
            # Penalties
            p_norm = predictions[:, 3:4] if epoch < PRETRAIN_EPOCHS else model(x_batch)[:, 3:4]
            p_real = p_norm * scales['p']
            
            loss_p_upper = torch.mean(F.relu(p_real - 8000.0)**2)
            loss_p_lower = torch.mean(F.relu(0.0 - p_real)**2)
            loss_v_upper = F.relu(mu_positive - 0.006)**2
            loss_v_lower = F.relu(0.002 - mu_positive)**2
            
            loss = loss_vel + loss_vel_bnd + loss_pde + loss_wss + loss_pres_boundary + loss_p_upper + loss_p_lower + loss_v_upper + loss_v_lower
            
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if epoch % 100 == 0:
        elapsed = time.time() - running_time
        model.eval()
        
        with torch.no_grad():
            # Quick batched accuracy for U (Train Set)
            train_err_sq, train_tgt_sq = 0.0, 0.0
            for x_b, u_b, _, _, _ in train_loader:
                pred_b = model(x_b)
                u_pred_b = pred_b[:, 0:3]
                train_err_sq += torch.sum((u_pred_b - u_b)**2).item()
                train_tgt_sq += torch.sum((u_b)**2).item()
            train_acc = (1.0 - (train_err_sq**0.5) / (train_tgt_sq**0.5 + 1e-8)) * 100.0

            # Quick batched accuracy for P at Boundaries (Val Set)
            val_err_sq, val_tgt_sq = 0.0, 0.0
            for x_v, _, p_v, _, mask_v in val_loader:
                mask_v = mask_v.squeeze()
                if mask_v.any():
                    pred_v = model(x_v[mask_v])
                    p_pred_v = pred_v[:, 3:4]
                    val_err_sq += torch.sum((p_pred_v - p_v[mask_v])**2).item()
                    val_tgt_sq += torch.sum((p_v[mask_v])**2).item()
            val_acc = (1.0 - (val_err_sq**0.5) / (val_tgt_sq**0.5 + 1e-8)) * 100.0 if val_tgt_sq > 0 else 0.0

        current_mu = F.softplus(model.viscosity).item()
        avg_total = epoch_loss / max(1, len(train_loader))

        print(f"Epoch {epoch} | Loss: {avg_total:.4f} | Train Acc (U): {train_acc:.2f}% | Val Acc (P Bnd): {val_acc:.2f}% | Visc: {current_mu:.5f} | Time: {elapsed:.1f}s")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, CHECKPOINT_PATH)
        
        model.train()
        running_time = time.time()


end_time = time.time()
print(f"\nTraining Complete in {(end_time - start_time)/60:.2f} minutes.")
torch.save(model.state_dict(), FINAL_MODEL_PATH)

model.eval()
test_loss_accum = 0

with torch.no_grad():
    for x_test, u_test, _, _, _ in test_loader:
        test_pred = model(x_test)
        u_test_pred = test_pred[:, 0:3]
        test_loss_accum += F.mse_loss(u_test_pred, u_test).item()

final_test_mse = test_loss_accum / len(test_loader)
print(f"FINAL TEST MSE: {final_test_mse:.6f}")