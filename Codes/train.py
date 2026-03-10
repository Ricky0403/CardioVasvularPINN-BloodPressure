import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn.functional as F
import time

# Import custom modules
from data_loader import DataLoader as PINN_DataLoader
from model import PINNModel as PINN
from physics_loss import get_physics_loss, get_wss_loss
from normalizer import MinMaxNormalizer

torch.set_float32_matmul_precision('high')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)


def calculate_metrics(prediction, target):
    with torch.no_grad():
        # 1. MSE Loss
        mse = F.mse_loss(prediction, target)
        
        # 2. Relative L2 Accuracy
        error_norm = torch.norm(prediction - target)
        target_norm = torch.norm(target)
        
        # Add small epsilon to avoid division by zero
        relative_error = error_norm / (target_norm + 1e-8)
        accuracy = (1.0 - relative_error.item()) * 100.0
        
    return mse.item(), accuracy

# --- HYPERPARAMETERS ---
EPOCHS = 10000
PRETRAIN_EPOCHS = 3000 
BATCH_SIZE = 15000
LEARNING_RATE = 1e-3

# --- CHECKPOINTING SETUP ---
SAVE_DIR = "../Models"
CHECKPOINT_PATH = os.path.join(SAVE_DIR, "pinn_checkpoint.pth")
FINAL_MODEL_PATH = os.path.join(SAVE_DIR, "pinn_final.pth")
os.makedirs(SAVE_DIR, exist_ok=True)
RESUME_TRAINING = True

# 1. Load Data
loader = PINN_DataLoader(folder_path="../VelocityData3D", wall_file_path="../VelocityData3D/WallMesh/wall.vtp")
coords_t, vel, pres, wss, b_mask = loader.load(time_step=0.2)

# 2. Normalize Data
norm_coords = MinMaxNormalizer(coords_t, method='column-wise', device=DEVICE)
norm_vel = MinMaxNormalizer(vel, method='global', device=DEVICE)
norm_pres = MinMaxNormalizer(pres, method='global', device=DEVICE)
norm_wss = MinMaxNormalizer(wss, method='global', device=DEVICE)

X = norm_coords.encode(coords_t).to(DEVICE)
Y_vel = norm_vel.encode(vel).to(DEVICE)
Y_pres = norm_pres.encode(pres).to(DEVICE)
Y_wss = norm_wss.encode(wss).to(DEVICE)
b_mask = b_mask.to(DEVICE).squeeze()

def get_range(normalizer, dim=None):
    diff = normalizer.max - normalizer.min
    if normalizer.method == 'column-wise':
        return diff[dim].item() / 2.0  # ADDED / 2.0
    return diff.item() / 2.0           # ADDED / 2.0

def get_min(normalizer, dim=None):
    if normalizer.method == 'column-wise':
        return normalizer.min[dim].item()
    return normalizer.min.item()

scales = {
    'x': get_range(norm_coords, 0),
    'y': get_range(norm_coords, 1),
    'z': get_range(norm_coords, 2),
    't': get_range(norm_coords, 3),
    'u': get_range(norm_vel),
    'v': get_range(norm_vel),
    'w': get_range(norm_vel),
    'p': get_range(norm_pres),
    # --- ADD THESE TO PREVENT THE CRASH ---
    'min_u': get_min(norm_vel),
    'min_v': get_min(norm_vel),
    'min_w': get_min(norm_vel)
}

dataset = TensorDataset(X, Y_vel, Y_pres, Y_wss, b_mask)
total_points = len(dataset)

train_size = int(0.80 * total_points)
val_size = int(0.10 * total_points)
test_size = total_points - train_size - val_size 

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
global_ones_tensor = torch.ones((BATCH_SIZE, 1), dtype=torch.float32, device=DEVICE, requires_grad=False)

# 3. Initialize Model & Optimizer
layers = [5, 64, 64, 64, 64, 64, 64, 64, 4]
model = PINN(layers).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

start_epoch = 0

# --- RESUME FROM CHECKPOINT LOGIC ---
if RESUME_TRAINING and os.path.exists(CHECKPOINT_PATH):
    print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resumed training from epoch {start_epoch}")

# --- SET INITIAL CURRICULUM STATE ---
if start_epoch < PRETRAIN_EPOCHS:
    print(f"Starting in Phase 1. Viscosity frozen.")
    model.freeze_viscosity()
else:
    print(f"Starting in Phase 2. Viscosity unfrozen.")
    model.unfreeze_viscosity()

# 4. Training Loop
model.train()
start_time = time.time()

for epoch in range(start_epoch, EPOCHS):
    epoch_loss = 0
    
    # Only listen for the exact transition moment
    if epoch == PRETRAIN_EPOCHS:
        print("Transitioning to Phase 2: Physics-Informed. Viscosity unfrozen.")
        model.unfreeze_viscosity()

    for x_batch, v_batch, p_batch, wss_batch, mask_batch in train_loader:
        optimizer.zero_grad()
        
        if epoch < PRETRAIN_EPOCHS:
            # PHASE 1: Simple unified pass (NO requires_grad overhead here!)
            predictions = model(x_batch)
            pred_vel = predictions[:, 0:3]
            pred_pres = predictions[:, 3:4]
            
            loss_vel = F.mse_loss(pred_vel, v_batch)
            
            # SAFEGUARD: Phase 1 NaN Trap
            loss_pres_boundary = 0.0
            if mask_batch.any(): 
                loss_pres_boundary = F.mse_loss(pred_pres[mask_batch], p_batch[mask_batch])
                
            loss = loss_vel + loss_pres_boundary
            
        else:
            # PHASE 2: True PINN (Graph-safe splitting & Memory Optimization)
            x_boundary = x_batch[mask_batch].clone().detach().requires_grad_(True)
            x_interior = x_batch[~mask_batch].clone().detach().requires_grad_(True)
            
            mu_positive = F.softplus(model.viscosity)
            
            loss_vel = 0.0
            loss_pde = 0.0
            loss_vel_bnd = 0.0
            loss_pres_boundary = 0.0
            loss_wss = 0.0
            loss_p_upper = 0.0
            loss_p_lower = 0.0
            
            # 1. Viscosity Guardrails
            loss_v_upper = torch.mean(F.relu(mu_positive - 0.006)**2) * 1000.0
            loss_v_lower = torch.mean(F.relu(0.002 - mu_positive)**2) * 1000.0
            
            # SAFEGUARD: Interior Physics
            if (~mask_batch).any():
                pred_interior = model(x_interior)
                loss_vel = F.mse_loss(pred_interior[:, 0:3], v_batch[~mask_batch])

                # 2. Pressure Guardrails (Moved safely AFTER pred_interior is defined)
                p_real = norm_pres.decode(pred_interior[:, 3:4])
                loss_p_upper = torch.mean(F.relu(p_real - 8000.0)**2)
                loss_p_lower = torch.mean(F.relu(0.0 - p_real)**2)

                current_size = x_interior.shape[0]
                batch_ones = global_ones_tensor[:current_size]
                loss_pde = get_physics_loss(pred_interior, x_interior, mu_positive, scales, batch_ones)
            
            # SAFEGUARD: Boundary Physics
            if mask_batch.any():
                pred_boundary = model(x_boundary)
                
                loss_vel_bnd = F.mse_loss(pred_boundary[:, 0:3], v_batch[mask_batch])
                loss_pres_boundary = F.mse_loss(pred_boundary[:, 3:4], p_batch[mask_batch])
                
                wss_target_norm = wss_batch[mask_batch]
                wss_target_real = norm_wss.decode(wss_target_norm)
                
                current_b_size = x_boundary.shape[0]
                batch_ones_bnd = global_ones_tensor[:current_b_size]
                
                # PASS IT INTO THE FUNCTION
                loss_wss = get_wss_loss(pred_boundary, x_boundary, wss_target_real, mu_positive, scales, batch_ones_bnd)
            
            # 3. Weighted Total Loss (Stops the viscosity from exploding)
            weight_pde = 0.1
            weight_wss = 0.1
            
            loss = (loss_vel + loss_vel_bnd + loss_pres_boundary + 
                    (weight_pde * loss_pde) + (weight_wss * loss_wss) + 
                    loss_v_upper + loss_v_lower + loss_p_upper + loss_p_lower)
            
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    # Terminal Output
    if epoch % 100 == 0:
        model.eval()
        with torch.no_grad():
            # Quick Training Accuracy (Velocity) on the last seen batch
            eval_preds = model(x_batch)
            eval_vel = eval_preds[:, 0:3]
            train_mse, train_acc = calculate_metrics(eval_vel, v_batch)
            
            # TRUE Validation Accuracy (Looping through unseen Val Set for boundary pressure)
            val_err_sq, val_tgt_sq = 0.0, 0.0
            for x_v, _, p_v, _, mask_v in val_loader:
                if mask_v.any():
                    pred_v = model(x_v[mask_v])
                    p_pred_v = pred_v[:, 3:4]
                    val_err_sq += torch.sum((p_pred_v - p_v[mask_v])**2).item()
                    val_tgt_sq += torch.sum((p_v[mask_v])**2).item()
                    
            val_acc = (1.0 - (val_err_sq**0.5) / (val_tgt_sq**0.5 + 1e-8)) * 100.0 if val_tgt_sq > 0 else 0.0
            
        current_mu = F.softplus(model.viscosity).item()
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch} | "
              f"Loss: {epoch_loss/max(1, len(train_loader)):.5f} | "
              f"Train Acc (U): {train_acc:.2f}% | "
              f"Val Acc (P Bnd): {val_acc:.2f}% | "
              f"Visc: {current_mu:.5f} | "
              f"Time: {elapsed:.1f}s")
        
        # Reset the clock for the next 100 epochs
        start_time = time.time()
        model.train()

    # --- SAVE CHECKPOINT ---
    if epoch > 0 and epoch % 100 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, CHECKPOINT_PATH)

print(f"\nTraining Complete.")
torch.save(model.state_dict(), FINAL_MODEL_PATH)
print(f"Final model saved to {FINAL_MODEL_PATH}")

print("\nRunning Final Test Evaluation on Unseen Data...")
model.eval()
test_loss_accum = 0

with torch.no_grad():
    for x_test, u_test, _, _, _ in test_loader:
        test_pred = model(x_test)
        u_test_pred = test_pred[:, 0:3]
        test_loss_accum += F.mse_loss(u_test_pred, u_test).item()

final_test_mse = test_loss_accum / max(1, len(test_loader))
print(f"FINAL TEST MSE (Velocity): {final_test_mse:.6f}")