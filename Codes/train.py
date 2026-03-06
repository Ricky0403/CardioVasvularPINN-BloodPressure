import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
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
RESUME_TRAINING = False 

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
        return diff[dim].item()
    return diff.item()

scales = {
    'x': get_range(norm_coords, 0),
    'y': get_range(norm_coords, 1),
    'z': get_range(norm_coords, 2),
    't': get_range(norm_coords, 3),
    'u': get_range(norm_vel),
    'v': get_range(norm_vel),
    'w': get_range(norm_vel),
    'p': get_range(norm_pres)
}

dataset = TensorDataset(X, Y_vel, Y_pres, Y_wss, b_mask)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
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
            
            # SAFEGUARD: Interior Physics
            if (~mask_batch).any():
                pred_interior = model(x_interior)
                loss_vel = F.mse_loss(pred_interior[:, 0:3], v_batch[~mask_batch])

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
                
                loss_wss = get_wss_loss(pred_boundary, x_boundary, wss_target_real, mu_positive, scales)
            
            loss = loss_vel + loss_vel_bnd + loss_pde + loss_wss + loss_pres_boundary
            
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    # Terminal Output
    # Terminal Output
    if epoch % 100 == 0:
        model.eval()
        with torch.no_grad():
            # Do a fresh forward pass on the last batch to guarantee variables exist
            eval_preds = model(x_batch)
            eval_vel = eval_preds[:, 0:3]
            eval_pres = eval_preds[:, 3:4]
            
            # Calculate Training Accuracy (Velocity)
            train_mse, train_acc = calculate_metrics(eval_vel, v_batch)
            
            # Calculate Validation Accuracy (Pressure) - With NaN Safeguard
            val_acc = 0.0
            if mask_batch.any():
                val_mse, val_acc = calculate_metrics(eval_pres[mask_batch], p_batch[mask_batch])
            
        current_mu = F.softplus(model.viscosity).item()
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch} | "
              f"Loss: {epoch_loss/len(train_loader):.5f} | "
              f"Train Acc (U): {train_acc:.2f}% | "
              f"Val Acc (P): {val_acc:.2f}% | "
              f"Visc: {current_mu:.5f} | "
              f"Time: {elapsed:.1f}s")
        
        model.train()

    # --- SAVE CHECKPOINT ---
    if epoch > 0 and epoch % 100 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, CHECKPOINT_PATH)
        print(f"--> Checkpoint saved at epoch {epoch}")

# --- SAVE FINAL MODEL ---
end_time = time.time()
print(f"\nTraining Complete in {(end_time - start_time)/60:.2f} minutes.")
torch.save(model.state_dict(), FINAL_MODEL_PATH)
print(f"Final model saved to {FINAL_MODEL_PATH}")