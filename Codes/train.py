import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import time

# Import custom modules
from normalizer import MinMaxNormalizer
from data_loader import DataLoader as PINN_DataLoader
from model import PINNModel as PINN
from physics_loss import get_physics_loss

torch.set_float32_matmul_precision('high')

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

Total_Duration = 1.0
Batch_Size = 15000
Epoches = 5000
LEARNING_RATE = 1e-3       
save_dir = "../Models"
SAVE_PATH = os.path.join(save_dir, "pinn_model_tanh.pth")
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_path = r"../VelocityData3D" 
raw_loader = PINN_DataLoader(data_path)
    
num_files = len(raw_loader.files)
dt = Total_Duration / num_files

X, U, P = raw_loader.load(time_step=dt)

spatial_normalizer = MinMaxNormalizer(X, method='column-wise', device=device)
velocity_normalizer = MinMaxNormalizer(U, method='global', device=device)
pressure_normalizer = MinMaxNormalizer(P, method='global', device=device)

X_norm = spatial_normalizer.encode(X)
U_norm = velocity_normalizer.encode(U)
P_norm = pressure_normalizer.encode(P)

X_norm = X_norm.to(device)
U_norm = U_norm.to(device)
P_norm = P_norm.to(device)

scales = {
    'x': (spatial_normalizer.max[0] - spatial_normalizer.min[0]) / 2.0,
    'y': (spatial_normalizer.max[1] - spatial_normalizer.min[1]) / 2.0,
    'z': (spatial_normalizer.max[2] - spatial_normalizer.min[2]) / 2.0,
    't': (spatial_normalizer.max[3] - spatial_normalizer.min[3]) / 2.0,

    'u': (velocity_normalizer.max - velocity_normalizer.min) / 2.0,
    'v': (velocity_normalizer.max - velocity_normalizer.min) / 2.0,
    'w': (velocity_normalizer.max - velocity_normalizer.min) / 2.0,

    'p': (pressure_normalizer.max - pressure_normalizer.min) / 2.0
}

for key in scales:
    if isinstance(scales[key], torch.Tensor):
        scales[key] = scales[key].to(device)
    else:
        scales[key] = torch.tensor(scales[key]).to(device)

dataset = TensorDataset(X_norm, U_norm, P_norm)
train_loader = DataLoader(dataset, batch_size=Batch_Size, shuffle=True)

model = PINN(layers=[4, 64, 64, 64, 64, 64, 64, 64, 4], activation=nn.SiLU()).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

ones_wrapper = torch.ones((Batch_Size, 1), device=device, requires_grad=False)

print("Starting Training")
start_time = time.time()
running_time = start_time
for epoch in range(Epoches):
    total_loss = 0
    data_loss_accum = 0
    phys_loss_accum = 0

    for batch_idx, (x_batch, u_batch, p_batch) in enumerate(train_loader):
        x_batch = x_batch.clone().detach().requires_grad_(True)

        optimizer.zero_grad()

        prediction = model(x_batch)
        u_pred = prediction[:, 0:3] 

        loss_data = F.mse_loss(u_pred, u_batch)
            
        loss_physics = get_physics_loss(prediction, x_batch, model.viscosity, ones_wrapper, scales=scales)
            
        loss = loss_data*100.0 + loss_physics
            
        loss.backward()
        optimizer.step()
            
        total_loss += loss.item()
        data_loss_accum += loss_data.item()
        phys_loss_accum += loss_physics.item()

    if epoch % 100 == 0:
        elapsed = time.time() - running_time
        
        # Run a full forward pass on ALL data to check accuracy
        with torch.no_grad():
            full_pred = model(X_norm)
            u_full_pred = full_pred[:, 0:3] # Predicted Velocity
            p_full_pred = full_pred[:, 3:4] # Predicted Pressure
            
            # --- A. Training Accuracy (Velocity) ---
            _, train_acc = calculate_metrics(u_full_pred, U_norm)
            
            # --- B. Validation Accuracy (Pressure) ---
            val_loss, val_acc = calculate_metrics(p_full_pred, P_norm)
        
        current_mu = model.viscosity.item()
        
        # Averages for printing
        avg_total = total_loss / len(train_loader)
        avg_data = data_loss_accum / len(train_loader)
        avg_phys = phys_loss_accum / len(train_loader)

        print(f"Epoch {epoch} | "
              f"Loss: {avg_total:.4f} | "
              f"Train Acc (U): {train_acc:.2f}% | "
              f"Val Acc (P): {val_acc:.2f}% | "    
              f"Visc: {current_mu:.5f} | "
              f"Time: {elapsed:.1f}s")
        
        running_time = time.time()

end_time = time.time()
print(f"\nTraining Complete in {(end_time - start_time)/60:.2f} minutes.")
torch.save(model.state_dict(), SAVE_PATH)