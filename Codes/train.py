import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

# Import your custom modules
from data_loader import DataLoader as PINN_DataLoader
from model import PINNModel as PINN
from physics_loss import get_physics_loss

torch.set_float32_matmul_precision('high')

def calculate_metrics(prediction, target):
    with torch.no_grad():
        # 1. MSE Loss
        mse = torch.mean((prediction - target)**2)
        
        # 2. Relative L2 Accuracy
        error_norm = torch.norm(prediction - target)
        target_norm = torch.norm(target)
        
        # Add small epsilon to avoid division by zero
        relative_error = error_norm / (target_norm + 1e-8)
        accuracy = (1.0 - relative_error.item()) * 100.0
        
    return mse.item(), accuracy

Total_Duration = 1.0
Batch_Size = 10000
Epoches = 5000
LEARNING_RATE = 1e-3       
SAVE_PATH = "../Models/pinn_model_tanh.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_path = r"../VelocityData3D" 
raw_loader = PINN_DataLoader(data_path)
    
num_files = len(raw_loader.files)
dt = Total_Duration / num_files

X, U, P = raw_loader.load(time_step=dt)

X = X.to(device)
U = U.to(device)
P = P.to(device)

dataset = TensorDataset(X, U, P)
train_loader = DataLoader(dataset, batch_size=Batch_Size, shuffle=True)

model = PINN(layers=[4, 64, 64, 64, 64, 64, 64, 64, 4], activation=nn.Tanh()).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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
            
        loss_data = torch.mean((u_pred - u_batch)**2)
            
        loss_physics = get_physics_loss(prediction, x_batch, model.viscosity)
            
        loss = loss_data + loss_physics
            
        loss.backward()
        optimizer.step()
            
        total_loss += loss.item()
        data_loss_accum += loss_data.item()
        phys_loss_accum += loss_physics.item()

    if epoch % 100 == 0:
        elapsed = time.time() - running_time
        
        # Run a full forward pass on ALL data to check accuracy
        with torch.no_grad():
            full_pred = model(X)
            u_full_pred = full_pred[:, 0:3] # Predicted Velocity
            p_full_pred = full_pred[:, 3:4] # Predicted Pressure
            
            # --- A. Training Accuracy (Velocity) ---
            _, train_acc = calculate_metrics(u_full_pred, U)
            
            # --- B. Validation Accuracy (Pressure) ---
            val_loss, val_acc = calculate_metrics(p_full_pred, P)
        
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