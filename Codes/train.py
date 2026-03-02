import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
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
SAVE_PATH = os.path.join(save_dir, "pinn_model_sigmoid.pth")
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_path = r"../VelocityData3D" 
wall_path = os.path.join(data_path, "WallMesh", "wall.vtp")
raw_loader = PINN_DataLoader(data_path,wall_path)
    
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
    'd': (spatial_normalizer.max[4] - spatial_normalizer.min[4]) / 2.0,

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
total_points = len(dataset)

# --- 1. THE STRICT 80/10/10 MATH ---
train_size = int(0.80 * total_points)
val_size = int(0.10 * total_points)
# We calculate test_size by subtracting the other two to guarantee the sum is perfect
test_size = total_points - train_size - val_size 

print(f"Data Split -> Train: {train_size} | Val: {val_size} | Test: {test_size}")

# --- 2. THE 3-WAY RANDOM SPLIT ---
train_dataset, val_dataset, test_dataset = random_split(
    dataset, 
    [train_size, val_size, test_size]
)

train_loader = DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=Batch_Size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=Batch_Size, shuffle=False)

model = PINN(layers=[5, 64, 64, 64, 64, 64, 64, 64, 4], activation=nn.SiLU()).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

ones_wrapper = torch.ones((Batch_Size, 1), device=device, requires_grad=False)

# --- CHECKPOINT SETUP ---
CHECKPOINT_PATH = os.path.join(save_dir, "pinn_checkpoint.pth")
start_epoch = 0

if os.path.exists(CHECKPOINT_PATH):
    print(f"Found checkpoint at {CHECKPOINT_PATH}. Loading...")
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming training from Epoch {start_epoch}")
else:
    print("No checkpoint found. Starting fresh.")

print("Starting Training")
start_time = time.time()
running_time = start_time
for epoch in range(start_epoch, Epoches+1):
    total_loss = 0
    data_loss_accum = 0
    phys_loss_accum = 0

    for batch_idx, (x_batch, u_batch, p_batch) in enumerate(train_loader):
        x_batch = x_batch.clone().detach().requires_grad_(True)

        optimizer.zero_grad()

        prediction = model(x_batch)
        u_pred = prediction[:, 0:3] 

        loss_data = F.mse_loss(u_pred, u_batch)
            
        loss_physics = get_physics_loss(prediction, x_batch, F.softplus(model.viscosity), ones_wrapper, scales=scales)
        
        p_norm = prediction[:, 3:4]         
        p_real = p_norm * scales['p']
        
        P_MAX = 8000.0  
        P_MIN = 0.0   
        VISC_MAX = 0.006 
        VISC_MIN = 0.002
        
        current_visc = F.softplus(model.viscosity)
        
        # Pressure Penalties
        loss_p_upper = torch.mean(F.relu(p_real - P_MAX)**2)
        loss_p_lower = torch.mean(F.relu(P_MIN - p_real)**2)
        
        # Viscosity Penalties
        loss_v_upper = F.relu(current_visc - VISC_MAX)**2
        loss_v_lower = F.relu(VISC_MIN - current_visc)**2
        
        loss = loss_data + loss_physics + loss_p_upper + loss_p_lower + loss_v_upper + loss_v_lower
            
        loss.backward()
        optimizer.step()
            
        total_loss += loss.item()
        data_loss_accum += loss_data.item()
        phys_loss_accum += loss_physics.item()

    if epoch % 100 == 0:
        elapsed = time.time() - running_time
        model.eval()
        
        # Run a full forward pass on ALL data to check accuracy
        with torch.no_grad():
            # 1. Extract the exact Tensors for Train and Val subsets
            X_train_full = dataset.tensors[0][train_dataset.indices]
            U_train_full = dataset.tensors[1][train_dataset.indices]
            
            X_val_full = dataset.tensors[0][val_dataset.indices]
            P_val_full = dataset.tensors[2][val_dataset.indices]
            
            # --- A. Training Accuracy (Velocity on the Train Split) ---
            pred_train = model(X_train_full)
            u_train_pred = pred_train[:, 0:3] 
            _, train_acc = calculate_metrics(u_train_pred, U_train_full)
            
            # --- B. Validation Accuracy (Pressure on the Val Split) ---
            pred_val = model(X_val_full)
            p_val_pred = pred_val[:, 3:4] 
            val_loss, val_acc = calculate_metrics(p_val_pred, P_val_full)
        
        current_mu = F.softplus(model.viscosity).item()
        
        # Averages for printing (Using accumulated values from the batch loop)
        avg_total = total_loss / len(train_loader)
        avg_data = data_loss_accum / len(train_loader)
        avg_phys = phys_loss_accum / len(train_loader)

        print(f"Epoch {epoch} | "
              f"Loss: {avg_total:.4f} | "
              f"Train Acc (U): {train_acc:.2f}% | "
              f"Val Acc (P): {val_acc:.2f}% | "    
              f"Visc: {current_mu:.5f} | "
              f"Time: {elapsed:.1f}s")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, CHECKPOINT_PATH)
        print("Checkpoint saved.")
        # Switch back to training mode
        model.train()
        
        running_time = time.time()

end_time = time.time()
print(f"\nTraining Complete in {(end_time - start_time)/60:.2f} minutes.")
torch.save(model.state_dict(), SAVE_PATH)


model.eval()
test_loss_accum = 0

with torch.no_grad():
    for x_test, u_test, p_test in test_loader:
        test_pred = model(x_test)
        u_test_pred = test_pred[:, 0:3]
        
        # Calculate Final MSE
        t_loss = F.mse_loss(u_test_pred, u_test)
        test_loss_accum += t_loss.item()

final_test_mse = test_loss_accum / len(test_loader)
print(f"FINAL MSE: {final_test_mse:.6f}")