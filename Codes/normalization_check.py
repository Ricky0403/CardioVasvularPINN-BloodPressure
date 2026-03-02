from data_loader import DataLoader
import os

data_path = r"../VelocityData3D" 
wall_path = os.path.join(data_path, "WallMesh", "wall.vtp")
loader = DataLoader(data_path,wall_path)
coords_time, vel, pres = loader.load(time_step=0.01)

# Run this to check your data range
print(f"X (min, max): {coords_time[:, 0].min().item():.2f}, {coords_time[:, 0].max().item():.2f}")
print(f"Y (min, max): {coords_time[:, 1].min().item():.2f}, {coords_time[:, 1].max().item():.2f}")
print(f"Z (min, max): {coords_time[:, 2].min().item():.2f}, {coords_time[:, 2].max().item():.2f}")
print(f"Distance to Wall (d) (min, max): {coords_time[:, 4].min().item():.2f}, {coords_time[:, 4].max().item():.2f}")

print("-" * 30)
# Velocity Components (U, V, W)
print(f"U (Vel X) (min, max): {vel[:, 0].min().item():.2f}, {vel[:, 0].max().item():.2f}")
print(f"V (Vel Y) (min, max): {vel[:, 1].min().item():.2f}, {vel[:, 1].max().item():.2f}")
print(f"W (Vel Z) (min, max): {vel[:, 2].min().item():.2f}, {vel[:, 2].max().item():.2f}")

print("-" * 30)
# Pressure
print(f"Pressure (min, max): {pres.min().item():.2f}, {pres.max().item():.2f}") 
    