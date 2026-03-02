import pyvista as pv
import glob
import os
import torch
import numpy as np
from scipy.spatial import cKDTree

class DataLoader:
    def __init__(self, folder_path, wall_file_path):
        """
        Lodes all the data into dimension & time & distance, velocity and pressure vectors
        :param folder_path: Path to the .vtu files
        :param wall_file_path: Path to the wall.vtp file
        """
        self.files = sorted(glob.glob(os.path.join(folder_path, "*.vtu")))
        self.wall_file_path = wall_file_path

        if not self.files:
            raise FileNotFoundError(f"No .vtu files found in {folder_path}")

        print(f"{len(self.files)} files found")

    def load(self, time_step):
        # 1. Load the wall geometry and build the high-speed search tree
        if not self.wall_file_path:
            raise ValueError("Please provide the path to the wall.vtp file.")
            
        wall_mesh = pv.read(self.wall_file_path)
        wall_points = wall_mesh.points
        kd_tree = cKDTree(wall_points)

        coordinates_and_time = []
        velocity = []
        pressure = []
        
        for i, file_path in enumerate(self.files):
            time_val = i * time_step
            mesh = pv.read(file_path)
            
            # Extract spatial coordinates
            coords = mesh.points
            
            # 2. Calculate distances from every fluid point to the nearest wall point
            distances, _ = kd_tree.query(coords)
            distances = distances.reshape(-1, 1)  # Reshape to a column vector
            
            vel = mesh.point_data["velocity"]
            pres = mesh.point_data["pressure"]

            time = np.full((coords.shape[0], 1), time_val)

            # 3. Stack x, y, z, t, AND distance
            coords_t_dist = np.hstack((coords, time, distances))

            coordinates_and_time.append(coords_t_dist)
            velocity.append(vel)
            pressure.append(pres)
            
        coordinates_and_time = torch.tensor(np.vstack(coordinates_and_time), dtype=torch.float32)
        velocity = torch.tensor(np.vstack(velocity), dtype=torch.float32)
        pressure = torch.tensor(np.hstack(pressure), dtype=torch.float32).unsqueeze(-1)
        
        return coordinates_and_time, velocity, pressure