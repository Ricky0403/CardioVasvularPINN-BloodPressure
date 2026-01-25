import pyvista as pv
import glob
import os
import torch
import numpy as np


class DataLoader:
    def __init__(self, folder_path):
        """
        Lodes all the data into dimension & time, velocity and pressure vectors
        :param folder_path:
        """
        self.files = sorted(glob.glob(os.path.join(folder_path, "*.vtu")))

        if not self.files:
            raise FileNotFoundError(f"No .vtu files found in {folder_path}")

        print(f"{len(self.files)} files found")

    def load(self, time_step):

        coordinates_and_time = []
        velocity = []
        pressure = []
        for i, file_path in enumerate(self.files):
            time_val = i*time_step
            mesh = pv.read(file_path)
            coords_t = mesh.points
            vel = mesh.point_data["velocity"]
            pres = mesh.point_data["pressure"]

            time = np.full((coords_t.shape[0], 1), time_val)

            coords_t = np.hstack((coords_t, time))

            coordinates_and_time.append(coords_t)
            velocity.append(vel)
            pressure.append(pres)
        coordinates_and_time = torch.tensor(np.vstack(coordinates_and_time), dtype=torch.float32)
        velocity = torch.tensor(np.vstack(velocity), dtype=torch.float32)
        pressure = torch.tensor(np.hstack(pressure), dtype=torch.float32).unsqueeze(-1)
        return coordinates_and_time, velocity, pressure