import pyvista as pv
import glob
import os
import torch
import numpy as np
from scipy.spatial import cKDTree


class FNODataLoader:
    """
    Loads unstructured .vtu velocity/pressure data and voxelizes it onto a
    uniform 3D grid for use with the Fourier Neural Operator.

    The FFT requires structured (uniform) grids, so we:
      1. Compute the bounding box of the mesh
      2. Create a uniform (res x res x res) grid
      3. Interpolate data from unstructured mesh to grid via IDW (k=4 neighbors)
      4. Build a binary mask marking which grid points lie inside the vessel
    """

    def __init__(self, folder_path, wall_file_path=None, resolution=32):
        """
        Args:
            folder_path:    Path to the directory containing .vtu files
            wall_file_path: (Optional) Path to wall mesh file for mask refinement
            resolution:     Number of grid points per spatial dimension
        """
        self.files = sorted(glob.glob(os.path.join(folder_path, "*.vtu")))
        self.wall_file_path = wall_file_path
        self.resolution = resolution

        if not self.files:
            raise FileNotFoundError(f"No .vtu files found in {folder_path}")

        print(f"{len(self.files)} files found — voxelizing to {resolution}³ grid")

    def load(self):
        """
        Load all timesteps and voxelize onto a structured grid.

        Returns:
            fields:      (n_timesteps, 4, res, res, res)  velocity(3) + pressure(1), standardized
            mask:        (res, res, res)                  binary mask (1 = inside vessel)
            grid_coords: (3, res, res, res)               normalized [0,1] spatial coordinates
            stats:       dict with per-channel mean/std for denormalization
        """
        res = self.resolution

        # --- 1. Geometry from the first mesh ---
        mesh0 = pv.read(self.files[0])
        points = mesh0.points  # (N, 3)

        # Bounding box with 10% padding so vessel isn't touching the grid edge
        p_min = points.min(axis=0)
        p_max = points.max(axis=0)
        padding = 0.1 * (p_max - p_min)
        p_min -= padding
        p_max += padding

        # --- 2. Uniform grid ---
        x_lin = np.linspace(p_min[0], p_max[0], res)
        y_lin = np.linspace(p_min[1], p_max[1], res)
        z_lin = np.linspace(p_min[2], p_max[2], res)
        Xg, Yg, Zg = np.meshgrid(x_lin, y_lin, z_lin, indexing='ij')
        grid_pts = np.stack([Xg.ravel(), Yg.ravel(), Zg.ravel()], axis=1)

        # --- 3. Build KDTree and compute vessel mask ---
        tree = cKDTree(points)
        dists_1nn, _ = tree.query(grid_pts, k=1)

        # Characteristic spacing: median nearest-neighbor distance in the mesh
        mesh_nn_dists, _ = tree.query(points, k=2)
        char_spacing = np.median(mesh_nn_dists[:, 1])
        mask_threshold = 2.0 * char_spacing

        mask = (dists_1nn < mask_threshold).astype(np.float32)
        mask_3d = mask.reshape(res, res, res)
        mask_bool = mask_3d.astype(bool)

        pct = 100 * mask.mean()
        print(f"Vessel mask: {mask.sum():.0f}/{res**3} points inside ({pct:.1f}%)")
        print(f"Mesh spacing: {char_spacing:.4f}, threshold: {mask_threshold:.4f}")

        # --- 4. IDW interpolation weights (k=4 neighbors) ---
        dists_k, idxs_k = tree.query(grid_pts, k=4)
        weights = 1.0 / (dists_k + 1e-10)
        weights /= weights.sum(axis=1, keepdims=True)

        # --- 5. Load & voxelize every timestep ---
        all_fields = []
        for i, fpath in enumerate(self.files):
            mesh = pv.read(fpath)
            vel  = mesh.point_data["velocity"]                  # (N, 3)
            pres = mesh.point_data["pressure"].reshape(-1, 1)   # (N, 1)
            data = np.hstack([vel, pres])                       # (N, 4)

            # IDW interpolation onto the structured grid
            interp = np.sum(data[idxs_k] * weights[:, :, np.newaxis], axis=1)
            interp *= mask[:, np.newaxis]               # zero outside vessel

            field = interp.reshape(res, res, res, 4).transpose(3, 0, 1, 2)
            all_fields.append(field)

            if (i + 1) % 10 == 0:
                print(f"  Loaded {i + 1}/{len(self.files)} timesteps")

        fields = np.stack(all_fields, axis=0).astype(np.float32)  # (T, 4, res, res, res)

        # Add a time channel: normalized [0, 1] over the cardiac cycle
        T = fields.shape[0]
        time_channel = np.linspace(0, 1, T, dtype=np.float32)  # (T,)
        # Broadcast to (T, 1, res, res, res)
        time_grid = time_channel[:, None, None, None, None] * np.ones((T, 1, res, res, res), dtype=np.float32)
        fields = np.concatenate([fields, time_grid], axis=1)  # now (T, 5, res, res, res)

        # --- 6. Per-channel standardization (zero mean, unit var, inside vessel only) ---
        stats = {}
        for c in range(4):
            channel_vals = []
            for t in range(fields.shape[0]):
                channel_vals.append(fields[t, c][mask_bool])
            channel_vals = np.concatenate(channel_vals)
            mean_c = float(channel_vals.mean())
            std_c  = float(channel_vals.std()) + 1e-8
            stats[f'mean_{c}'] = mean_c
            stats[f'std_{c}']  = std_c

            fields[:, c] = (fields[:, c] - mean_c) / std_c
            fields[:, c] *= mask_3d[np.newaxis]     # re-zero outside

        # --- 7. Normalized coordinate grid [0, 1] ---
        span = p_max - p_min
        grid_coords = np.stack([
            (Xg - p_min[0]) / span[0],
            (Yg - p_min[1]) / span[1],
            (Zg - p_min[2]) / span[2],
        ], axis=0).astype(np.float32)  # (3, res, res, res)

        print(f"Data ready — fields {fields.shape}, mask {mask_3d.shape}")

        return (
            torch.tensor(fields),
            torch.tensor(mask_3d),
            torch.tensor(grid_coords),
            stats,
        )
