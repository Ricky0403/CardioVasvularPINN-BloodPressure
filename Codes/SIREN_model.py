import torch
import torch.nn as nn
import numpy as np

class FourierFeatureMapping(nn.Module):
    def __init__(self, in_features, mapping_size, scale=10.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_features, mapping_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = 2.0 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, w0=30.0, is_first=False, winner_noise_std=0.0):
        super().__init__()
        self.w0 = w0
        self.linear = nn.Linear(in_features, out_features)
        
        with torch.no_grad():
            if is_first:
                bound = 1 / in_features
            else:
                bound = np.sqrt(6 / in_features) / w0
            self.linear.weight.uniform_(-bound, bound)
            
            if winner_noise_std > 0.0:
                noise = torch.randn_like(self.linear.weight) * winner_noise_std
                self.linear.weight.add_(noise)

    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))
    
class PirateResidualBlock(nn.Module):
    def __init__(self, hidden_features, w0=30.0):
        super().__init__()
        self.layer = SirenLayer(hidden_features, hidden_features, w0=w0)
        self.alpha = nn.Parameter(torch.tensor(0.01)) 

    def forward(self, x):
        return self.alpha * self.layer(x) + x


class SIREN_PINN(nn.Module):
    def __init__(self, in_features=5, hidden_features=128, out_features=4): 
        super().__init__()
        
        # Trainable Viscosity Parameter
        self.viscosity = nn.Parameter(torch.tensor([-5.65], dtype=torch.float32))
        
        self.fourier = FourierFeatureMapping(in_features, mapping_size=128)
        fourier_out_dim = 128 * 2
        
        self.macro_branch = nn.Sequential(
            SirenLayer(fourier_out_dim, hidden_features, w0=10.0, is_first=True, winner_noise_std=0.01),
            PirateResidualBlock(hidden_features, w0=10.0),
            PirateResidualBlock(hidden_features, w0=10.0)
        )
        
        self.micro_branch = nn.Sequential(
            SirenLayer(fourier_out_dim, hidden_features, w0=30.0, is_first=True, winner_noise_std=0.05),
            PirateResidualBlock(hidden_features, w0=30.0),
            PirateResidualBlock(hidden_features, w0=30.0)
        )
        
        self.final_layer = nn.Linear(hidden_features * 2, out_features)

    def forward(self, x):
        x_emb = self.fourier(x)
        macro_out = self.macro_branch(x_emb)
        micro_out = self.micro_branch(x_emb)
        combined = torch.cat([macro_out, micro_out], dim=-1)
        return self.final_layer(combined)

    # --- ADDED VISCOSITY CONTROLS ---
    def freeze_viscosity(self):
        self.viscosity.requires_grad = False

    def unfreeze_viscosity(self):
        self.viscosity.requires_grad = True