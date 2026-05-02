"""
3D Fourier Neural Operator (FNO) — Li et al. 2020
===================================================
Learns an operator mapping between function spaces by parameterizing
the integral kernel directly in Fourier space.

Architecture  (paper §3-4, Figure 1a):
  1. Pointwise lifting   P :  in_channels  →  width   (nn.Linear)
  2. N Fourier layers:   v_{t+1}(x) = 0( W·v_t(x) + K(φ)·v_t(x) )
       K(φ) = F^{-1}( R_φ · F(v_t) )      (SpectralConv3d)
       W    = 1x1x1 convolution             (nn.Conv3d, kernel=1)
  3. Pointwise projection Q :  width  →  out_channels  (two nn.Linear layers)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------------
#  Core layer: 3-D Spectral Convolution  (Eq. 5-6 of the paper)
# ---------------------------------------------------------------------------
class SpectralConv3d(nn.Module):
    """
    Performs a linear transform on the truncated lower Fourier modes:
        K(φ)v  =  F^{-1}( R_φ  ·  F(v) )

    The 3-D real FFT stores only positive z-frequencies (Hermitian symmetry),
    so for dimensions (x, y) we have four "corners" of low-frequency modes
    (positive and negative), and for z only the positive side.
    """

    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1   # Fourier modes to keep in x
        self.modes2 = modes2   # Fourier modes to keep in y
        self.modes3 = modes3   # Fourier modes to keep in z

        scale = 1.0 / (in_channels * out_channels)
        shape = (in_channels, out_channels, modes1, modes2, modes3)

        # Keep spectral weights in float32 always — FFT needs it
        # Complex-valued learnable weight tensor R for each corner
        self.w1 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.w3 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.w4 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))

    @staticmethod
    def _cmul(a, b):
        """Batched complex matrix–vector multiply via einsum."""
        return torch.einsum("bixyz,ioxyz->boxyz", a, b)

    def forward(self, x):
        # FFT requires float32 — cast up, compute, cast back
        orig_dtype = x.dtype
        x = x.float()

        sx, sy, sz = x.size(-3), x.size(-2), x.size(-1)
        m1, m2, m3 = self.modes1, self.modes2, self.modes3

        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        out_ft = torch.zeros(
            x.size(0), self.out_channels, sx, sy, sz // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )

        out_ft[:, :, :m1,  :m2,  :m3] = self._cmul(x_ft[:, :, :m1,  :m2,  :m3], self.w1)
        out_ft[:, :, -m1:, :m2,  :m3] = self._cmul(x_ft[:, :, -m1:, :m2,  :m3], self.w2)
        out_ft[:, :, :m1,  -m2:, :m3] = self._cmul(x_ft[:, :, :m1,  -m2:, :m3], self.w3)
        out_ft[:, :, -m1:, -m2:, :m3] = self._cmul(x_ft[:, :, -m1:, -m2:, :m3], self.w4)

        result = torch.fft.irfftn(out_ft, s=(sx, sy, sz))
        return result  # keep float32; autocast handles mixed precision boundaries


# ---------------------------------------------------------------------------
#  Full model: 3-D Fourier Neural Operator
# ---------------------------------------------------------------------------
class FNO3d(nn.Module):
    """
    3-D Fourier Neural Operator for spatiotemporal PDE problems.

    Args:
        modes1/2/3 : number of Fourier modes kept per spatial dim (paper: 12 for 2-D)
        width      : hidden channel dimension  dv  (paper: 32 for 2-D)
        in_channels: input field channels  (e.g. vel(3)+pres(1)+mask(1)+coords(3) = 8)
        out_channels: target field channels (e.g. vel(3)+pres(1) = 4)
        num_layers : number of stacked Fourier layers (paper: 4)
    """

    def __init__(
        self,
        modes1=8,
        modes2=8,
        modes3=8,
        width=32,
        in_channels=8,
        out_channels=4,
        num_layers=4,
    ):
        super().__init__()
        self.num_layers = num_layers

        # --- Lifting  P ---
        self.lift = nn.Linear(in_channels, width)

        # --- Fourier layers ---
        self.spec_convs  = nn.ModuleList()
        self.local_convs = nn.ModuleList()
        for _ in range(num_layers):
            self.spec_convs.append(
                SpectralConv3d(width, width, modes1, modes2, modes3)
            )
            self.local_convs.append(nn.Conv3d(width, width, kernel_size=1))

        # --- Projection  Q ---
        self.proj1 = nn.Linear(width, 128)
        self.proj2 = nn.Linear(128, out_channels)

    def reset_output_head(self):
        """Helper to reinitialize pointwise projection layers for curriculum learning."""
        nn.init.xavier_uniform_(self.proj1.weight)
        nn.init.zeros_(self.proj1.bias)
        nn.init.xavier_uniform_(self.proj2.weight)
        nn.init.zeros_(self.proj2.bias)

    def forward(self, x):
        """
        x : (B, C_in, X, Y, Z)
        returns : (B, C_out, X, Y, Z)
        """
        # Lifting  (pointwise across channels)
        x = x.permute(0, 2, 3, 4, 1)       # → (B, X, Y, Z, C_in)
        x = self.lift(x)
        x = x.permute(0, 4, 1, 2, 3)       # → (B, width, X, Y, Z)

        # Fourier layers
        for i in range(self.num_layers):
            def layer_fn(x, i=i):
                x1 = self.spec_convs[i](x)
                x2 = self.local_convs[i](x)
                x = x1 + x2
                if i < self.num_layers - 1:
                    x = F.gelu(x)
                return x
            x = checkpoint(layer_fn, x, use_reentrant=False)

        # Projection
        x = x.permute(0, 2, 3, 4, 1)       # → (B, X, Y, Z, width)
        x = F.gelu(self.proj1(x))
        x = self.proj2(x)
        return x.permute(0, 4, 1, 2, 3)    # → (B, C_out, X, Y, Z)