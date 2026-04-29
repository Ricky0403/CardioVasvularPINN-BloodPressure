"""
3D Fourier Neural Operator variants for blood-flow surrogate modelling.
=======================================================================

Three architectures provided:

  FNO3d   — original flat operator (kept for checkpoint compatibility)
  HUFNO3d — parallel fusion: SpectralConv3d ∥ MiniUNet3d inside EVERY layer
             (recommended — designed for pressure prediction in vessel flow)

Why HUFNO3d for blood-flow pressure prediction
──────────────────────────────────────────────
Blood pressure lives at two scales simultaneously:

  Global: the cardiac pressure wave propagates through the whole vessel.
          SpectralConv3d captures this — FFT modes are global basis functions
          that naturally represent smooth large-scale pressure gradients.

  Local:  the no-slip boundary layer, recirculation zones and vena contracta
          are sharp features near the wall.  MiniUNet3d captures these with
          local 3×3×3 kernels that can represent sharp discontinuities that
          a global Fourier basis would need O(res) modes to represent.

The HUFNO layer fuses both paths in parallel (sum before nonlinearity), so
the model allocates spectral capacity to pressure waves and convolutional
capacity to boundary geometry without the two competing for the same weights.

Architecture of one HUFNOLayer3d
─────────────────────────────────────────────
  Input  (B, width, X, Y, Z)
       │
       ├─ SpectralConv3d ──────── x1  (global: Fourier modes, pressure waves)
       ├─ MiniUNet3d ─────────── x2  (local:  wall geometry, recirculation)
       └─ Conv3d(k=1) ─────────── x3  (pointwise: channel mixing / residual)
       │
       └─ GELU(x1 + x2 + x3) → output

Output channel convention (out_ch = 5):
  ch 0: u  velocity-x   (standardised)
  ch 1: v  velocity-y   (standardised)
  ch 2: w  velocity-z   (standardised)
  ch 3: p  pressure     (standardised)  ← primary prediction goal
  ch 4: t  cardiac-cycle time index (0→1, passed through unchanged)

VRAM (WIDTH=32, GRID_RES=64, B=1, bfloat16, gradient-checkpointed):
  ~340 MB total → comfortable on 8 GB.
  Width 48 → ~620 MB,  Width 64 → ~960 MB  (both fit).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ────────────────────────────────────────────────────────────────────────────
#  Core: 3-D Spectral Convolution  (Eq. 5-6 of Li et al. 2020)
# ────────────────────────────────────────────────────────────────────────────
class SpectralConv3d(nn.Module):
    """
    K(φ)v = F^{-1}( R_φ · F(v) )

    Real FFT → only positive z-frequencies stored (Hermitian symmetry).
    Four low-frequency corners in (x, y) × positive-z strip.
    Weights always float32 (FFT kernel requirement); cast down after irfftn.
    """

    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        scale = 1.0 / (in_channels * out_channels)
        shape = (in_channels, out_channels, modes1, modes2, modes3)
        self.w1 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.w3 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.w4 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))

    @staticmethod
    def _cmul(a, b):
        return torch.einsum("bixyz,ioxyz->boxyz", a, b)

    def forward(self, x):
        orig = x.dtype
        x    = x.float()
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

        return torch.fft.irfftn(out_ft, s=(sx, sy, sz)).to(orig)


# ────────────────────────────────────────────────────────────────────────────
#  Mini 3-D U-Net — local hierarchical feature extractor
# ────────────────────────────────────────────────────────────────────────────
def _gn(ch):
    """Safe GroupNorm: clamp num_groups to max that divides ch."""
    for g in (8, 4, 2, 1):
        if ch % g == 0:
            return nn.GroupNorm(g, ch)


class _Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, stride=2, padding=1),
            _gn(out_ch), nn.GELU(),
        )
    def forward(self, x): return self.net(x)


class _Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose3d(in_ch, in_ch, 2, stride=2)
        self.fuse = nn.Sequential(
            nn.Conv3d(in_ch + skip_ch, out_ch, 3, padding=1),
            _gn(out_ch), nn.GELU(),
        )
    def forward(self, x, skip):
        return self.fuse(torch.cat([self.up(x), skip], dim=1))


class MiniUNet3d(nn.Module):
    """
    3-D U-Net that fits INSIDE one HUFNO layer.

    (B, W, X, Y, Z) → (B, W, X, Y, Z)   same shape, richer features.

    Hierarchy (W = width):
      skip0 = x                             (B, W,   X,   Y,   Z )
      e1    = down1(skip0)                  (B, W/2, X/2, Y/2, Z/2)
      bt    = down2(e1) → bottleneck        (B, W/4, X/4, Y/4, Z/4)
      d1    = up1(bt, e1)                   (B, W/2, X/2, Y/2, Z/2)
      out   = up2(d1, skip0)               (B, W,   X,   Y,   Z )

    GroupNorm instead of BatchNorm → stable at batch_size=1.
    """

    def __init__(self, width):
        super().__init__()
        w2, w4 = width // 2, width // 4

        self.down1      = _Down(width, w2)
        self.down2      = _Down(w2, w4)
        self.bottleneck = nn.Sequential(
            nn.Conv3d(w4, w4, 3, padding=1), _gn(w4), nn.GELU()
        )
        self.up1 = _Up(w4, w2, w2)
        self.up2 = _Up(w2, width, width)

    def forward(self, x):
        skip0 = x
        e1    = self.down1(skip0)
        bt    = self.bottleneck(self.down2(e1))
        d1    = self.up1(bt, e1)
        return self.up2(d1, skip0)


# ────────────────────────────────────────────────────────────────────────────
#  HUFNO layer: three parallel paths, single fusion
# ────────────────────────────────────────────────────────────────────────────
class HUFNOLayer3d(nn.Module):
    """
    σ( SpectralConv(x) + MiniUNet(x) + Conv1×1(x) )

    All three paths read the SAME input x — strictly parallel, not sequential.
    This lets the model specialise each path without interference.
    """

    def __init__(self, width, modes):
        super().__init__()
        self.spec_conv  = SpectralConv3d(width, width, modes, modes, modes)
        self.mini_unet  = MiniUNet3d(width)
        self.local_conv = nn.Conv3d(width, width, kernel_size=1)

    def forward(self, x):
        return F.gelu(
            self.spec_conv(x) + self.mini_unet(x) + self.local_conv(x)
        )


# ────────────────────────────────────────────────────────────────────────────
#  HUFNO3d — full model
# ────────────────────────────────────────────────────────────────────────────
class HUFNO3d(nn.Module):
    """
    Hierarchical U-Net Fourier Neural Operator.

    Args:
        modes       : Fourier modes per spatial dim.
                      Must satisfy: modes ≤ (GRID_RES // 4) // 2.
                      For GRID_RES=64 → modes ≤ 8  ✓
        width       : hidden channel depth (default 32, can raise to 48/64).
        in_channels : 9   (vel×3 + pres + time + mask + coords×3).
        out_channels: 5   (vel×3 + pres + time).
        num_layers  : number of HUFNO layers (default 4).
    """

    def __init__(
        self,
        modes=8,
        width=32,
        in_channels=9,
        out_channels=5,
        num_layers=4,
    ):
        super().__init__()
        self.num_layers = num_layers

        self.lift   = nn.Linear(in_channels, width)
        self.layers = nn.ModuleList([
            HUFNOLayer3d(width, modes) for _ in range(num_layers)
        ])
        self.proj1  = nn.Linear(width, 128)
        self.proj2  = nn.Linear(128, out_channels)

    def forward(self, x):
        """x: (B, C_in, X, Y, Z) → (B, C_out, X, Y, Z)"""
        # Lift
        x = x.permute(0, 2, 3, 4, 1)
        x = self.lift(x)
        x = x.permute(0, 4, 1, 2, 3)

        # HUFNO layers with gradient checkpointing
        for layer in self.layers:
            x = checkpoint(layer, x, use_reentrant=False)

        # Project
        x = x.permute(0, 2, 3, 4, 1)
        x = F.gelu(self.proj1(x))
        x = self.proj2(x)
        return x.permute(0, 4, 1, 2, 3)


# ────────────────────────────────────────────────────────────────────────────
#  FNO3d — kept for checkpoint backward-compatibility only
# ────────────────────────────────────────────────────────────────────────────
class FNO3d(nn.Module):
    """Original flat FNO3d.  Use HUFNO3d for new training."""

    def __init__(self, modes1=8, modes2=8, modes3=8,
                 width=32, in_channels=9, out_channels=5, num_layers=4):
        super().__init__()
        self.num_layers = num_layers
        self.lift = nn.Linear(in_channels, width)
        self.spec_convs  = nn.ModuleList([
            SpectralConv3d(width, width, modes1, modes2, modes3)
            for _ in range(num_layers)
        ])
        self.local_convs = nn.ModuleList([
            nn.Conv3d(width, width, kernel_size=1) for _ in range(num_layers)
        ])
        self.proj1 = nn.Linear(width, 128)
        self.proj2 = nn.Linear(128, out_channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        x = self.lift(x)
        x = x.permute(0, 4, 1, 2, 3)
        for i in range(self.num_layers):
            def fn(x, i=i):
                out = self.spec_convs[i](x) + self.local_convs[i](x)
                return F.gelu(out) if i < self.num_layers - 1 else out
            x = checkpoint(fn, x, use_reentrant=False)
        x = x.permute(0, 2, 3, 4, 1)
        x = F.gelu(self.proj1(x))
        x = self.proj2(x)
        return x.permute(0, 4, 1, 2, 3)