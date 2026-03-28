import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ────────────────────────────────────────────────────────────────────────────
#  Core layer: 3-D Spectral Convolution  (Eq. 5-6 of the FNO paper)
# ────────────────────────────────────────────────────────────────────────────
class SpectralConv3d(nn.Module):
    """
    Performs a linear transform on the truncated lower Fourier modes:
        K(φ)v  =  F^{-1}( R_φ  ·  F(v) )

    The 3-D real FFT stores only positive z-frequencies (Hermitian symmetry),
    so for (x, y) we keep four low-frequency corners; for z only positive side.
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

        # Complex-valued learnable weights, kept in float32 — FFT requires it.
        self.w1 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.w3 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.w4 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))

    @staticmethod
    def _cmul(a, b):
        """Batched complex matrix–vector multiply via einsum."""
        return torch.einsum("bixyz,ioxyz->boxyz", a, b)

    def forward(self, x):
        orig_dtype = x.dtype
        x = x.float()                           # FFT needs float32

        sx, sy, sz = x.size(-3), x.size(-2), x.size(-1)
        m1, m2, m3 = self.modes1, self.modes2, self.modes3

        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        out_ft = torch.zeros(
            x.size(0), self.out_channels, sx, sy, sz // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )

        # Four low-frequency corners in (x, y); positive z only
        out_ft[:, :, :m1,  :m2,  :m3] = self._cmul(x_ft[:, :, :m1,  :m2,  :m3], self.w1)
        out_ft[:, :, -m1:, :m2,  :m3] = self._cmul(x_ft[:, :, -m1:, :m2,  :m3], self.w2)
        out_ft[:, :, :m1,  -m2:, :m3] = self._cmul(x_ft[:, :, :m1,  -m2:, :m3], self.w3)
        out_ft[:, :, -m1:, -m2:, :m3] = self._cmul(x_ft[:, :, -m1:, -m2:, :m3], self.w4)

        result = torch.fft.irfftn(out_ft, s=(sx, sy, sz))
        return result.to(orig_dtype)            # cast back to bfloat16


# ────────────────────────────────────────────────────────────────────────────
#  Original: 3-D Fourier Neural Operator  (kept for checkpoint compatibility)
# ────────────────────────────────────────────────────────────────────────────
class FNO3d(nn.Module):
    """
    Standard FNO3d — global spectral operator on the full input grid.

    Kept for backward compatibility with existing checkpoints.
    For new training runs, prefer HFNO3d.

    Args:
        modes1/2/3  : Fourier modes kept per spatial dimension
        width       : hidden channel dimension
        in_channels : input channels (vel×3 + pres + time + mask + coords = 9)
        out_channels: output channels (vel×3 + pres + time = 5)
        num_layers  : number of stacked Fourier layers
    """

    def __init__(
        self,
        modes1=8, modes2=8, modes3=8,
        width=32,
        in_channels=9,
        out_channels=5,
        num_layers=4,
    ):
        super().__init__()
        self.num_layers = num_layers

        self.lift = nn.Linear(in_channels, width)

        self.spec_convs  = nn.ModuleList()
        self.local_convs = nn.ModuleList()
        for _ in range(num_layers):
            self.spec_convs.append(SpectralConv3d(width, width, modes1, modes2, modes3))
            self.local_convs.append(nn.Conv3d(width, width, kernel_size=1))

        self.proj1 = nn.Linear(width, 128)
        self.proj2 = nn.Linear(128, out_channels)

    def forward(self, x):
        """x : (B, C_in, X, Y, Z)  →  (B, C_out, X, Y, Z)"""
        x = x.permute(0, 2, 3, 4, 1)
        x = self.lift(x)
        x = x.permute(0, 4, 1, 2, 3)

        for i in range(self.num_layers):
            def layer_fn(x, i=i):
                x1 = self.spec_convs[i](x)
                x2 = self.local_convs[i](x)
                x  = x1 + x2
                if i < self.num_layers - 1:
                    x = F.gelu(x)
                return x
            x = checkpoint(layer_fn, x, use_reentrant=False)

        x = x.permute(0, 2, 3, 4, 1)
        x = F.gelu(self.proj1(x))
        x = self.proj2(x)
        return x.permute(0, 4, 1, 2, 3)


# ────────────────────────────────────────────────────────────────────────────
#  H-FNO building blocks
# ────────────────────────────────────────────────────────────────────────────
class DownConv(nn.Module):
    """
    Encoder block: extracts local boundary features and halves the spatial grid.

    Strided Conv3d + GroupNorm + GELU.
    GroupNorm is preferred over BatchNorm for 3-D fluid data because:
      - Batch size is typically 1 (VRAM constraint) → BN statistics are noisy.
      - GN normalises within each sample across spatial locations, which is
        stable regardless of batch size.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # num_groups=8 divides cleanly for out_channels ∈ {16, 32, 64, 128}
        n_groups = min(8, out_channels)
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(n_groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    """
    Decoder block: upsamples and merges skip-connection features from the encoder.

    ConvTranspose3d doubles the spatial resolution, then the skip features
    (high-resolution boundary information) are concatenated and a 3×3×3
    convolution fuses them.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        n_groups = min(8, out_channels)
        self.up   = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            # After concat: out_channels (from up) + out_channels (from skip) = out_channels*2
            nn.Conv3d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(n_groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x, skip_feature):
        x = self.up(x)
        x = torch.cat([x, skip_feature], dim=1)   # inject boundary detail
        return self.conv(x)


# ────────────────────────────────────────────────────────────────────────────
#  HFNO3d — Hierarchical Fourier Neural Operator
# ────────────────────────────────────────────────────────────────────────────
class HFNO3d(nn.Module):
    """
    Hierarchical 3-D Fourier Neural Operator.

    Stage 1 — Encoder (local feature extraction):
        Conv3d  :  in_channels → width//2   (full res,   e.g. 64³)
        DownConv:  width//2   → width       (half res,   e.g. 32³)
        DownConv:  width      → width*2     (quarter res, e.g. 16³)

        Strided convolutions extract sharp vessel-wall geometry at each scale
        before the FFT can blur them.

    Stage 2 — Spectral bottleneck (global operator):
        N × (SpectralConv3d + 1×1 Conv + GELU)  on the quarter-res grid.

        Physics advantage: the FFT here is 64× cheaper than on the full grid,
        and the modes cover a larger fraction of the spectrum, so large-scale
        pressure waves and momentum transport are better resolved.

    Stage 3 — Decoder (reconstruction):
        UpConv  : width*2  → width       (quarter → half)
        UpConv  : width    → width//2    (half    → full)
        Conv3d 1×1: width//2 → out_channels

        Each UpConv adds the corresponding encoder skip connection so the
        exact vessel wall position is available at full resolution.

    Parameter note:
        With width=32 and num_fno_layers=4 the bottleneck SpectralConv3d
        layers operate on 64 channels — this is ~4× more parameters than
        the plain FNO3d (≈33M vs ≈8.4M) but the FNO is running on a 64×
        smaller grid (16³ vs 64³), so the forward pass is cheaper and
        gradient checkpointing keeps VRAM manageable.

        To reduce params while keeping the hierarchical structure, set
        bottleneck_width = width  instead of width*2 in the constructor.

    Args:
        modes        : Fourier modes per dimension (applied at quarter-res grid).
                       Constraint: modes ≤ (res // 4) // 2.
                       At 64³ input with 2 down-steps → 16³ bottleneck → modes ≤ 8. ✓
        width        : base hidden channel width (encoder starts at width//2).
        in_channels  : input channels (vel×3 + pres + time + mask + coords = 9).
        out_channels : output channels (vel×3 + pres + time = 5).
        num_fno_layers: number of FNO blocks at the bottleneck.
    """

    def __init__(
        self,
        modes=8,
        width=32,
        in_channels=9,
        out_channels=5,
        num_fno_layers=4,
    ):
        super().__init__()
        self.num_fno_layers = num_fno_layers
        bottleneck_ch = width * 2   # channel depth at the FNO bottleneck

        # ── Stage 1: Encoder ────────────────────────────────────────────────
        # x0: full res,   width//2  channels  (skip for up2)
        # x1: half res,   width     channels  (skip for up1)
        # x_bt: quarter res, bottleneck_ch  channels  (bottleneck input)
        self.inc   = nn.Conv3d(in_channels, width // 2, kernel_size=3, padding=1)
        self.down1 = DownConv(width // 2, width)
        self.down2 = DownConv(width,      bottleneck_ch)

        # ── Stage 2: Spectral bottleneck ────────────────────────────────────
        self.fno_blocks    = nn.ModuleList()
        self.local_mixers  = nn.ModuleList()
        for _ in range(num_fno_layers):
            self.fno_blocks.append(
                SpectralConv3d(bottleneck_ch, bottleneck_ch, modes, modes, modes)
            )
            self.local_mixers.append(
                nn.Conv3d(bottleneck_ch, bottleneck_ch, kernel_size=1)
            )

        # ── Stage 3: Decoder ────────────────────────────────────────────────
        self.up1      = UpConv(bottleneck_ch, width)
        self.up2      = UpConv(width,          width // 2)
        self.out_conv = nn.Conv3d(width // 2, out_channels, kernel_size=1)

    def forward(self, x):
        """
        x : (B, C_in, X, Y, Z)
        returns : (B, C_out, X, Y, Z)   — same spatial dimensions as input
        """
        # ── Stage 1: Encode ─────────────────────────────────────────────────
        x0 = F.gelu(self.inc(x))     # (B, width//2, X, Y, Z)         — full res
        x1 = self.down1(x0)          # (B, width,    X/2, Y/2, Z/2)   — half res
        x_bt = self.down2(x1)        # (B, width*2,  X/4, Y/4, Z/4)   — bottleneck

        # ── Stage 2: Spectral bottleneck (with gradient checkpointing) ──────
        for i in range(self.num_fno_layers):
            def bottleneck_fn(x_bt, i=i):
                x_fno = self.fno_blocks[i](x_bt)
                x_loc = self.local_mixers[i](x_bt)
                return F.gelu(x_fno + x_loc)
            x_bt = checkpoint(bottleneck_fn, x_bt, use_reentrant=False)

        # ── Stage 3: Decode + skip connections ──────────────────────────────
        x_up = self.up1(x_bt, x1)   # (B, width,    X/2, Y/2, Z/2)
        x_up = self.up2(x_up, x0)   # (B, width//2, X, Y, Z)

        return self.out_conv(x_up)   # (B, C_out, X, Y, Z)