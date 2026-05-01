"""
3-D U-ResNet (Multi-Scale Residual Learning) for Cardiovascular Flow
=====================================================================
Replaces the Fourier Neural Operator with a U-Net encoder-decoder
architecture augmented with residual blocks at every scale.

Architecture:
  Encoder:    3 downsampling stages  (32³ → 16³ → 8³ → 4³)
  Bottleneck: 2 ResBlocks at the coarsest scale
  Decoder:    3 upsampling stages with skip connections (4³ → 8³ → 16³ → 32³)
  Output:     Conv3d projection to target channels

Each stage uses GroupNorm + GELU activation with residual connections.
Gradient checkpointing is used to fit 8-step rollout in limited VRAM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------------
#  Building block: 3-D Residual Block
# ---------------------------------------------------------------------------
class ResBlock3d(nn.Module):
    """
    Pre-activation residual block:
        x → GN → GELU → Conv3d → GN → GELU → Conv3d → + x

    If in_channels != out_channels, a 1x1 projection is used on the skip path.
    """

    def __init__(self, in_channels, out_channels=None, groups=8):
        super().__init__()
        out_channels = out_channels or in_channels

        self.gn1 = nn.GroupNorm(min(groups, in_channels), in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(min(groups, out_channels), out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

        # Skip projection if channel count changes
        self.skip = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.skip(x)
        out = F.gelu(self.gn1(x))
        out = self.conv1(out)
        out = F.gelu(self.gn2(out))
        out = self.conv2(out)
        return out + residual


# ---------------------------------------------------------------------------
#  Encoder block: Downsample + 2 ResBlocks
# ---------------------------------------------------------------------------
class DownBlock(nn.Module):
    """Strided convolution for 2× downsampling, followed by residual blocks."""

    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        self.down_conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.gn = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.res1 = ResBlock3d(out_ch, groups=groups)
        self.res2 = ResBlock3d(out_ch, groups=groups)

    def forward(self, x):
        x = F.gelu(self.gn(self.down_conv(x)))
        x = self.res1(x)
        x = self.res2(x)
        return x


# ---------------------------------------------------------------------------
#  Decoder block: Upsample + Concatenate skip + 2 ResBlocks
# ---------------------------------------------------------------------------
class UpBlock(nn.Module):
    """Transposed convolution for 2× upsampling with skip connection."""

    def __init__(self, in_ch, skip_ch, out_ch, groups=8):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=2, stride=2)
        self.fuse = nn.Conv3d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1)
        self.gn = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.res1 = ResBlock3d(out_ch, groups=groups)
        self.res2 = ResBlock3d(out_ch, groups=groups)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle potential size mismatch from odd dimensions
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = F.gelu(self.gn(self.fuse(x)))
        x = self.res1(x)
        x = self.res2(x)
        return x


# ---------------------------------------------------------------------------
#  Full model: 3-D U-ResNet
# ---------------------------------------------------------------------------
class UResNet3d(nn.Module):
    """
    3-D U-Net with Residual Blocks for spatiotemporal PDE problems.

    Resolution path (for 32³ input):
        Encoder:  32³ → 16³ → 8³ → 4³
        Decoder:  4³ → 8³ → 16³ → 32³

    Args:
        in_channels:  input field channels (e.g. vel(3)+pres(1)+time(1)+mask(1)+coords(3) = 9)
        out_channels: target field channels (e.g. vel(3)+pres(1)+time(1) = 5)
        base_width:   channel count at the first encoder level; doubles at each stage
        groups:       number of groups for GroupNorm
        use_checkpoint: use gradient checkpointing to reduce VRAM usage
    """

    def __init__(
        self,
        in_channels=9,
        out_channels=5,
        base_width=32,
        groups=8,
        use_checkpoint=True,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        w = base_width

        # --- Encoder ---
        # Initial feature extraction at full resolution
        self.enc0 = nn.Sequential(
            nn.Conv3d(in_channels, w, kernel_size=3, padding=1),
            nn.GroupNorm(min(groups, w), w),
            nn.GELU(),
            ResBlock3d(w, groups=groups),
            ResBlock3d(w, groups=groups),
        )
        self.down1 = DownBlock(w, w * 2, groups=groups)       # 32³ → 16³
        self.down2 = DownBlock(w * 2, w * 4, groups=groups)   # 16³ → 8³
        self.down3 = DownBlock(w * 4, w * 8, groups=groups)   # 8³  → 4³

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            ResBlock3d(w * 8, groups=groups),
            ResBlock3d(w * 8, groups=groups),
            ResBlock3d(w * 8, groups=groups),   # extra block for richer global features
        )

        # --- Decoder ---
        self.up3 = UpBlock(w * 8, w * 4, w * 4, groups=groups)   # 4³  → 8³
        self.up2 = UpBlock(w * 4, w * 2, w * 2, groups=groups)   # 8³  → 16³
        self.up1 = UpBlock(w * 2, w,     w,     groups=groups)   # 16³ → 32³

        # --- Output projection ---
        self.out_conv = nn.Sequential(
            nn.Conv3d(w, w, kernel_size=3, padding=1),
            nn.GroupNorm(min(groups, w), w),
            nn.GELU(),
            nn.Conv3d(w, out_channels, kernel_size=1),
        )

    def _encoder(self, x):
        e0 = self.enc0(x)
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        e3 = self.down3(e2)
        return e0, e1, e2, e3

    def _bottleneck(self, e3):
        return self.bottleneck(e3)

    def _decoder(self, b, e0, e1, e2):
        d2 = self.up3(b, e2)
        d1 = self.up2(d2, e1)
        d0 = self.up1(d1, e0)
        return self.out_conv(d0)

    def forward(self, x):
        """
        x : (B, C_in, X, Y, Z)
        returns : (B, C_out, X, Y, Z)
        """
        if self.use_checkpoint and self.training:
            e0, e1, e2, e3 = checkpoint(self._encoder, x, use_reentrant=False)
            b = checkpoint(self._bottleneck, e3, use_reentrant=False)
            out = checkpoint(self._decoder, b, e0, e1, e2, use_reentrant=False)
        else:
            e0, e1, e2, e3 = self._encoder(x)
            b = self._bottleneck(e3)
            out = self._decoder(b, e0, e1, e2)
        return out
