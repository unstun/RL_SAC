"""Plug-and-play modules for CNN-DDQN (V11).

Modules
-------
- NoisyLinear  (Fortunato et al., ICLR 2018)
- CBAM         (Woo et al., ECCV 2018)
- SpatialMHA   (Vaswani et al., 2017 — applied on CNN spatial feature maps)
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# NoisyLinear  (Fortunato et al., "Noisy Networks for Exploration", ICLR 2018)
# Factorised Gaussian noise variant.
# ---------------------------------------------------------------------------

class NoisyLinear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` with learnable noise."""

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_eps", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_eps", torch.empty(out_features))

        self._sigma_init = float(sigma_init)
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self._sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(self._sigma_init / math.sqrt(self.in_features))

    @staticmethod
    def _factorised_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        eps_in = self._factorised_noise(self.in_features)
        eps_out = self._factorised_noise(self.out_features)
        self.weight_eps.copy_(eps_out.outer(eps_in))
        self.bias_eps.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_eps
            b = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)


# ---------------------------------------------------------------------------
# CBAM  (Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018)
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """Squeeze-and-excitation style channel attention with avg+max pooling."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        avg = x.mean(dim=(2, 3))                   # (B, C)
        mx = x.amax(dim=(2, 3))                    # (B, C)
        att = torch.sigmoid(self.fc(avg) + self.fc(mx))  # (B, C)
        return x * att.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    """Spatial attention via channel-wise avg+max pooling → conv."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=pad, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        avg = x.mean(dim=1, keepdim=True)           # (B, 1, H, W)
        mx = x.amax(dim=1, keepdim=True)            # (B, 1, H, W)
        att = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * att


class CBAM(nn.Module):
    """Channel + Spatial attention (串联)."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7) -> None:
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# ---------------------------------------------------------------------------
# SpatialMHA  (Multi-Head Self-Attention on CNN spatial feature maps)
# Treats each spatial position as a token; embed_dim = channels.
# ---------------------------------------------------------------------------

class SpatialMHA(nn.Module):
    """Multi-head self-attention over spatial positions of a feature map."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True,
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)       # (B, H*W, C)
        out, _ = self.mha(tokens, tokens, tokens)    # self-attention
        out = self.norm(tokens + out)                # residual + LN
        return out.transpose(1, 2).reshape(B, C, H, W)
