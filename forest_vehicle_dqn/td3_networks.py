"""TD3 network modules: GlobalCNNEncoder actor + LocalCNNEncoder for V16-C obs."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from forest_vehicle_dqn.sac_networks import GlobalCNNEncoder  # reuse


class TD3Actor(nn.Module):
    """Deterministic policy: obs → tanh(MLP(encoder(maps, scalars))) → [-1,1]²."""

    def __init__(self, encoder: GlobalCNNEncoder, action_dim: int = 2,
                 hidden_dim: int = 256):
        super().__init__()
        self.encoder = encoder
        self.fc1 = nn.Linear(encoder.feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, maps: torch.Tensor,
                scalars: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(self.encoder(maps, scalars)))
        h = F.relu(self.fc2(h))
        return torch.tanh(self.out(h))


class LocalCNNEncoder(nn.Module):
    """V16-C 同款编码器：接受 (maps, scalars) dict obs，maps 为 (B,1,12,12)。

    复用 CNNQNetwork 的 3-layer CNN 架构（Conv s1 → Conv s2 → Conv s2）。
    feature_dim = conv_flat_dim + scalar_dim，供 TD3Actor/Critic 使用。
    """

    def __init__(self, map_size: int = 12, map_channels: int = 1,
                 scalar_dim: int = 10) -> None:
        super().__init__()
        self.map_size = int(map_size)
        self.map_channels = int(map_channels)
        self.scalar_dim = int(scalar_dim)

        self.conv = nn.Sequential(
            nn.Conv2d(map_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, map_channels, map_size, map_size)
            conv_out_dim = int(self.conv(dummy).flatten(1).shape[1])
        self.feature_dim = conv_out_dim + scalar_dim

    def forward(self, maps: torch.Tensor,
                scalars: torch.Tensor) -> torch.Tensor:
        """maps: (B, C, H, W)  scalars: (B, scalar_dim) → features (B, feature_dim)."""
        h = self.conv(maps).flatten(1)
        return torch.cat([h, scalars], dim=1)
