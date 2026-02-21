"""SAC Actor-Critic networks with Global CNN encoder."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


class GlobalCNNEncoder(nn.Module):
    """CNN encoder for 48x48 global map + scalar features."""

    def __init__(self, map_size: int = 48, map_channels: int = 3,
                 scalar_dim: int = 12):
        super().__init__()
        self.scalar_dim = scalar_dim
        self.conv = nn.Sequential(
            nn.Conv2d(map_channels, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, map_channels, map_size, map_size)
            conv_out = self.conv(dummy).view(1, -1).shape[1]
        self.feature_dim = conv_out + scalar_dim

    def forward(self, maps: torch.Tensor,
                scalars: torch.Tensor) -> torch.Tensor:
        h = self.conv(maps).flatten(1)
        return torch.cat([h, scalars], dim=1)


class SACActor(nn.Module):
    """Gaussian policy with tanh squashing."""

    def __init__(self, encoder: GlobalCNNEncoder, action_dim: int = 2,
                 hidden_dim: int = 256):
        super().__init__()
        self.encoder = encoder
        self.fc1 = nn.Linear(encoder.feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, maps: torch.Tensor, scalars: torch.Tensor):
        h = F.relu(self.fc1(self.encoder(maps, scalars)))
        h = F.relu(self.fc2(h))
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, maps: torch.Tensor, scalars: torch.Tensor):
        mean, log_std = self.forward(maps, scalars)
        std = log_std.exp()
        dist = Normal(mean, std)
        x = dist.rsample()  # reparameterization trick
        action = torch.tanh(x)
        # log_prob with tanh correction
        log_prob = dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        return action, log_prob

    def deterministic(self, maps: torch.Tensor, scalars: torch.Tensor):
        mean, _ = self.forward(maps, scalars)
        return torch.tanh(mean)


class SACCritic(nn.Module):
    """Twin Q-networks."""

    def __init__(self, encoder: GlobalCNNEncoder, action_dim: int = 2,
                 hidden_dim: int = 256):
        super().__init__()
        self.encoder = encoder
        feat = encoder.feature_dim + action_dim
        self.q1 = nn.Sequential(
            nn.Linear(feat, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(feat, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, maps: torch.Tensor, scalars: torch.Tensor,
                actions: torch.Tensor):
        h = self.encoder(maps, scalars)
        ha = torch.cat([h, actions], dim=1)
        return self.q1(ha), self.q2(ha)


class SACEntropyCritic(nn.Module):
    """Twin Q-networks for cumulative entropy estimation (TECRL)."""

    def __init__(self, encoder: GlobalCNNEncoder, action_dim: int = 2,
                 hidden_dim: int = 256):
        super().__init__()
        self.encoder = encoder
        feat = encoder.feature_dim + action_dim
        self.q1 = nn.Sequential(
            nn.Linear(feat, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(feat, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, maps: torch.Tensor, scalars: torch.Tensor,
                actions: torch.Tensor):
        h = self.encoder(maps, scalars)
        ha = torch.cat([h, actions], dim=1)
        return self.q1(ha), self.q2(ha)
