from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from forest_vehicle_dqn.modules import CBAM, NoisyLinear, SpatialMHA


class MLPQNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, *, hidden_dim: int = 128, hidden_layers: int = 2):
        super().__init__()

        if hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1")

        layers: list[nn.Module] = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Backwards-compatible name (historically this repo only had an MLP Q-network).
QNetwork = MLPQNetwork


@dataclass(frozen=True)
class FlatObsCnnLayout:
    scalar_dim: int
    map_channels: int
    map_size: int


def infer_flat_obs_cnn_layout(obs_dim: int) -> FlatObsCnnLayout:
    """Infer (scalar_dim, map_channels, map_size) for this repo's flat observations.

    Supported layouts:
    - AMRGridEnv:   obs = [5 scalars] + [1 * (N*N) map]
    - AMRBicycleEnv:obs = [10 scalars] + [1 * (N*N) map]  (occ)
    """

    d = int(obs_dim)
    if d <= 0:
        raise ValueError("obs_dim must be > 0")

    candidates: list[FlatObsCnnLayout] = []
    for scalar_dim, channels in ((5, 1), (10, 1)):
        rem = d - int(scalar_dim)
        if rem <= 0:
            continue
        if rem % int(channels) != 0:
            continue
        per = rem // int(channels)
        n = int(round(math.sqrt(per)))
        if n > 0 and n * n == per:
            candidates.append(FlatObsCnnLayout(scalar_dim=int(scalar_dim), map_channels=int(channels), map_size=int(n)))

    if not candidates:
        raise ValueError(
            f"Cannot infer CNN layout from obs_dim={d}. Expected 5+N^2 (grid) or 10+N^2 (bicycle)."
        )
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous CNN layout for obs_dim={d}: {candidates}")
    return candidates[0]


class CNNQNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        scalar_dim: int,
        map_channels: int,
        map_size: int,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
        dueling: bool = False,
        cbam: bool = False,
        noisy_net: bool = False,
        mha: bool = False,
        mha_heads: int = 4,
        n_quantiles: int = 1,
    ) -> None:
        super().__init__()

        self.scalar_dim = int(scalar_dim)
        self.map_channels = int(map_channels)
        self.map_size = int(map_size)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.dueling = bool(dueling)
        self.noisy_net = bool(noisy_net)
        self.n_quantiles = max(1, int(n_quantiles))

        if self.scalar_dim < 0:
            raise ValueError("scalar_dim must be >= 0")
        if self.map_channels < 1:
            raise ValueError("map_channels must be >= 1")
        if self.map_size < 1:
            raise ValueError("map_size must be >= 1")
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1")

        expected = int(self.scalar_dim) + int(self.map_channels) * int(self.map_size) * int(self.map_size)
        if int(input_dim) != expected:
            raise ValueError(
                f"CNNQNetwork expected input_dim={expected} (scalar_dim={self.scalar_dim}, "
                f"map_channels={self.map_channels}, map_size={self.map_size}), got {int(input_dim)}"
            )

        # A real 2D CNN over the downsampled global maps. Designed for small maps (e.g. 12x12).
        conv_out_channels = 64
        self.conv = nn.Sequential(
            nn.Conv2d(self.map_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, conv_out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Optional CBAM attention on conv feature maps.
        self.cbam_mod: CBAM | None = CBAM(conv_out_channels) if cbam else None

        # Optional spatial multi-head self-attention on conv feature maps.
        self.spatial_mha: SpatialMHA | None = SpatialMHA(conv_out_channels, mha_heads) if mha else None

        with torch.no_grad():
            dummy = torch.zeros((1, self.map_channels, self.map_size, self.map_size), dtype=torch.float32)
            conv_out = self.conv(dummy)
            conv_out_dim = int(conv_out.flatten(start_dim=1).shape[1])
        fc_in_dim = int(self.scalar_dim) + int(conv_out_dim)

        # Helper: NoisyLinear when noisy_net else nn.Linear.
        def _lin(in_f: int, out_f: int) -> nn.Module:
            return NoisyLinear(in_f, out_f) if self.noisy_net else nn.Linear(in_f, out_f)

        # For QR-DQN: effective output = n_actions * n_quantiles.
        eff_out = int(output_dim) * self.n_quantiles

        if self.dueling:
            # Shared layers: all but the last hidden layer.
            shared: list[nn.Module] = []
            shared.append(_lin(fc_in_dim, int(hidden_dim)))
            shared.append(nn.ReLU())
            for _ in range(max(0, int(hidden_layers) - 2)):
                shared.append(_lin(int(hidden_dim), int(hidden_dim)))
                shared.append(nn.ReLU())
            self.shared = nn.Sequential(*shared)
            # Value stream -> V(s): 1 scalar per quantile.
            self.value_stream = nn.Sequential(
                _lin(int(hidden_dim), int(hidden_dim)),
                nn.ReLU(),
                _lin(int(hidden_dim), self.n_quantiles),
            )
            # Advantage stream -> A(s, a): per-action per-quantile.
            self.advantage_stream = nn.Sequential(
                _lin(int(hidden_dim), int(hidden_dim)),
                nn.ReLU(),
                _lin(int(hidden_dim), eff_out),
            )
            self.head = None  # type: ignore[assignment]
        else:
            layers: list[nn.Module] = []
            layers.append(_lin(fc_in_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            for _ in range(int(hidden_layers) - 1):
                layers.append(_lin(int(hidden_dim), int(hidden_dim)))
                layers.append(nn.ReLU())
            layers.append(_lin(int(hidden_dim), eff_out))
            self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() != 2:
            raise ValueError("CNNQNetwork expects (batch, obs_dim) input")
        if int(x.shape[1]) != int(self.input_dim):
            raise ValueError(f"CNNQNetwork expected input_dim={self.input_dim}, got {int(x.shape[1])}")

        scalars = x[:, : self.scalar_dim]
        maps_flat = x[:, self.scalar_dim :]
        maps = maps_flat.reshape(int(x.shape[0]), self.map_channels, self.map_size, self.map_size)
        conv = self.conv(maps)                       # (B, 64, H', W')

        if self.cbam_mod is not None:
            conv = self.cbam_mod(conv)
        if self.spatial_mha is not None:
            conv = self.spatial_mha(conv)

        conv_flat = conv.flatten(start_dim=1)
        feats = torch.cat([scalars, conv_flat], dim=1)

        if self.dueling:
            shared_out = self.shared(feats)
            value = self.value_stream(shared_out)        # (B, n_quantiles)
            advantage = self.advantage_stream(shared_out)  # (B, n_actions * n_quantiles)
            if self.n_quantiles > 1:
                B = int(x.shape[0])
                adv = advantage.reshape(B, self.output_dim, self.n_quantiles)
                val = value.unsqueeze(1)                 # (B, 1, n_quantiles)
                q = val + adv - adv.mean(dim=1, keepdim=True)
                return q.reshape(B, self.output_dim * self.n_quantiles)
            return value + advantage - advantage.mean(dim=1, keepdim=True)

        return self.head(feats)

    def reset_noise(self) -> None:
        """Reset noise for all NoisyLinear sub-modules (no-op when noisy_net=False)."""
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()
