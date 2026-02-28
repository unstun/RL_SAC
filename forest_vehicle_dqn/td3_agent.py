"""TD3 (Twin Delayed DDPG) Agent — deterministic continuous control."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from forest_vehicle_dqn.sac_agent import SACReplayBuffer
from forest_vehicle_dqn.sac_networks import GlobalCNNEncoder, SACCritic
from forest_vehicle_dqn.td3_networks import TD3Actor, LocalCNNEncoder


@dataclass
class TD3Config:
    map_size: int = 48
    map_channels: int = 3
    scalar_dim: int = 12
    action_dim: int = 2
    hidden_dim: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    lr_actor: float = 1e-3
    lr_critic: float = 1e-3
    batch_size: int = 256
    buffer_size: int = 1_000_000
    # TD3-specific
    explore_noise: float = 0.1
    target_noise: float = 0.2
    target_noise_clip: float = 0.5
    policy_delay: int = 2
    # Stability (reuse SAC defaults)
    reward_scale: float = 0.01
    grad_clip_norm: float = 1.0
    reward_clip_min: float = -500.0
    reward_clip_max: float = 1100.0
    target_q_min: float = -50.0
    target_q_max: float = 50.0
    use_huber_loss: bool = True
    critic_warmup_steps: int = 2000
    # Local obs mode (V16-C style: 12x12 + 10 scalars)
    use_local_encoder: bool = False


class TD3Agent:
    """Twin Delayed DDPG agent for continuous control."""

    def __init__(self, config: TD3Config, *, device: str = "cpu",
                 seed: int = 0):
        self.cfg = config
        self.device = torch.device(device)
        self._rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self._update_step = 0

        # Actor + target actor
        if config.use_local_encoder:
            enc_a = LocalCNNEncoder(
                config.map_size, config.map_channels, config.scalar_dim)
            enc_c = LocalCNNEncoder(
                config.map_size, config.map_channels, config.scalar_dim)
        else:
            enc_a = GlobalCNNEncoder(
                config.map_size, config.map_channels, config.scalar_dim)
            enc_c = GlobalCNNEncoder(
                config.map_size, config.map_channels, config.scalar_dim)
        self.actor = TD3Actor(
            enc_a, config.action_dim, config.hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        for p in self.actor_target.parameters():
            p.requires_grad_(False)

        # Critic + target critic (reuse SACCritic = twin Q)
        self.critic = SACCritic(
            enc_c, config.action_dim, config.hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        # Optimizers
        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=config.lr_actor)
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=config.lr_critic)

        # Replay buffer (reuse SAC's)
        self.buffer = SACReplayBuffer(
            config.buffer_size, config.map_channels, config.map_size,
            config.scalar_dim, config.action_dim)

    def act(self, obs: dict, *, explore: bool = True) -> np.ndarray:
        maps_t = torch.as_tensor(
            obs["maps"], dtype=torch.float32, device=self.device).unsqueeze(0)
        scalars_t = torch.as_tensor(
            obs["scalars"], dtype=torch.float32,
            device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(maps_t, scalars_t).cpu().numpy().squeeze(0)
        if explore:
            noise = self._rng.normal(0, self.cfg.explore_noise,
                                     size=action.shape).astype(np.float32)
            action = np.clip(action + noise, -1.0, 1.0)
        return action

    def observe(self, obs, action, reward, next_obs, done):
        reward = float(np.clip(reward, self.cfg.reward_clip_min,
                               self.cfg.reward_clip_max))
        self.buffer.add(
            obs["maps"], obs["scalars"], action, reward,
            next_obs["maps"], next_obs["scalars"], done)

    def update(self) -> Dict[str, float]:
        if self.buffer.size < self.cfg.batch_size:
            return {}
        self._update_step += 1
        cfg = self.cfg

        maps, scalars, actions, rewards, n_maps, n_scalars, dones = \
            self.buffer.sample(cfg.batch_size, self._rng)

        maps_t = torch.as_tensor(maps, device=self.device)
        scalars_t = torch.as_tensor(scalars, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device)
        rewards_t = torch.as_tensor(
            rewards, device=self.device).unsqueeze(1) * cfg.reward_scale
        n_maps_t = torch.as_tensor(n_maps, device=self.device)
        n_scalars_t = torch.as_tensor(n_scalars, device=self.device)
        dones_t = torch.as_tensor(dones, device=self.device).unsqueeze(1)

        # --- Critic update (every step) ---
        with torch.no_grad():
            # Target policy smoothing
            n_action = self.actor_target(n_maps_t, n_scalars_t)
            noise = torch.randn_like(n_action) * cfg.target_noise
            noise = noise.clamp(-cfg.target_noise_clip, cfg.target_noise_clip)
            n_action = (n_action + noise).clamp(-1.0, 1.0)
            tq1, tq2 = self.critic_target(n_maps_t, n_scalars_t, n_action)
            target_q = rewards_t + (1 - dones_t) * cfg.gamma * torch.min(
                tq1, tq2)
            target_q = target_q.clamp(cfg.target_q_min, cfg.target_q_max)

        q1, q2 = self.critic(maps_t, scalars_t, actions_t)
        if cfg.use_huber_loss:
            critic_loss = (F.smooth_l1_loss(q1, target_q)
                           + F.smooth_l1_loss(q2, target_q))
        else:
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), cfg.grad_clip_norm)
        self.critic_opt.step()

        info: Dict[str, float] = {
            "critic_loss": critic_loss.item(),
            "q1_mean": q1.mean().item(),
            "q2_mean": q2.mean().item(),
            "target_q_mean": target_q.mean().item(),
            "target_q_max": target_q.max().item(),
            "reward_batch_mean": rewards_t.mean().item(),
        }

        # --- Delayed actor + target update (every D steps) ---
        if self._update_step % cfg.policy_delay == 0:
            new_action = self.actor(maps_t, scalars_t)
            q1_new, _ = self.critic(maps_t, scalars_t, new_action)
            actor_loss = -q1_new.mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), cfg.grad_clip_norm)
            self.actor_opt.step()
            info["actor_loss"] = actor_loss.item()

            # Soft target update (both actor and critic)
            with torch.no_grad():
                for tp, p in zip(self.actor_target.parameters(),
                                 self.actor.parameters()):
                    tp.data.mul_(1 - cfg.tau).add_(p.data, alpha=cfg.tau)
                for tp, p in zip(self.critic_target.parameters(),
                                 self.critic.parameters()):
                    tp.data.mul_(1 - cfg.tau).add_(p.data, alpha=cfg.tau)

        return info

    def update_critic_only(self) -> Dict[str, float]:
        """Single critic update (no actor/target update)."""
        if self.buffer.size < self.cfg.batch_size:
            return {}
        cfg = self.cfg
        maps, scalars, actions, rewards, n_maps, n_scalars, dones = \
            self.buffer.sample(cfg.batch_size, self._rng)

        maps_t = torch.as_tensor(maps, device=self.device)
        scalars_t = torch.as_tensor(scalars, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device)
        rewards_t = torch.as_tensor(
            rewards, device=self.device).unsqueeze(1) * cfg.reward_scale
        n_maps_t = torch.as_tensor(n_maps, device=self.device)
        n_scalars_t = torch.as_tensor(n_scalars, device=self.device)
        dones_t = torch.as_tensor(dones, device=self.device).unsqueeze(1)

        with torch.no_grad():
            n_action = self.actor_target(n_maps_t, n_scalars_t)
            noise = torch.randn_like(n_action) * cfg.target_noise
            noise = noise.clamp(-cfg.target_noise_clip, cfg.target_noise_clip)
            n_action = (n_action + noise).clamp(-1.0, 1.0)
            tq1, tq2 = self.critic_target(n_maps_t, n_scalars_t, n_action)
            target_q = rewards_t + (1 - dones_t) * cfg.gamma * torch.min(
                tq1, tq2)
            target_q = target_q.clamp(cfg.target_q_min, cfg.target_q_max)

        q1, q2 = self.critic(maps_t, scalars_t, actions_t)
        if cfg.use_huber_loss:
            critic_loss = (F.smooth_l1_loss(q1, target_q)
                           + F.smooth_l1_loss(q2, target_q))
        else:
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), cfg.grad_clip_norm)
        self.critic_opt.step()

        with torch.no_grad():
            for tp, p in zip(self.critic_target.parameters(),
                             self.critic.parameters()):
                tp.data.mul_(1 - cfg.tau).add_(p.data, alpha=cfg.tau)
        return {"critic_loss": critic_loss.item()}

    def warmup_critic(self, steps: int) -> None:
        for _ in range(steps):
            self.update_critic_only()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])

    # ------------------------------------------------------------------
    # Flat obs helpers (for V16-C local obs: flat 154-dim → dict)
    # ------------------------------------------------------------------

    def _flat_to_dict(self, flat_obs: np.ndarray) -> dict:
        """Convert flat 154-dim DQN obs to dict expected by actor/critic."""
        sd = self.cfg.scalar_dim
        ms = self.cfg.map_size
        mc = self.cfg.map_channels
        scalars = flat_obs[:sd]
        maps = flat_obs[sd:].reshape(mc, ms, ms)
        return {"maps": maps, "scalars": scalars}

    def act_flat(self, flat_obs: np.ndarray, *,
                 explore: bool = True) -> np.ndarray:
        """act() variant accepting flat obs array (V16-C format)."""
        return self.act(self._flat_to_dict(flat_obs), explore=explore)

    def observe_flat(self, flat_obs: np.ndarray, action: np.ndarray,
                     reward: float, next_flat_obs: np.ndarray,
                     done: bool) -> None:
        """observe() variant accepting flat obs arrays (V16-C format)."""
        self.observe(self._flat_to_dict(flat_obs), action,
                     reward, self._flat_to_dict(next_flat_obs), done)

    def pretrain_bc(self, demos: list, *, steps: int = 5000) -> None:
        """Behaviour cloning pretraining from expert demos (dict format)."""
        if not demos:
            return
        for _ in range(steps):
            demo = demos[self._rng.integers(0, len(demos))]
            maps_t = torch.as_tensor(
                demo["maps"], dtype=torch.float32,
                device=self.device).unsqueeze(0)
            scalars_t = torch.as_tensor(
                demo["scalars"], dtype=torch.float32,
                device=self.device).unsqueeze(0)
            target_t = torch.as_tensor(
                demo["action"], dtype=torch.float32,
                device=self.device).unsqueeze(0)
            pred = self.actor(maps_t, scalars_t)
            loss = F.mse_loss(pred, target_t)
            self.actor_opt.zero_grad()
            loss.backward()
            self.actor_opt.step()
