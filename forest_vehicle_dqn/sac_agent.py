"""SAC Agent with replay buffer and BC pretraining."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from forest_vehicle_dqn.sac_networks import (
    GlobalCNNEncoder, SACActor, SACCritic, SACEntropyCritic,
)


@dataclass
class SACConfig:
    map_size: int = 48
    map_channels: int = 3
    scalar_dim: int = 12
    action_dim: int = 2
    hidden_dim: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1_000_000
    target_entropy: float = -2.0  # -dim(action)
    reward_scale: float = 0.01  # scale raw rewards to stabilize Q-values
    grad_clip_norm: float = 1.0  # max gradient norm for critic/actor
    reward_clip_min: float = -500.0  # clip raw reward lower bound
    reward_clip_max: float = 1100.0  # clip raw reward upper bound
    target_q_min: float = -50.0  # clamp target Q lower bound
    target_q_max: float = 50.0   # clamp target Q upper bound
    use_huber_loss: bool = True   # use Huber (smooth_l1) instead of MSE for critic
    critic_warmup_steps: int = 2000  # critic-only updates before actor starts
    # TECRL (reward-entropy separation)
    use_tecrl: bool = False
    lr_entropy_critic: float = 3e-4
    entropy_budget_ratio: float = 0.6  # ρ: trajectory entropy budget
    alpha_min: float = 0.0  # minimum alpha (0 = no floor)


class SACReplayBuffer:
    """Simple replay buffer for dict observations (maps + scalars)."""

    def __init__(self, capacity: int, map_channels: int, map_size: int,
                 scalar_dim: int, action_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.maps = np.zeros(
            (capacity, map_channels, map_size, map_size), dtype=np.float32)
        self.scalars = np.zeros((capacity, scalar_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_maps = np.zeros(
            (capacity, map_channels, map_size, map_size), dtype=np.float32)
        self.next_scalars = np.zeros(
            (capacity, scalar_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, maps, scalars, action, reward, next_maps, next_scalars,
            done):
        i = self.ptr
        self.maps[i] = maps
        self.scalars[i] = scalars
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_maps[i] = next_maps
        self.next_scalars[i] = next_scalars
        self.dones[i] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idxs = rng.integers(0, self.size, size=batch_size)
        return (self.maps[idxs], self.scalars[idxs], self.actions[idxs],
                self.rewards[idxs], self.next_maps[idxs],
                self.next_scalars[idxs], self.dones[idxs])


class SACAgent:
    """Soft Actor-Critic agent for continuous control."""

    def __init__(self, config: SACConfig, *, device: str = "cpu",
                 seed: int = 0):
        self.cfg = config
        self.device = torch.device(device)
        self._rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        # Networks
        enc_actor = GlobalCNNEncoder(
            config.map_size, config.map_channels, config.scalar_dim)
        enc_critic = GlobalCNNEncoder(
            config.map_size, config.map_channels, config.scalar_dim)
        self.actor = SACActor(
            enc_actor, config.action_dim, config.hidden_dim).to(self.device)
        self.critic = SACCritic(
            enc_critic, config.action_dim, config.hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        # TECRL: entropy critic (estimates cumulative entropy)
        self.use_tecrl = config.use_tecrl
        if self.use_tecrl:
            enc_ent = GlobalCNNEncoder(
                config.map_size, config.map_channels, config.scalar_dim)
            self.entropy_critic = SACEntropyCritic(
                enc_ent, config.action_dim, config.hidden_dim
            ).to(self.device)
            self.entropy_critic_target = copy.deepcopy(self.entropy_critic)
            for p in self.entropy_critic_target.parameters():
                p.requires_grad_(False)
            self.entropy_critic_opt = torch.optim.Adam(
                self.entropy_critic.parameters(), lr=config.lr_entropy_critic)
            # H_budget = ρ * H_0 / (1 - γ), H_0 = -dim(action)
            self.entropy_budget = (
                config.entropy_budget_ratio * (-config.action_dim)
                / (1.0 - config.gamma)
            )

        # Automatic entropy tuning
        self.log_alpha = torch.zeros(
            1, requires_grad=True, device=self.device)
        self.target_entropy = config.target_entropy

        # Optimizers
        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=config.lr_actor)
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=config.lr_critic)
        self.alpha_opt = torch.optim.Adam(
            [self.log_alpha], lr=config.lr_alpha)

        # Replay buffer
        self.buffer = SACReplayBuffer(
            config.buffer_size, config.map_channels, config.map_size,
            config.scalar_dim, config.action_dim)

    @property
    def alpha(self) -> float:
        return self.log_alpha.exp().item()

    def act(self, obs: dict, *, explore: bool = True) -> np.ndarray:
        maps_t = torch.as_tensor(
            obs["maps"], dtype=torch.float32, device=self.device).unsqueeze(0)
        scalars_t = torch.as_tensor(
            obs["scalars"], dtype=torch.float32,
            device=self.device).unsqueeze(0)
        with torch.no_grad():
            if explore:
                action, _ = self.actor.sample(maps_t, scalars_t)
            else:
                action = self.actor.deterministic(maps_t, scalars_t)
        return action.cpu().numpy().squeeze(0)

    def observe(self, obs, action, reward, next_obs, done):
        reward = float(np.clip(reward, self.cfg.reward_clip_min,
                               self.cfg.reward_clip_max))
        self.buffer.add(
            obs["maps"], obs["scalars"], action, reward,
            next_obs["maps"], next_obs["scalars"], done)

    def update(self) -> Dict[str, float]:
        if self.use_tecrl:
            return self.update_tecrl()
        if self.buffer.size < self.cfg.batch_size:
            return {}
        maps, scalars, actions, rewards, n_maps, n_scalars, dones = \
            self.buffer.sample(self.cfg.batch_size, self._rng)

        maps_t = torch.as_tensor(maps, device=self.device)
        scalars_t = torch.as_tensor(scalars, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device)
        rewards_t = torch.as_tensor(rewards, device=self.device).unsqueeze(1)
        rewards_t = rewards_t * self.cfg.reward_scale  # scale rewards
        n_maps_t = torch.as_tensor(n_maps, device=self.device)
        n_scalars_t = torch.as_tensor(n_scalars, device=self.device)
        dones_t = torch.as_tensor(dones, device=self.device).unsqueeze(1)

        alpha = self.log_alpha.exp().detach()

        # --- Critic update ---
        with torch.no_grad():
            n_action, n_log_prob = self.actor.sample(n_maps_t, n_scalars_t)
            tq1, tq2 = self.critic_target(n_maps_t, n_scalars_t, n_action)
            target_q = torch.min(tq1, tq2) - alpha * n_log_prob.unsqueeze(1)
            target_q = rewards_t + (1 - dones_t) * self.cfg.gamma * target_q
            target_q = target_q.clamp(self.cfg.target_q_min,
                                      self.cfg.target_q_max)

        q1, q2 = self.critic(maps_t, scalars_t, actions_t)
        if self.cfg.use_huber_loss:
            critic_loss = (F.smooth_l1_loss(q1, target_q)
                           + F.smooth_l1_loss(q2, target_q))
        else:
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.cfg.grad_clip_norm)
        self.critic_opt.step()

        # --- Actor update ---
        new_action, log_prob = self.actor.sample(maps_t, scalars_t)
        q1_new, q2_new = self.critic(maps_t, scalars_t, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (alpha * log_prob.unsqueeze(1) - q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.cfg.grad_clip_norm)
        self.actor_opt.step()

        # --- Alpha update ---
        alpha_loss = -(self.log_alpha * (
            log_prob.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [self.log_alpha], self.cfg.grad_clip_norm)
        self.alpha_opt.step()
        if self.cfg.alpha_min > 0:
            with torch.no_grad():
                self.log_alpha.data.clamp_(min=math.log(self.cfg.alpha_min))

        # --- Soft target update ---
        with torch.no_grad():
            for tp, p in zip(self.critic_target.parameters(),
                             self.critic.parameters()):
                tp.data.mul_(1 - self.cfg.tau).add_(p.data, alpha=self.cfg.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha,
            "q1_mean": q1.mean().item(),
            "q2_mean": q2.mean().item(),
            "target_q_mean": target_q.mean().item(),
            "target_q_max": target_q.max().item(),
            "reward_batch_mean": rewards_t.mean().item(),
        }

    def update_tecrl(self) -> Dict[str, float]:
        """TECRL update: separate reward-critic and entropy-critic."""
        if self.buffer.size < self.cfg.batch_size:
            return {}
        maps, scalars, actions, rewards, n_maps, n_scalars, dones = \
            self.buffer.sample(self.cfg.batch_size, self._rng)

        maps_t = torch.as_tensor(maps, device=self.device)
        scalars_t = torch.as_tensor(scalars, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device)
        rewards_t = torch.as_tensor(rewards, device=self.device).unsqueeze(1)
        rewards_t = rewards_t * self.cfg.reward_scale
        n_maps_t = torch.as_tensor(n_maps, device=self.device)
        n_scalars_t = torch.as_tensor(n_scalars, device=self.device)
        dones_t = torch.as_tensor(dones, device=self.device).unsqueeze(1)
        gamma = self.cfg.gamma
        alpha = self.log_alpha.exp().detach()

        # --- 1. Reward critic update (no alpha in target) ---
        with torch.no_grad():
            n_action, n_log_prob = self.actor.sample(n_maps_t, n_scalars_t)
            tq1_r, tq2_r = self.critic_target(
                n_maps_t, n_scalars_t, n_action)
            target_q_r = rewards_t + (1 - dones_t) * gamma * torch.min(
                tq1_r, tq2_r)
            target_q_r = target_q_r.clamp(
                self.cfg.target_q_min, self.cfg.target_q_max)

        q1_r, q2_r = self.critic(maps_t, scalars_t, actions_t)
        if self.cfg.use_huber_loss:
            critic_loss = (F.smooth_l1_loss(q1_r, target_q_r)
                           + F.smooth_l1_loss(q2_r, target_q_r))
        else:
            critic_loss = (F.mse_loss(q1_r, target_q_r)
                           + F.mse_loss(q2_r, target_q_r))
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.cfg.grad_clip_norm)
        self.critic_opt.step()

        # --- 2. Entropy critic update (target = -log_prob + γ Q_e) ---
        with torch.no_grad():
            tq1_e, tq2_e = self.entropy_critic_target(
                n_maps_t, n_scalars_t, n_action)
            target_q_e = (-n_log_prob.unsqueeze(1)
                          + (1 - dones_t) * gamma * torch.min(tq1_e, tq2_e))

        q1_e, q2_e = self.entropy_critic(maps_t, scalars_t, actions_t)
        if self.cfg.use_huber_loss:
            ent_critic_loss = (F.smooth_l1_loss(q1_e, target_q_e)
                               + F.smooth_l1_loss(q2_e, target_q_e))
        else:
            ent_critic_loss = (F.mse_loss(q1_e, target_q_e)
                               + F.mse_loss(q2_e, target_q_e))
        self.entropy_critic_opt.zero_grad()
        ent_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.entropy_critic.parameters(), self.cfg.grad_clip_norm)
        self.entropy_critic_opt.step()

        # --- 3. Actor update: maximize Q_r + α * Q_e ---
        new_action, log_prob = self.actor.sample(maps_t, scalars_t)
        q1_r_new, q2_r_new = self.critic(maps_t, scalars_t, new_action)
        q_r_new = torch.min(q1_r_new, q2_r_new)
        q1_e_new, q2_e_new = self.entropy_critic(
            maps_t, scalars_t, new_action)
        q_e_new = torch.min(q1_e_new, q2_e_new)
        actor_loss = (-q_r_new - alpha * q_e_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.cfg.grad_clip_norm)
        self.actor_opt.step()

        # --- 4. Alpha update: trajectory entropy constraint ---
        alpha_loss = -(self.log_alpha * (
            q_e_new.detach() + log_prob.detach().unsqueeze(1)
            - self.entropy_budget)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [self.log_alpha], self.cfg.grad_clip_norm)
        self.alpha_opt.step()
        if self.cfg.alpha_min > 0:
            with torch.no_grad():
                self.log_alpha.data.clamp_(min=math.log(self.cfg.alpha_min))

        # --- 5. Soft target update (both critics) ---
        with torch.no_grad():
            for tp, p in zip(self.critic_target.parameters(),
                             self.critic.parameters()):
                tp.data.mul_(1 - self.cfg.tau).add_(
                    p.data, alpha=self.cfg.tau)
            for tp, p in zip(self.entropy_critic_target.parameters(),
                             self.entropy_critic.parameters()):
                tp.data.mul_(1 - self.cfg.tau).add_(
                    p.data, alpha=self.cfg.tau)

        return {
            "critic_loss": critic_loss.item(),
            "ent_critic_loss": ent_critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha,
            "q1_mean": q1_r.mean().item(),
            "q2_mean": q2_r.mean().item(),
            "q_e_mean": q_e_new.mean().item(),
            "target_q_r_mean": target_q_r.mean().item(),
            "target_q_mean": target_q_r.mean().item(),  # 兼容日志
            "target_q_max": target_q_r.max().item(),     # 兼容日志
            "reward_batch_mean": rewards_t.mean().item(),
        }

    def update_critic_only(self) -> Dict[str, float]:
        """Single critic update step (no actor/alpha update)."""
        if self.buffer.size < self.cfg.batch_size:
            return {}
        maps, scalars, actions, rewards, n_maps, n_scalars, dones = \
            self.buffer.sample(self.cfg.batch_size, self._rng)

        maps_t = torch.as_tensor(maps, device=self.device)
        scalars_t = torch.as_tensor(scalars, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device)
        rewards_t = torch.as_tensor(rewards, device=self.device).unsqueeze(1)
        rewards_t = rewards_t * self.cfg.reward_scale
        n_maps_t = torch.as_tensor(n_maps, device=self.device)
        n_scalars_t = torch.as_tensor(n_scalars, device=self.device)
        dones_t = torch.as_tensor(dones, device=self.device).unsqueeze(1)

        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            n_action, n_log_prob = self.actor.sample(n_maps_t, n_scalars_t)
            tq1, tq2 = self.critic_target(n_maps_t, n_scalars_t, n_action)
            target_q = torch.min(tq1, tq2) - alpha * n_log_prob.unsqueeze(1)
            target_q = rewards_t + (1 - dones_t) * self.cfg.gamma * target_q
            target_q = target_q.clamp(self.cfg.target_q_min,
                                      self.cfg.target_q_max)

        q1, q2 = self.critic(maps_t, scalars_t, actions_t)
        if self.cfg.use_huber_loss:
            critic_loss = (F.smooth_l1_loss(q1, target_q)
                           + F.smooth_l1_loss(q2, target_q))
        else:
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.cfg.grad_clip_norm)
        self.critic_opt.step()

        with torch.no_grad():
            for tp, p in zip(self.critic_target.parameters(),
                             self.critic.parameters()):
                tp.data.mul_(1 - self.cfg.tau).add_(p.data, alpha=self.cfg.tau)

        return {"critic_loss": critic_loss.item()}

    def warmup_critic(self, steps: int) -> None:
        """Run critic-only updates to bootstrap Q-estimates after BC."""
        for _ in range(steps):
            self.update_critic_only()

    def pretrain_bc(self, demos: list, *, steps: int = 5000):
        """Behavior cloning pretraining from expert demonstrations."""
        if not demos:
            return
        for step in range(steps):
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
            pred = self.actor.deterministic(maps_t, scalars_t)
            loss = F.mse_loss(pred, target_t)
            self.actor_opt.zero_grad()
            loss.backward()
            self.actor_opt.step()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.data,
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(),
        }
        if self.use_tecrl:
            state["entropy_critic"] = self.entropy_critic.state_dict()
            state["entropy_critic_target"] = \
                self.entropy_critic_target.state_dict()
            state["entropy_critic_opt"] = \
                self.entropy_critic_opt.state_dict()
        torch.save(state, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.log_alpha.data.copy_(ckpt["log_alpha"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        self.alpha_opt.load_state_dict(ckpt["alpha_opt"])
        if self.use_tecrl and "entropy_critic" in ckpt:
            self.entropy_critic.load_state_dict(ckpt["entropy_critic"])
            self.entropy_critic_target.load_state_dict(
                ckpt["entropy_critic_target"])
            self.entropy_critic_opt.load_state_dict(
                ckpt["entropy_critic_opt"])
