from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from forest_vehicle_dqn.networks import CNNQNetwork, MLPQNetwork, infer_flat_obs_cnn_layout
from forest_vehicle_dqn.replay_buffer import ReplayBuffer
from forest_vehicle_dqn.schedules import linear_epsilon


@dataclass(frozen=True)
class AgentConfig:
    # Forest (dt=0.05s, long horizon) needs a higher discount factor; gamma=0.9 makes
    # the terminal reward effectively vanish after a few dozen steps.
    gamma: float = 0.995
    n_step: int = 1
    learning_rate: float = 5e-4
    replay_capacity: int = 100_000
    batch_size: int = 128
    target_update_steps: int = 1000
    # When >0, use Polyak (soft) target updates at every training step:
    #   target = (1 - tau) * target + tau * online
    # When 0, use periodic hard updates every `target_update_steps`.
    target_update_tau: float = 0.0
    grad_clip_norm: float = 10.0

    eps_start: float = 0.9
    eps_final: float = 0.01
    eps_decay: int = 2000

    hidden_layers: int = 3
    hidden_dim: int = 256

    # Dueling DQN (Wang et al., 2016): split Q into V(s) + A(s,a) - mean(A).
    dueling: bool = False
    # CBAM attention on CNN feature maps (Woo et al., ECCV 2018).
    cbam: bool = False
    # NoisyNet (Fortunato et al., ICLR 2018): replace Linear with NoisyLinear.
    noisy_net: bool = False
    # Spatial multi-head self-attention on CNN feature maps (Vaswani et al., 2017).
    mha: bool = False
    mha_heads: int = 4
    # QR-DQN (Dabney et al., AAAI 2018): 1 = standard DQN, >1 = quantile regression.
    n_quantiles: int = 1

    # Expert margin loss (DQfD-style) for forest stabilization.
    demo_margin: float = 0.8
    demo_lambda: float = 1.0
    demo_ce_lambda: float = 1.0

    # Replay sampling: optionally oversample rare-but-important transitions (forest long-horizon).
    replay_stratified: bool = False
    replay_prioritized: bool = False
    per_alpha: float = 0.4
    per_beta0: float = 0.6
    per_beta_steps: int = 100_000
    per_eps_agent: float = 1e-3
    per_eps_demo: float = 1.0
    per_boost_near_goal: float = 0.0
    per_boost_stuck: float = 0.0
    per_boost_hazard: float = 0.0
    replay_frac_demo: float = 0.0
    replay_frac_goal: float = 0.0
    replay_frac_stuck: float = 0.0
    replay_frac_hazard: float = 0.0

    # Demo training mode:
    # - "legacy": previous stabilizer (n-step TD + margin + optional CE) + optional stratified replay
    # - "dqfd": strict DQfD-style (PER + 1-step TD + n-step TD + margin + L2), no CE
    demo_mode: str = "dqfd"
    dqfd_lambda_n: float = 1.0
    l2_reg: float = 1e-5
    # Training-only admissibility auxiliary supervision (BCE on per-action admissibility mask).
    # Inference remains pure argmax(Q) and never consults this head.
    aux_admissibility_lambda: float = 0.0
    # Munchausen DQN (Vieillard et al., NeurIPS 2020): soft backup + log-policy
    # reward augmentation.  Only active when base_algo == "mdqn".
    mdqn_tau: float = 0.03    # softmax temperature
    mdqn_alpha: float = 0.9   # log-policy scaling
    mdqn_l0: float = -1.0     # log-policy clipping lower bound


AlgoArch = Literal["mlp", "cnn", "cnn-dual", "cnn-drqn", "cnn-hist"]
AlgoBase = Literal["dqn", "ddqn", "mdqn"]


def parse_rl_algo(algo: str) -> tuple[str, AlgoArch, AlgoBase, bool]:
    """Return (canonical_name, arch, base_algo, is_legacy_alias)."""

    a = str(algo).lower().strip()
    if a in {"dqn", "ddqn"}:
        base: AlgoBase = "dqn" if a == "dqn" else "ddqn"
        return (f"mlp-{base}", "mlp", base, True)

    if a == "iddqn":
        # Legacy name (avoid collision with published "IDDQN").
        # This project uses it for a Polyak/soft-target Double DQN variant.
        return ("mlp-pddqn", "mlp", "ddqn", True)

    if a == "cnn-iddqn":
        # Legacy name (avoid collision with published "IDDQN").
        return ("cnn-pddqn", "cnn", "ddqn", True)

    supported = {
        "mlp-dqn", "mlp-ddqn", "mlp-pddqn", "mlp-mdqn",
        "cnn-dqn", "cnn-ddqn", "cnn-pddqn", "cnn-mdqn",
    }
    if a in supported:
        arch_s, variant = a.split("-", 1)
        arch: AlgoArch = "mlp" if arch_s == "mlp" else "cnn"
        if variant == "dqn":
            base: AlgoBase = "dqn"
        elif variant == "mdqn":
            base = "mdqn"
        else:
            base = "ddqn"
        return (a, arch, base, False)

    if a == "cnn-dual-ddqn":
        return ("cnn-dual-ddqn", "cnn-dual", "ddqn", False)
    if a == "cnn-drqn":
        return ("cnn-drqn", "cnn-drqn", "dqn", False)
    if a == "cnn-histformer":
        return ("cnn-histformer", "cnn-hist", "dqn", False)

    raise ValueError(
        "algo must be one of: mlp-dqn mlp-ddqn mlp-pddqn mlp-mdqn "
        "cnn-dqn cnn-ddqn cnn-pddqn cnn-mdqn "
        "(legacy: dqn ddqn iddqn cnn-iddqn)"
    )


class DQNFamilyAgent:
    def __init__(
        self,
        algo: str,
        obs_dim: int,
        n_actions: int,
        *,
        config: AgentConfig,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        canonical_algo, arch, base_algo, _legacy = parse_rl_algo(algo)
        self.algo = canonical_algo
        self.arch = arch
        self.base_algo = base_algo
        self.config = config
        self.device = torch.device(device)

        self._rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        # DQN is the baseline (plain Q-learning).
        # DDQN keeps the same architecture but uses the Double DQN TD target (online argmax + target eval).
        self._net_cls: type[nn.Module]
        self._net_kwargs: dict[str, object]
        if self.arch == "cnn":
            layout = infer_flat_obs_cnn_layout(int(obs_dim))
            self._net_cls = CNNQNetwork
            self._net_kwargs = {
                "scalar_dim": int(layout.scalar_dim),
                "map_channels": int(layout.map_channels),
                "map_size": int(layout.map_size),
                "dueling": bool(getattr(config, "dueling", False)),
                "cbam": bool(getattr(config, "cbam", False)),
                "noisy_net": bool(getattr(config, "noisy_net", False)),
                "mha": bool(getattr(config, "mha", False)),
                "mha_heads": int(getattr(config, "mha_heads", 4)),
                "n_quantiles": int(getattr(config, "n_quantiles", 1)),
            }
        elif self.arch == "cnn-dual":
            from forest_vehicle_dqn.networks import DualScaleCNNQNetwork
            self._net_cls = DualScaleCNNQNetwork
            self._net_kwargs = {}
        else:
            self._net_cls = MLPQNetwork
            self._net_kwargs = {}

        self.q = self._net_cls(obs_dim, n_actions, hidden_dim=config.hidden_dim, hidden_layers=config.hidden_layers, **self._net_kwargs).to(self.device)
        self.q_target = self._net_cls(obs_dim, n_actions, hidden_dim=config.hidden_dim, hidden_layers=config.hidden_layers, **self._net_kwargs).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()

        aux_lambda = float(getattr(config, "aux_admissibility_lambda", 0.0))
        self.aux_adm_head: nn.Linear | None = None
        if self.arch == "cnn" and aux_lambda > 0.0:
            self.aux_adm_head = nn.Linear(int(n_actions), int(n_actions)).to(self.device)

        self.optimizer = torch.optim.Adam(self._optimizer_params(), lr=config.learning_rate)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")

        self.replay = ReplayBuffer(
            config.replay_capacity,
            obs_dim,
            n_actions,
            rng=self._rng,
            stratified=bool(getattr(config, "replay_stratified", False)),
            prioritized=bool(getattr(config, "replay_prioritized", False)),
            per_alpha=float(getattr(config, "per_alpha", 0.4)),
            per_eps_agent=float(getattr(config, "per_eps_agent", 1e-3)),
            per_eps_demo=float(getattr(config, "per_eps_demo", 1.0)),
            per_boost_near_goal=float(getattr(config, "per_boost_near_goal", 0.0)),
            per_boost_stuck=float(getattr(config, "per_boost_stuck", 0.0)),
            per_boost_hazard=float(getattr(config, "per_boost_hazard", 0.0)),
            frac_demo=float(getattr(config, "replay_frac_demo", 0.0)),
            frac_goal=float(getattr(config, "replay_frac_goal", 0.0)),
            frac_stuck=float(getattr(config, "replay_frac_stuck", 0.0)),
            frac_hazard=float(getattr(config, "replay_frac_hazard", 0.0)),
        )

        self._train_steps = 0
        self._n_actions = int(n_actions)
        self._obs_dim = int(obs_dim)
        self._n_step = int(max(1, int(getattr(config, "n_step", 1))))
        # (obs_t, action_t, reward_t, next_obs_{t+1}, done_{t+1}, demo, next_action_mask_{t+1}, replay_flags)
        self._nstep_buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool, bool, np.ndarray | None, int]] = deque()

    def _optimizer_params(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = list(self.q.parameters())
        if self.aux_adm_head is not None:
            params.extend(list(self.aux_adm_head.parameters()))
        return params

    def _demo_mode(self) -> str:
        mode = str(getattr(self.config, "demo_mode", "legacy")).lower().strip()
        if mode not in {"legacy", "dqfd"}:
            raise ValueError("demo_mode must be one of: legacy, dqfd")
        return mode

    def _per_beta(self) -> float:
        beta0 = float(getattr(self.config, "per_beta0", 0.0))
        beta0 = float(np.clip(beta0, 0.0, 1.0))
        beta_steps = int(getattr(self.config, "per_beta_steps", 1))
        if beta_steps <= 0:
            return 1.0
        t = float(np.clip(float(self._train_steps) / float(max(1, int(beta_steps))), 0.0, 1.0))
        return float(beta0 + (1.0 - beta0) * t)

    def _quantile_td_loss(
        self,
        q_quantiles: torch.Tensor,
        target_quantiles: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Quantile Huber loss for QR-DQN (Dabney et al., AAAI 2018).

        Args:
            q_quantiles: (B, N) predicted quantiles for chosen action.
            target_quantiles: (B, N) target quantiles.
            weights: (B,) optional importance-sampling weights.
        Returns:
            Scalar loss.
        """
        nq = q_quantiles.shape[1]
        tau = (torch.arange(nq, device=q_quantiles.device, dtype=torch.float32) + 0.5) / float(nq)
        # (B, N_pred, N_target)
        diff = target_quantiles.unsqueeze(1) - q_quantiles.unsqueeze(2)
        huber = F.smooth_l1_loss(
            q_quantiles.unsqueeze(2).expand_as(diff),
            target_quantiles.unsqueeze(1).expand_as(diff),
            reduction="none",
        )
        qw = (tau.view(1, -1, 1) - (diff < 0).float()).abs()
        loss_per_sample = (qw * huber).sum(dim=2).mean(dim=1)  # (B,)
        if weights is not None:
            loss_per_sample = loss_per_sample * weights
        return loss_per_sample.mean()

    def _next_q_quantiles(
        self,
        next_obs: torch.Tensor,
        mask: torch.Tensor,
        nq: int,
    ) -> torch.Tensor:
        """Compute target quantiles for next state (QR-DQN helper).

        Returns: (B, nq) quantiles for greedy/DDQN next action.
        """
        B = int(next_obs.shape[0])
        raw_target = self.q_target(next_obs).reshape(B, self._n_actions, nq)
        q_mean_target = raw_target.mean(dim=2)
        q_mean_target = q_mean_target.masked_fill(~mask, torch.finfo(q_mean_target.dtype).min)

        if self.base_algo == "ddqn":
            raw_online = self.q(next_obs).reshape(B, self._n_actions, nq)
            q_mean_online = raw_online.mean(dim=2)
            q_mean_online = q_mean_online.masked_fill(~mask, torch.finfo(q_mean_online.dtype).min)
            next_action = q_mean_online.argmax(dim=1)
        else:
            next_action = q_mean_target.argmax(dim=1)

        tgt = raw_target.gather(1, next_action.view(-1, 1, 1).expand(-1, -1, nq)).squeeze(1)
        return torch.where(torch.isfinite(tgt), tgt, torch.zeros_like(tgt))

    def _rebuild_networks(
        self,
        net_cls: type[nn.Module],
        *,
        hidden_dim: int,
        hidden_layers: int,
        net_kwargs: dict[str, object] | None = None,
    ) -> None:
        net_kwargs = {} if net_kwargs is None else dict(net_kwargs)
        self.q = net_cls(self._obs_dim, self._n_actions, hidden_dim=hidden_dim, hidden_layers=hidden_layers, **net_kwargs).to(self.device)
        self.q_target = net_cls(self._obs_dim, self._n_actions, hidden_dim=hidden_dim, hidden_layers=hidden_layers, **net_kwargs).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()
        aux_lambda = float(getattr(self.config, "aux_admissibility_lambda", 0.0))
        if self.arch == "cnn" and aux_lambda > 0.0:
            self.aux_adm_head = nn.Linear(int(self._n_actions), int(self._n_actions)).to(self.device)
        else:
            self.aux_adm_head = None
        self.optimizer = torch.optim.Adam(self._optimizer_params(), lr=self.config.learning_rate)

    def epsilon(self, episode: int) -> float:
        return linear_epsilon(
            episode,
            eps_start=self.config.eps_start,
            eps_final=self.config.eps_final,
            decay_episodes=self.config.eps_decay,
        )

    def _q_values(self, raw: torch.Tensor) -> torch.Tensor:
        """Convert raw network output to Q-values, handling QR-DQN quantiles."""
        nq = int(getattr(self.config, "n_quantiles", 1))
        if nq > 1:
            # raw: (..., n_actions * n_quantiles) -> mean over quantiles
            shape = raw.shape[:-1] + (self._n_actions, nq)
            return raw.reshape(shape).mean(dim=-1)
        return raw

    def act(self, obs: np.ndarray, *, episode: int, explore: bool = True) -> int:
        # NoisyNet provides exploration via parameter noise; skip ε-greedy.
        use_noisy = bool(getattr(self.config, "noisy_net", False))
        if explore and not use_noisy and (self._rng.random() < self.epsilon(episode)):
            return int(self._rng.integers(0, self._n_actions))

        with torch.no_grad():
            x = torch.from_numpy(obs.astype(np.float32, copy=False)).to(self.device)
            q = self._q_values(self.q(x.unsqueeze(0))).squeeze(0)
            return int(torch.argmax(q).item())

    def act_masked(
        self,
        obs: np.ndarray,
        *,
        episode: int,
        explore: bool = True,
        action_mask: np.ndarray | None = None,
    ) -> int:
        """Epsilon-greedy action selection with an optional boolean action mask."""

        mask = None
        if action_mask is not None:
            mask = np.asarray(action_mask, dtype=bool).reshape(-1)
            if mask.size != self._n_actions:
                raise ValueError("action_mask must have shape (n_actions,)")

        use_noisy = bool(getattr(self.config, "noisy_net", False))
        if explore and not use_noisy and (self._rng.random() < self.epsilon(episode)):
            if mask is None:
                return int(self._rng.integers(0, self._n_actions))
            idxs = np.nonzero(mask)[0]
            if idxs.size == 0:
                return int(self._rng.integers(0, self._n_actions))
            return int(self._rng.choice(idxs))

        with torch.no_grad():
            x = torch.from_numpy(obs.astype(np.float32, copy=False)).to(self.device)
            q = self._q_values(self.q(x.unsqueeze(0))).squeeze(0)
            if mask is not None:
                q = q.clone()
                q[torch.from_numpy(~mask).to(self.device)] = torch.finfo(q.dtype).min
            return int(torch.argmax(q).item())

    def top_actions(self, obs: np.ndarray, *, k: int) -> np.ndarray:
        """Return the top-k action indices by Q-value (descending)."""
        kk = int(max(1, int(k)))
        with torch.no_grad():
            x = torch.from_numpy(obs.astype(np.float32, copy=False)).to(self.device)
            q = self._q_values(self.q(x.unsqueeze(0))).squeeze(0)
            kk = int(min(int(kk), int(q.numel())))
            return torch.topk(q, k=kk, dim=0).indices.detach().cpu().numpy()

    def _add_to_replay(
        self,
        obs: np.ndarray,
        action: int,
        reward_1: float,
        next_obs_1: np.ndarray,
        done_1: bool,
        *,
        next_action_mask_1: np.ndarray | None,
        reward_n: float,
        next_obs_n: np.ndarray,
        done_n: bool,
        next_action_mask_n: np.ndarray | None,
        demo: bool,
        n_steps: int,
        replay_flags: int,
    ) -> None:
        self.replay.add(
            obs,
            int(action),
            float(reward_1),
            next_obs_1,
            bool(done_1),
            next_action_mask_1=next_action_mask_1,
            reward_n=float(reward_n),
            next_obs_n=next_obs_n,
            done_n=bool(done_n),
            next_action_mask_n=next_action_mask_n,
            demo=bool(demo),
            n_steps_n=int(n_steps),
            flags=int(replay_flags),
        )

    def observe(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        *,
        demo: bool = False,
        truncated: bool = False,
        next_action_mask: np.ndarray | None = None,
        replay_flags: int = 0,
    ) -> None:
        """Record a transition (supports n-step returns).

        `done` should reflect *true* terminal states (collision/reached). Use `truncated=True`
        for time-limit episode ends so the n-step buffer does not leak across episodes while
        still allowing bootstrapping from the final state.
        """

        if int(self._n_step) <= 1:
            self._add_to_replay(
                obs,
                int(action),
                float(reward),
                next_obs,
                bool(done),
                next_action_mask_1=next_action_mask,
                reward_n=float(reward),
                next_obs_n=next_obs,
                done_n=bool(done),
                next_action_mask_n=next_action_mask,
                demo=bool(demo),
                n_steps=1,
                replay_flags=int(replay_flags),
            )
            if bool(done) or bool(truncated):
                self.end_episode()
            return

        self._nstep_buffer.append(
            (obs, int(action), float(reward), next_obs, bool(done), bool(demo), next_action_mask, int(replay_flags))
        )

        episode_end = bool(done) or bool(truncated)
        if (len(self._nstep_buffer) < int(self._n_step)) and not episode_end:
            return

        self._flush_nstep_buffer(force=episode_end)

    def _flush_nstep_buffer(self, *, force: bool) -> None:
        if int(self._n_step) <= 1:
            return

        gamma = float(self.config.gamma)
        n_target = int(self._n_step)

        # When `force` is True (episode boundary), flush everything with truncated n-step horizons.
        while self._nstep_buffer:
            horizon = min(int(n_target), int(len(self._nstep_buffer)))
            ret = 0.0
            n_used = 0
            # 1-step info (always available at the first element).
            obs1, a1, r1, next_obs_1, done_1, demo1, next_mask_1, flags1 = self._nstep_buffer[0]
            next_obs_n = self._nstep_buffer[0][3]
            next_mask_n = self._nstep_buffer[0][6]
            done_n = False
            flags_n = 0
            for i in range(horizon):
                _o, _a, r, no, d, _demo, nm, fl = self._nstep_buffer[i]
                ret += (gamma**i) * float(r)
                n_used = i + 1
                next_obs_n = no
                next_mask_n = nm
                done_n = bool(d)
                flags_n |= int(fl)
                if done_n:
                    break

            self._add_to_replay(
                obs1,
                int(a1),
                float(r1),
                next_obs_1,
                bool(done_1),
                next_action_mask_1=next_mask_1,
                reward_n=float(ret),
                next_obs_n=next_obs_n,
                done_n=bool(done_n),
                next_action_mask_n=next_mask_n,
                demo=bool(demo1),
                n_steps=int(n_used),
                replay_flags=int(flags1 | flags_n),
            )
            self._nstep_buffer.popleft()

            # If we're not at an episode boundary, only emit one transition per step.
            if not bool(force):
                break

    def end_episode(self) -> None:
        """Flush any pending n-step transitions at an episode boundary."""
        if int(self._n_step) <= 1:
            return
        self._flush_nstep_buffer(force=True)

    def pretrain_on_demos(self, *, steps: int) -> int:
        """Pretrain on demonstration transitions.

        - legacy: supervised warm-start (CE + margin) on demo transitions only.
        - dqfd: strict DQfD-style pretraining by running TD updates with PER on the demo buffer.
        """

        n_steps = int(steps)
        if n_steps <= 0:
            return 0
        if len(self.replay) < int(self.config.batch_size):
            return 0

        if self._demo_mode() == "dqfd":
            trained = 0
            for _ in range(n_steps):
                out = self.update()
                if not out:
                    break
                trained += 1
            return int(trained)

        demo_lambda = float(getattr(self.config, "demo_lambda", 0.0))
        demo_margin = float(getattr(self.config, "demo_margin", 0.0))
        demo_ce_lambda = float(getattr(self.config, "demo_ce_lambda", 0.0))

        trained = 0
        for _ in range(n_steps):
            if len(self.replay) < int(self.config.batch_size):
                break

            batch = self.replay.sample(self.config.batch_size)
            obs = torch.from_numpy(batch.obs).to(self.device)
            actions = torch.from_numpy(batch.actions).to(self.device)
            demos = torch.from_numpy(batch.demos).to(self.device)

            demo_mask = demos.float().clamp(0.0, 1.0) > 0.0
            if not bool(torch.any(demo_mask)):
                continue

            q_all = self.q(obs)
            q_demo = q_all[demo_mask]
            a_demo = actions.long()[demo_mask]

            loss = torch.tensor(0.0, device=self.device)
            if demo_ce_lambda > 0.0:
                loss = loss + float(demo_ce_lambda) * F.cross_entropy(q_demo, a_demo, reduction="mean")

            if demo_lambda > 0.0 and demo_margin > 0.0:
                q_a = q_demo.gather(1, a_demo.view(-1, 1)).squeeze(1)
                q_other = q_demo.clone()
                q_other.scatter_(1, a_demo.view(-1, 1), torch.finfo(q_other.dtype).min)
                q_max_other = q_other.max(dim=1).values
                margin = torch.relu(q_max_other + float(demo_margin) - q_a).mean()
                loss = loss + float(demo_lambda) * margin

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.config.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(self._optimizer_params(), max_norm=self.config.grad_clip_norm)
            self.optimizer.step()
            trained += 1

        if trained > 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return int(trained)

    def update(self) -> dict[str, float]:
        if len(self.replay) < self.config.batch_size:
            return {}

        # NoisyNet: refresh noise before each update.
        if bool(getattr(self.config, "noisy_net", False)) and hasattr(self.q, "reset_noise"):
            self.q.reset_noise()
            self.q_target.reset_noise()

        mode = self._demo_mode()
        if mode == "dqfd":
            batch = self.replay.sample(self.config.batch_size, beta=float(self._per_beta()))
        else:
            batch = self.replay.sample(self.config.batch_size)
        obs = torch.from_numpy(batch.obs).to(self.device)
        actions = torch.from_numpy(batch.actions).to(self.device)
        rewards_1 = torch.from_numpy(batch.rewards_1).to(self.device)
        next_obs_1 = torch.from_numpy(batch.next_obs_1).to(self.device)
        next_action_masks_1 = torch.from_numpy(batch.next_action_masks_1).to(self.device)
        dones_1 = torch.from_numpy(batch.dones_1).to(self.device)

        rewards_n = torch.from_numpy(batch.rewards_n).to(self.device)
        next_obs_n = torch.from_numpy(batch.next_obs_n).to(self.device)
        next_action_masks_n = torch.from_numpy(batch.next_action_masks_n).to(self.device)
        dones_n = torch.from_numpy(batch.dones_n).to(self.device)
        n_steps_n = torch.from_numpy(batch.n_steps_n).to(self.device)
        demos = torch.from_numpy(batch.demos).to(self.device)
        weights = torch.from_numpy(batch.weights).to(self.device)

        q_all_raw = self.q(obs)
        nq = max(1, int(getattr(self.config, "n_quantiles", 1)))
        if nq > 1:
            B = int(obs.shape[0])
            q_all_q = q_all_raw.reshape(B, self._n_actions, nq)
            q_all = q_all_q.mean(dim=2)          # (B, n_actions) for margin loss
            q_quantiles = q_all_q.gather(1, actions.view(-1, 1, 1).expand(-1, -1, nq)).squeeze(1)  # (B, nq)
            q_values = q_quantiles.mean(dim=1)   # (B,) scalar for PER
        else:
            q_all = q_all_raw                    # (B, n_actions)
            q_quantiles = None
            q_values = q_all.gather(1, actions.view(-1, 1)).squeeze(1)

        if mode == "legacy":
            # Legacy implementation: n-step TD target only (+ optional demo losses).
            with torch.no_grad():
                mask = next_action_masks_n.to(torch.bool)
                gamma = float(self.config.gamma)
                gamma_n = torch.pow(
                    torch.tensor(gamma, device=self.device, dtype=torch.float32), n_steps_n.to(torch.float32)
                )

                if nq > 1:
                    # QR-DQN: compute per-quantile targets.
                    next_tgt_q = self._next_q_quantiles(next_obs_n, mask, nq)
                    target_q = rewards_n.unsqueeze(1) + (1.0 - dones_n.unsqueeze(1)) * (gamma_n.unsqueeze(1) * next_tgt_q)
                else:
                    if self.base_algo == "ddqn":
                        q_next_online = self._q_values(self.q(next_obs_n))
                        q_next_online = q_next_online.masked_fill(~mask, torch.finfo(q_next_online.dtype).min)
                        next_actions = torch.argmax(q_next_online, dim=1, keepdim=True)
                        q_next_target = self._q_values(self.q_target(next_obs_n))
                        q_next_target = q_next_target.masked_fill(~mask, torch.finfo(q_next_target.dtype).min)
                        next_q = q_next_target.gather(1, next_actions).squeeze(1)
                        next_q = torch.where(torch.isfinite(next_q), next_q, torch.zeros_like(next_q))
                        target = rewards_n + (1.0 - dones_n) * (gamma_n * next_q)
                    elif self.base_algo == "mdqn":
                        _tau = float(self.config.mdqn_tau)
                        _alpha = float(self.config.mdqn_alpha)
                        _l0 = float(self.config.mdqn_l0)
                        q_tgt_next = self._q_values(self.q_target(next_obs_n))
                        q_tgt_next = q_tgt_next.masked_fill(~mask, torch.finfo(q_tgt_next.dtype).min)
                        pi_next = F.softmax(q_tgt_next / _tau, dim=1)
                        log_pi_next = torch.log(pi_next + 1e-8).clamp(min=_l0)
                        soft_v = (pi_next * (q_tgt_next - _tau * log_pi_next)).sum(dim=1)
                        soft_v = torch.where(torch.isfinite(soft_v), soft_v, torch.zeros_like(soft_v))
                        q_tgt_curr = self._q_values(self.q_target(obs))
                        pi_curr = F.softmax(q_tgt_curr / _tau, dim=1)
                        log_pi_curr = torch.log(pi_curr + 1e-8).clamp(min=_l0)
                        log_pi_a = log_pi_curr.gather(1, actions.view(-1, 1)).squeeze(1)
                        target = (rewards_n + _alpha * _tau * log_pi_a
                                  + (1.0 - dones_n) * (gamma_n * soft_v))
                    else:
                        q_next_target = self._q_values(self.q_target(next_obs_n))
                        q_next_target = q_next_target.masked_fill(~mask, torch.finfo(q_next_target.dtype).min)
                        next_q = q_next_target.max(dim=1).values
                        next_q = torch.where(torch.isfinite(next_q), next_q, torch.zeros_like(next_q))
                        target = rewards_n + (1.0 - dones_n) * (gamma_n * next_q)

            if nq > 1:
                td_loss = self._quantile_td_loss(q_quantiles, target_q)
            else:
                losses = self.loss_fn(q_values, target)
                td_loss = losses.mean()

            demo_lambda = float(getattr(self.config, "demo_lambda", 0.0))
            demo_margin = float(getattr(self.config, "demo_margin", 0.0))
            margin_loss = torch.tensor(0.0, device=self.device)
            if demo_lambda > 0.0 and demo_margin > 0.0:
                demo_mask = demos.float().clamp(0.0, 1.0)
                if torch.any(demo_mask > 0.0):
                    q_other = q_all.clone()
                    q_other.scatter_(1, actions.view(-1, 1), torch.finfo(q_other.dtype).min)
                    q_max_other = q_other.max(dim=1).values
                    margin = torch.relu(q_max_other + float(demo_margin) - q_values)
                    margin_loss = (margin * demo_mask).mean()

            demo_ce_lambda = float(getattr(self.config, "demo_ce_lambda", 0.0))
            ce_loss = torch.tensor(0.0, device=self.device)
            if demo_ce_lambda > 0.0:
                demo_mask = demos.float().clamp(0.0, 1.0)
                if torch.any(demo_mask > 0.0):
                    ce = F.cross_entropy(q_all, actions.long(), reduction="none")
                    ce_loss = (ce * demo_mask).mean()

            aux_lambda = float(getattr(self.config, "aux_admissibility_lambda", 0.0))
            aux_adm_loss = torch.tensor(0.0, device=self.device)
            if self.aux_adm_head is not None and aux_lambda > 0.0:
                aux_logits = self.aux_adm_head(self.q(next_obs_1))
                aux_target = next_action_masks_1.to(torch.float32)
                aux_adm_loss = F.binary_cross_entropy_with_logits(aux_logits, aux_target, reduction="mean")

            loss = (
                td_loss
                + float(demo_lambda) * margin_loss
                + float(demo_ce_lambda) * ce_loss
                + float(aux_lambda) * aux_adm_loss
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.config.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(self._optimizer_params(), max_norm=self.config.grad_clip_norm)
            self.optimizer.step()

            self._train_steps += 1
            tau = float(getattr(self.config, "target_update_tau", 0.0))
            if tau > 0.0:
                with torch.no_grad():
                    for p_t, p in zip(self.q_target.parameters(), self.q.parameters(), strict=False):
                        p_t.data.lerp_(p.data, float(tau))
            elif self._train_steps % self.config.target_update_steps == 0:
                self.q_target.load_state_dict(self.q.state_dict())

            return {
                "loss": float(loss.item()),
                "td_loss": float(td_loss.item()),
                "margin_loss": float(margin_loss.item()),
                "ce_loss": float(ce_loss.item()),
                "aux_adm_loss": float(aux_adm_loss.item()),
            }

        # Strict DQfD-style update (PER + 1-step TD + n-step TD + margin + L2; no CE).
        with torch.no_grad():
            mask1 = next_action_masks_1.to(torch.bool)
            maskn = next_action_masks_n.to(torch.bool)
            gamma = float(self.config.gamma)
            gamma_n = torch.pow(
                torch.tensor(gamma, device=self.device, dtype=torch.float32), n_steps_n.to(torch.float32)
            )

            if nq > 1:
                # QR-DQN: per-quantile targets for both 1-step and n-step.
                nq1 = self._next_q_quantiles(next_obs_1, mask1, nq)
                target1_q = rewards_1.unsqueeze(1) + (1.0 - dones_1.unsqueeze(1)) * (float(gamma) * nq1)
                nqn = self._next_q_quantiles(next_obs_n, maskn, nq)
                targetn_q = rewards_n.unsqueeze(1) + (1.0 - dones_n.unsqueeze(1)) * (gamma_n.unsqueeze(1) * nqn)
            else:
                def _scalar_next_q(next_obs: torch.Tensor, msk: torch.Tensor) -> torch.Tensor:
                    if self.base_algo == "ddqn":
                        qno = self._q_values(self.q(next_obs))
                        qno = qno.masked_fill(~msk, torch.finfo(qno.dtype).min)
                        na = torch.argmax(qno, dim=1, keepdim=True)
                        qnt = self._q_values(self.q_target(next_obs))
                        qnt = qnt.masked_fill(~msk, torch.finfo(qnt.dtype).min)
                        nv = qnt.gather(1, na).squeeze(1)
                    elif self.base_algo == "mdqn":
                        _tau = float(self.config.mdqn_tau)
                        _l0 = float(self.config.mdqn_l0)
                        qnt = self._q_values(self.q_target(next_obs))
                        qnt = qnt.masked_fill(~msk, torch.finfo(qnt.dtype).min)
                        pi = F.softmax(qnt / _tau, dim=1)
                        lp = torch.log(pi + 1e-8).clamp(min=_l0)
                        nv = (pi * (qnt - _tau * lp)).sum(dim=1)
                    else:
                        qnt = self._q_values(self.q_target(next_obs))
                        qnt = qnt.masked_fill(~msk, torch.finfo(qnt.dtype).min)
                        nv = qnt.max(dim=1).values
                    return torch.where(torch.isfinite(nv), nv, torch.zeros_like(nv))
                next_q1 = _scalar_next_q(next_obs_1, mask1)
                target1 = rewards_1 + (1.0 - dones_1) * (float(gamma) * next_q1)
                next_qn = _scalar_next_q(next_obs_n, maskn)
                targetn = rewards_n + (1.0 - dones_n) * (gamma_n * next_qn)
                # M-DQN: add log-policy reward augmentation to targets.
                if self.base_algo == "mdqn":
                    _tau = float(self.config.mdqn_tau)
                    _alpha = float(self.config.mdqn_alpha)
                    _l0 = float(self.config.mdqn_l0)
                    q_tgt_c = self._q_values(self.q_target(obs))
                    pi_c = F.softmax(q_tgt_c / _tau, dim=1)
                    lp_c = torch.log(pi_c + 1e-8).clamp(min=_l0)
                    lp_a = lp_c.gather(1, actions.view(-1, 1)).squeeze(1)
                    bonus = _alpha * _tau * lp_a
                    target1 = target1 + bonus
                    targetn = targetn + bonus

        # TD losses (importance-sampled).
        is_w = weights.to(torch.float32).clamp_min(0.0)
        if nq > 1:
            loss1 = self._quantile_td_loss(q_quantiles, target1_q, is_w)
            lossn = self._quantile_td_loss(q_quantiles, targetn_q, is_w)
            td_error1 = (target1_q.mean(dim=1) - q_values).detach()
        else:
            td_error1 = target1 - q_values
            loss1 = (self.loss_fn(q_values, target1) * is_w).mean()
            lossn = (self.loss_fn(q_values, targetn) * is_w).mean()
        lambda_n = float(getattr(self.config, "dqfd_lambda_n", 1.0))
        td_loss = loss1 + float(lambda_n) * lossn

        # Expert large-margin loss (DQfD). Only applied to `demo` transitions.
        demo_lambda = float(getattr(self.config, "demo_lambda", 0.0))
        demo_margin = float(getattr(self.config, "demo_margin", 0.0))
        margin_loss = torch.tensor(0.0, device=self.device)
        if demo_lambda > 0.0 and demo_margin > 0.0:
            demo_mask = demos.float().clamp(0.0, 1.0)
            if torch.any(demo_mask > 0.0):
                q_other = q_all.clone()
                q_other.scatter_(1, actions.view(-1, 1), torch.finfo(q_other.dtype).min)
                q_max_other = q_other.max(dim=1).values
                margin = torch.relu(q_max_other + float(demo_margin) - q_values)
                margin_loss = (margin * demo_mask * is_w).mean()

        aux_lambda = float(getattr(self.config, "aux_admissibility_lambda", 0.0))
        aux_adm_loss = torch.tensor(0.0, device=self.device)
        if self.aux_adm_head is not None and aux_lambda > 0.0:
            aux_logits = self.aux_adm_head(self._q_values(self.q(next_obs_1)))
            aux_target = next_action_masks_1.to(torch.float32)
            aux_bce = F.binary_cross_entropy_with_logits(aux_logits, aux_target, reduction="none")
            aux_adm_loss = (aux_bce * is_w.view(-1, 1)).mean()

        # L2 regularization loss.
        l2_reg = float(getattr(self.config, "l2_reg", 0.0))
        l2_loss = torch.tensor(0.0, device=self.device)
        if l2_reg > 0.0:
            l2 = torch.tensor(0.0, device=self.device)
            for p in self.q.parameters():
                l2 = l2 + torch.sum(p.float() ** 2)
            l2_loss = float(l2_reg) * l2

        loss = td_loss + float(demo_lambda) * margin_loss + l2_loss + float(aux_lambda) * aux_adm_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(self._optimizer_params(), max_norm=self.config.grad_clip_norm)
        self.optimizer.step()

        self._train_steps += 1
        tau = float(getattr(self.config, "target_update_tau", 0.0))
        if tau > 0.0:
            with torch.no_grad():
                for p_t, p in zip(self.q_target.parameters(), self.q.parameters(), strict=False):
                    p_t.data.lerp_(p.data, float(tau))
        elif self._train_steps % self.config.target_update_steps == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        # PER priority update uses the 1-step TD error.
        try:
            self.replay.update_priorities(
                batch.idxs,
                td_error1.detach().cpu().numpy(),
                demos=batch.demos,
                flags=batch.flags,
            )
        except Exception:
            # Priority update should never crash training; treat as a soft failure.
            pass

        return {
            "loss": float(loss.item()),
            "td_loss": float(td_loss.item()),
            "td1_loss": float(loss1.item()),
            "tdn_loss": float(lossn.item()),
            "margin_loss": float(margin_loss.item()),
            "l2_loss": float(l2_loss.item()),
            "aux_adm_loss": float(aux_adm_loss.item()),
            "per_beta": float(self._per_beta()),
            "is_w_mean": float(is_w.mean().item()),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "algo": self.algo,
            "arch": self.arch,
            "base_algo": self.base_algo,
            "network": self.arch,
            "network_kwargs": dict(self._net_kwargs),
            "obs_dim": int(self._obs_dim),
            "n_actions": int(self._n_actions),
            "config": self.config.__dict__,
            "q_state_dict": self.q.state_dict(),
            "q_target_state_dict": self.q_target.state_dict(),
            "aux_adm_head_state_dict": (self.aux_adm_head.state_dict() if self.aux_adm_head is not None else None),
            "train_steps": self._train_steps,
        }
        torch.save(payload, path)

    def load(self, path: str | Path) -> None:
        payload = torch.load(Path(path), map_location=self.device)
        q_sd = payload["q_state_dict"]
        q_target_sd = payload.get("q_target_state_dict", {})
        aux_adm_sd = payload.get("aux_adm_head_state_dict", None)

        payload_algo = payload.get("algo", self.algo)
        canonical_algo, arch, base_algo, _legacy = parse_rl_algo(str(payload_algo))
        self.algo = canonical_algo
        self.arch = arch
        self.base_algo = base_algo

        network = str(payload.get("network", self.arch)).lower().strip()
        if network in {"plain", "qnetwork", "mlp"}:
            net_cls: type[nn.Module] = MLPQNetwork
            net_kwargs: dict[str, object] = {}
            self.arch = "mlp"
        elif network == "cnn":
            net_cls = CNNQNetwork
            net_kwargs_raw = payload.get("network_kwargs") or {}
            if not isinstance(net_kwargs_raw, dict):
                net_kwargs_raw = {}
            if not net_kwargs_raw:
                layout = infer_flat_obs_cnn_layout(int(self._obs_dim))
                net_kwargs = {"scalar_dim": layout.scalar_dim, "map_channels": layout.map_channels, "map_size": layout.map_size}
            else:
                net_kwargs = {str(k): v for k, v in net_kwargs_raw.items()}
            self.arch = "cnn"
        elif network == "cnn-dual":
            from forest_vehicle_dqn.networks import DualScaleCNNQNetwork
            net_cls = DualScaleCNNQNetwork
            net_kwargs = {}
            self.arch = "cnn-dual"
        else:
            raise ValueError(f"Unsupported network type in checkpoint: {network!r}")

        self._net_cls = net_cls
        self._net_kwargs = dict(net_kwargs)

        cfg = payload.get("config") or {}
        hidden_dim = int(cfg.get("hidden_dim", self.config.hidden_dim))
        hidden_layers = int(cfg.get("hidden_layers", self.config.hidden_layers))

        # Restore module flags from checkpoint so _q_values / act behave correctly.
        restore_keys = ("n_quantiles", "noisy_net", "cbam", "mha", "mha_heads", "dueling")
        restore = {k: cfg[k] for k in restore_keys if k in cfg}
        if restore:
            self.config = replace(self.config, **restore)

        # Architecture can change across experiments (hidden_dim/layers). Rebuild when shapes mismatch.
        try:
            self.q.load_state_dict(q_sd, strict=True)
        except RuntimeError:
            self._rebuild_networks(net_cls, hidden_dim=hidden_dim, hidden_layers=hidden_layers, net_kwargs=net_kwargs)

        self.q.load_state_dict(q_sd, strict=True)
        if q_target_sd:
            try:
                self.q_target.load_state_dict(q_target_sd, strict=True)
            except RuntimeError:
                # Fall back to syncing the target net if the checkpoint predates saving it (or shapes mismatch).
                self.q_target.load_state_dict(self.q.state_dict(), strict=True)
        else:
            self.q_target.load_state_dict(self.q.state_dict(), strict=True)

        if isinstance(aux_adm_sd, dict):
            if self.aux_adm_head is None:
                self.aux_adm_head = nn.Linear(int(self._n_actions), int(self._n_actions)).to(self.device)
            try:
                self.aux_adm_head.load_state_dict(aux_adm_sd, strict=True)
            except RuntimeError:
                self.aux_adm_head = None

        self._train_steps = int(payload.get("train_steps", 0))
        self.optimizer = torch.optim.Adam(self._optimizer_params(), lr=self.config.learning_rate)


# ---------------------------------------------------------------------------
# V36: Shared helpers for DRQN / HistFormer
# ---------------------------------------------------------------------------

class EpisodeReplayBuffer:
    """Stores completed episodes; samples contiguous sub-sequences.

    Each episode is a list of (obs, action, reward, next_obs, done) tuples.
    """

    def __init__(self, capacity: int = 1000, seq_len: int = 8) -> None:
        self.capacity = int(capacity)
        self.seq_len = int(seq_len)
        self._episodes: list[list] = []

    def add_episode(self, episode: list) -> None:
        if len(episode) < 2:
            return
        if len(self._episodes) >= self.capacity:
            self._episodes.pop(0)
        self._episodes.append(list(episode))

    def __len__(self) -> int:
        return len(self._episodes)

    def sample_sequences(
        self, batch_size: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        T = self.seq_len
        obs_l, act_l, rew_l, nobs_l, done_l = [], [], [], [], []
        while len(obs_l) < batch_size:
            ep = self._episodes[int(rng.integers(0, len(self._episodes)))]
            if len(ep) < T:
                ep = ep + [ep[-1]] * (T - len(ep))
                start = 0
            else:
                start = int(rng.integers(0, len(ep) - T + 1))
            seq = ep[start: start + T]
            obs_l.append(np.stack([t[0] for t in seq]))
            act_l.append(np.array([t[1] for t in seq], dtype=np.int64))
            rew_l.append(np.array([t[2] for t in seq], dtype=np.float32))
            nobs_l.append(np.stack([t[3] for t in seq]))
            done_l.append(np.array([t[4] for t in seq], dtype=np.float32))
        return (
            np.stack(obs_l), np.stack(act_l), np.stack(rew_l),
            np.stack(nobs_l), np.stack(done_l),
        )


class DRQNAgent:
    """V36-A: DRQN agent (CNN → LSTM → Q). Requires EpisodeReplayBuffer."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        *,
        config: AgentConfig,
        device: str | torch.device = "cpu",
        seed: int = 0,
        seq_len: int = 8,
    ) -> None:
        from forest_vehicle_dqn.networks import DRQNNetwork
        self.device = torch.device(device)
        self.config = config
        self._n_actions = int(n_actions)
        self._obs_dim = int(obs_dim)
        self.seq_len = int(seq_len)
        self._rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        self.q = DRQNNetwork(obs_dim, n_actions, hidden_dim=config.hidden_dim).to(self.device)
        self.q_target = DRQNNetwork(obs_dim, n_actions, hidden_dim=config.hidden_dim).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()

        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=config.learning_rate)
        self.replay = EpisodeReplayBuffer(capacity=1000, seq_len=seq_len)
        self._hidden: tuple | None = None
        self._cur_ep: list = []
        self._train_steps = 0

    def reset_episode(self) -> None:
        self._hidden = self.q.init_hidden(1, self.device)
        if self._cur_ep:
            self.replay.add_episode(self._cur_ep)
        self._cur_ep = []

    @property
    def eps(self) -> float:
        return float(np.clip(
            float(self.config.eps_start) - float(self._train_steps) *
            (float(self.config.eps_start) - float(self.config.eps_final)) /
            max(1, int(self.config.eps_decay)),
            float(self.config.eps_final), float(self.config.eps_start),
        ))

    def act(self, obs: np.ndarray, explore: bool = True) -> int:
        if explore and self._rng.random() < self.eps:
            return int(self._rng.integers(0, self._n_actions))
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q, self._hidden = self.q(x, self._hidden)
        return int(q.argmax(dim=-1).item())

    def observe(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        self._cur_ep.append((obs.copy(), int(action), float(reward), next_obs.copy(), bool(done)))

    def train_step(self) -> float | None:
        if len(self.replay) < 4:
            return None
        obs_b, act_b, rew_b, nobs_b, done_b = self.replay.sample_sequences(
            self.config.batch_size, self._rng
        )
        B, T = obs_b.shape[:2]
        dev = self.device
        obs_t = torch.as_tensor(obs_b, dtype=torch.float32, device=dev)
        act_t = torch.as_tensor(act_b, dtype=torch.long, device=dev)
        rew_t = torch.as_tensor(rew_b, dtype=torch.float32, device=dev)
        nobs_t = torch.as_tensor(nobs_b, dtype=torch.float32, device=dev)
        done_t = torch.as_tensor(done_b, dtype=torch.float32, device=dev)
        gamma = float(self.config.gamma)

        with torch.no_grad():
            q_next, _ = self.q_target(nobs_t)     # (B, T, n_actions)
            q_next_online, _ = self.q(nobs_t)     # DDQN: online net for argmax
            next_act = q_next_online.argmax(dim=-1, keepdim=True)
            q_next_val = q_next.gather(2, next_act).squeeze(2)  # (B, T)
            target = rew_t + gamma * q_next_val * (1.0 - done_t)

        q_vals, _ = self.q(obs_t)                  # (B, T, n_actions)
        q_taken = q_vals.gather(2, act_t.unsqueeze(2)).squeeze(2)  # (B, T)
        loss = F.smooth_l1_loss(q_taken, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()
        self._train_steps += 1
        if self._train_steps % self.config.target_update_steps == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return float(loss.item())

    def save(self, path: str | "Path") -> None:
        torch.save({
            "algo": "cnn-drqn", "network": "cnn-drqn",
            "q_state_dict": self.q.state_dict(),
            "q_target_state_dict": self.q_target.state_dict(),
            "train_steps": self._train_steps,
            "config": {"hidden_dim": self.config.hidden_dim, "hidden_layers": self.config.hidden_layers},
        }, str(path))

    def load(self, path: str | "Path") -> None:
        payload = torch.load(str(path), map_location=self.device)
        self.q.load_state_dict(payload["q_state_dict"])
        self.q_target.load_state_dict(payload.get("q_target_state_dict", payload["q_state_dict"]))
        self._train_steps = int(payload.get("train_steps", 0))


class HistFormerAgent:
    """V36-C: HistTransformer agent (CNN → rolling T-step buffer → Transformer → Q)."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        *,
        config: AgentConfig,
        device: str | torch.device = "cpu",
        seed: int = 0,
        seq_len: int = 8,
        d_model: int = 128,
    ) -> None:
        from forest_vehicle_dqn.networks import HistTransformerNetwork
        self.device = torch.device(device)
        self.config = config
        self._n_actions = int(n_actions)
        self._obs_dim = int(obs_dim)
        self.seq_len = int(seq_len)
        self.d_model = int(d_model)
        self._rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        self.q = HistTransformerNetwork(
            obs_dim, n_actions, d_model=d_model, seq_len=seq_len, hidden_dim=config.hidden_dim
        ).to(self.device)
        self.q_target = HistTransformerNetwork(
            obs_dim, n_actions, d_model=d_model, seq_len=seq_len, hidden_dim=config.hidden_dim
        ).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()

        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=config.learning_rate)
        self.replay = EpisodeReplayBuffer(capacity=1000, seq_len=seq_len)
        self._hist_buf: torch.Tensor | None = None  # (1, T, d_model)
        self._cur_ep: list = []
        self._train_steps = 0

    def reset_episode(self) -> None:
        self._hist_buf = torch.zeros(1, self.seq_len, self.d_model, device=self.device)
        if self._cur_ep:
            self.replay.add_episode(self._cur_ep)
        self._cur_ep = []

    @property
    def eps(self) -> float:
        return float(np.clip(
            float(self.config.eps_start) - float(self._train_steps) *
            (float(self.config.eps_start) - float(self.config.eps_final)) /
            max(1, int(self.config.eps_decay)),
            float(self.config.eps_final), float(self.config.eps_start),
        ))

    def act(self, obs: np.ndarray, explore: bool = True) -> int:
        if explore and self._rng.random() < self.eps:
            return int(self._rng.integers(0, self._n_actions))
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            feat = self.q.encode_step(x.unsqueeze(0))        # (1, d_model)
        assert self._hist_buf is not None
        self._hist_buf = torch.roll(self._hist_buf, -1, dims=1)
        self._hist_buf[0, -1, :] = feat[0]
        with torch.no_grad():
            q = self.q(self._hist_buf)                        # (1, n_actions)
        return int(q.argmax(dim=-1).item())

    def observe(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        self._cur_ep.append((obs.copy(), int(action), float(reward), next_obs.copy(), bool(done)))

    def train_step(self) -> float | None:
        if len(self.replay) < 4:
            return None
        obs_b, act_b, rew_b, nobs_b, done_b = self.replay.sample_sequences(
            self.config.batch_size, self._rng
        )
        B, T = obs_b.shape[:2]
        dev = self.device
        obs_t = torch.as_tensor(obs_b, dtype=torch.float32, device=dev)
        act_t = torch.as_tensor(act_b, dtype=torch.long, device=dev)
        rew_t = torch.as_tensor(rew_b, dtype=torch.float32, device=dev)
        nobs_t = torch.as_tensor(nobs_b, dtype=torch.float32, device=dev)
        done_t = torch.as_tensor(done_b, dtype=torch.float32, device=dev)
        gamma = float(self.config.gamma)
        obs_flat = obs_t.reshape(B * T, -1)
        nobs_flat = nobs_t.reshape(B * T, -1)
        with torch.no_grad():
            nfeat = self.q_target.encode_step(nobs_flat).reshape(B, T, self.d_model)
            nfeat_online = self.q.encode_step(nobs_flat).reshape(B, T, self.d_model)
            q_next = self.q_target(nfeat)
            q_next_online = self.q(nfeat_online)
            next_act = q_next_online.argmax(dim=-1, keepdim=True)
            q_next_val = q_next.gather(1, next_act).squeeze(1)
            target = rew_t[:, -1] + gamma * q_next_val * (1.0 - done_t[:, -1])
        feat = self.q.encode_step(obs_flat).reshape(B, T, self.d_model)
        q_vals = self.q(feat)
        q_taken = q_vals.gather(1, act_t[:, -1].unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q_taken, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()
        self._train_steps += 1
        if self._train_steps % self.config.target_update_steps == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return float(loss.item())

    def save(self, path: str | "Path") -> None:
        torch.save({
            "algo": "cnn-histformer", "network": "cnn-hist",
            "q_state_dict": self.q.state_dict(),
            "q_target_state_dict": self.q_target.state_dict(),
            "train_steps": self._train_steps,
            "config": {"hidden_dim": self.config.hidden_dim, "hidden_layers": self.config.hidden_layers},
        }, str(path))

    def load(self, path: str | "Path") -> None:
        payload = torch.load(str(path), map_location=self.device)
        self.q.load_state_dict(payload["q_state_dict"])
        self.q_target.load_state_dict(payload.get("q_target_state_dict", payload["q_state_dict"]))
        self._train_steps = int(payload.get("train_steps", 0))
