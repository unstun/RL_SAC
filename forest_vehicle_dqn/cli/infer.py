from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from forest_vehicle_dqn.config_io import apply_config_defaults, load_json, resolve_config_path, select_section
from forest_vehicle_dqn.runtime import configure_runtime, select_device, torch_cuda_arch_info, torch_runtime_info
from forest_vehicle_dqn.runs import create_run_dir, resolve_experiment_dir, resolve_models_dir

configure_runtime()

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.ticker as mticker
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]
    mpatches = None  # type: ignore[assignment]
    mticker = None  # type: ignore[assignment]
try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    import gym  # type: ignore
import numpy as np
import pandas as pd
import torch

from forest_vehicle_dqn.agents import AgentConfig, DQNFamilyAgent, parse_rl_algo
from forest_vehicle_dqn.baselines.mpc_local_planner import MPCConfig, run_mpc_local_planning
from forest_vehicle_dqn.baselines.pathplan import (
    AStarCurveOptConfig,
    default_ackermann_params,
    PlannerResult,
    forest_two_circle_footprint,
    grid_map_from_obstacles,
    plan_grid_astar,
    plan_hybrid_astar,
    plan_rrt_star,
    point_footprint,
)
from forest_vehicle_dqn.env import AMRBicycleEnv, AMRGridEnv, RewardWeights, bicycle_integrate_one_step, wrap_angle_rad
from forest_vehicle_dqn.maps import FOREST_ENV_ORDER, get_map_spec
from forest_vehicle_dqn.metrics import KPI, avg_abs_curvature, max_corner_degree, num_path_corners, path_length
from forest_vehicle_dqn.smoothing import chaikin_smooth


def _safe_slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")


@dataclass(frozen=True)
class PathTrace:
    path_xy_cells: list[tuple[float, float]]
    success: bool


@dataclass(frozen=True)
class ControlTrace:
    t_s: np.ndarray
    v_m_s: np.ndarray
    delta_rad: np.ndarray


@dataclass(frozen=True)
class RolloutResult:
    path_xy_cells: list[tuple[float, float]]
    compute_time_s: float
    reached: bool
    steps: int
    path_time_s: float
    controls: ControlTrace | None = None
    collision: bool = False
    truncated: bool = False
    debug: dict[str, object] | None = None


def _env_dt_s(env: gym.Env) -> float:
    if isinstance(env, AMRBicycleEnv):
        return float(env.model.dt)
    return 1.0


def forest_stop_action(env: AMRBicycleEnv) -> int:
    """Pick a discrete action that keeps the robot near the goal while driving v,delta -> 0."""
    dt = float(env.model.dt)
    if not (dt > 0.0):
        return 0

    x0 = float(env._x_m)
    y0 = float(env._y_m)
    psi0 = float(env._psi_rad)
    v0 = float(env._v_m_s)
    delta0 = float(env._delta_rad)

    gx_m = float(env.goal_xy[0]) * float(env.cell_size_m)
    gy_m = float(env.goal_xy[1]) * float(env.cell_size_m)
    tol_m = max(1e-6, float(env.goal_tolerance_m))
    v_max = max(1e-9, float(env.model.v_max_m_s))
    delta_max = max(1e-9, float(env.model.delta_max_rad))

    best_action = 0
    best_cost = float("inf")
    for a_id in range(int(env.action_table.shape[0])):
        delta_dot = float(env.action_table[a_id, 0])
        accel = float(env.action_table[a_id, 1])
        x, y, psi, v1, delta1 = bicycle_integrate_one_step(
            x_m=x0,
            y_m=y0,
            psi_rad=psi0,
            v_m_s=v0,
            delta_rad=delta0,
            delta_dot_rad_s=delta_dot,
            a_m_s2=accel,
            params=env.model,
        )
        _od, coll = env._od_and_collision_at_pose_m(x, y, psi)
        if bool(coll):
            continue

        d_goal = float(math.hypot(float(gx_m) - float(x), float(gy_m) - float(y)))
        d_term = d_goal / float(tol_m)
        v_term = abs(float(v1)) / float(v_max)
        delta_term = abs(float(delta1)) / float(delta_max)
        cost = 5.0 * float(d_term) + float(v_term) + float(delta_term)

        if float(cost) < float(best_cost):
            best_cost = float(cost)
            best_action = int(a_id)

    return int(best_action)


def _ensemble_fuse(
    q_all: list[torch.Tensor],
    *,
    mode: str = "qavg",
    negotiate_k: int = 20,
    negotiate_k_max: int = 30,
    negsoft_alpha: float = 0.7,
    negsoft_tau: float = 0.1,
) -> torch.Tensor:
    """Fuse Q-values from multiple ensemble members.

    Modes:
      qavg — simple Q-value average (baseline)
      pavg — policy averaging: softmax each → average policies → output as q
      vote — majority voting: each model argmax → vote → one-hot as q
      conf — confidence-weighted: weight by inverse entropy of softmax(Q)
      rank — Borda count: scale-invariant rank fusion
      qgap — Q-gap weighted: confidence = max-2nd gap (better than entropy)
      veto — primary-first: trust DDQN unless M-DQN vetoes → fallback conf
      topk — primary top-K filtered: DDQN proposes, secondary picks
    """
    stacked = torch.stack(q_all)  # (N, n_actions)
    if mode == "qavg":
        return stacked.mean(dim=0)
    elif mode == "pavg":
        tau = 0.03
        policies = torch.softmax(stacked / tau, dim=1)  # (N, n_actions)
        avg_policy = policies.mean(dim=0)  # (n_actions,)
        return avg_policy
    elif mode == "vote":
        n_actions = stacked.shape[1]
        votes = torch.zeros(n_actions, device=stacked.device)
        for q_i in q_all:
            votes[int(torch.argmax(q_i).item())] += 1.0
        return votes
    elif mode == "conf":
        tau = 0.03
        policies = torch.softmax(stacked / tau, dim=1)  # (N, n_actions)
        entropy = -(policies * torch.log(policies + 1e-8)).sum(dim=1)  # (N,)
        weights = 1.0 / (entropy + 1e-6)  # (N,)
        weights = weights / weights.sum()
        return (weights.unsqueeze(1) * stacked).sum(dim=0)
    elif mode == "rank":
        # Borda count: scale-invariant rank fusion
        n_models, n_actions = stacked.shape
        ranks = torch.zeros_like(stacked)
        for i in range(n_models):
            order = torch.argsort(stacked[i], descending=True)
            r = torch.zeros(n_actions, device=stacked.device, dtype=stacked.dtype)
            r[order] = torch.arange(n_actions, device=stacked.device, dtype=stacked.dtype)
            ranks[i] = r
        return -(ranks.sum(dim=0))  # negate: argmax picks lowest rank
    elif mode == "qgap":
        # Q-gap weighted: confidence = gap between top-2 Q-values
        sorted_q, _ = torch.sort(stacked, dim=1, descending=True)
        gaps = sorted_q[:, 0] - sorted_q[:, 1]  # (N,)
        weights = gaps / (gaps.sum() + 1e-8)
        return (weights.unsqueeze(1) * stacked).sum(dim=0)
    elif mode == "veto":
        # Primary-first with safety net: trust primary unless secondary vetoes
        primary_q = q_all[0]
        best_a = int(torch.argmax(primary_q).item())
        veto_k = 10
        vetoed = False
        for q_i in q_all[1:]:
            topk_idx = torch.topk(q_i, min(veto_k, q_i.shape[0])).indices
            if best_a not in topk_idx.tolist():
                vetoed = True
                break
        if not vetoed:
            return primary_q
        return _ensemble_fuse(q_all, mode="conf")
    elif mode == "topk":
        # Primary proposes top-K, secondary picks best among them
        primary_q = q_all[0]
        k, n_a = 10, primary_q.shape[0]
        _, topk_idx = torch.topk(primary_q, min(k, n_a))
        topk_set = set(topk_idx.tolist())
        if len(q_all) <= 1:
            return primary_q
        sec_stack = torch.stack(q_all[1:])
        sec_avg = sec_stack.mean(dim=0)
        result = torch.full((n_a,), float('-inf'), device=stacked.device)
        for a in topk_set:
            result[a] = sec_avg[a]
        return result
    elif mode == "negotiate":
        # DDQN proposes by rank, M-DQN approves/vetoes, iterate until agreed
        primary_q = q_all[0]
        approve_k = negotiate_k  # secondary approves if action is in its top-K
        secondary_topk = []
        for q_i in q_all[1:]:
            topk_idx = torch.topk(q_i, min(approve_k, q_i.shape[0])).indices
            secondary_topk.append(set(topk_idx.tolist()))
        primary_order = torch.argsort(primary_q, descending=True).tolist()
        for a in primary_order:
            if all(a in s for s in secondary_topk):
                result = primary_q.clone()
                result[a] = primary_q.max() + 1.0  # boost negotiated action
                return result
        return primary_q  # no agreement → trust primary
    elif mode == "negrelax":
        # Progressive relaxation: try K_min, K_min+5, ..., K_max
        primary_q = q_all[0]
        n_actions = primary_q.shape[0]
        k_min = negotiate_k
        k_max = min(negotiate_k_max, n_actions)
        primary_order = torch.argsort(primary_q, descending=True).tolist()
        for current_k in range(k_min, k_max + 1, 5):
            secondary_topk = []
            for q_i in q_all[1:]:
                topk_idx = torch.topk(q_i, min(current_k, n_actions)).indices
                secondary_topk.append(set(topk_idx.tolist()))
            for a in primary_order:
                if all(a in s for s in secondary_topk):
                    result = primary_q.clone()
                    result[a] = primary_q.max() + 1.0
                    return result
        return primary_q  # all relaxation failed → trust primary
    elif mode == "negsoft":
        # Soft-vote negotiate: rank(DDQN) + softmax(M-DQN) weighted sum
        primary_q = q_all[0]
        n = primary_q.shape[0]
        primary_order = torch.argsort(primary_q, descending=True)
        primary_score = torch.zeros(n, device=primary_q.device, dtype=primary_q.dtype)
        primary_score[primary_order] = torch.arange(
            n - 1, -1, -1, device=primary_q.device, dtype=primary_q.dtype
        )
        sec_score = torch.zeros(n, device=primary_q.device, dtype=primary_q.dtype)
        for q_i in q_all[1:]:
            sec_score += torch.softmax(q_i / negsoft_tau, dim=0)
        sec_score /= max(len(q_all) - 1, 1)
        sec_score *= n  # scale to same range as primary_score
        combined = negsoft_alpha * primary_score + (1.0 - negsoft_alpha) * sec_score
        return combined
    else:
        raise ValueError(f"Unknown ensemble_mode: {mode!r}")


def rollout_agent(
    env: gym.Env,
    agent: DQNFamilyAgent,
    *,
    max_steps: int,
    seed: int,
    reset_options: dict[str, object] | None = None,
    time_mode: str = "rollout",
    obs_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    forest_adm_horizon: int = 15,
    forest_adm_persistence: int = 0,
    forest_topk: int = 10,
    forest_min_od_m: float = 0.0,
    forest_min_progress_m: float = 1e-4,
    forest_no_fallback: bool = False,
    collect_controls: bool = False,
    trace_path: Path | None = None,
    ensemble_agents: list[DQNFamilyAgent] | None = None,
    ensemble_mode: str = "qavg",
    negotiate_k: int = 20,
    negotiate_k_max: int = 30,
    negsoft_alpha: float = 0.7,
    negsoft_tau: float = 0.1,
    ensemble_adm_prefilter: bool = False,
) -> RolloutResult:
    obs, info0 = env.reset(seed=seed, options=reset_options)
    if obs_transform is not None:
        obs = obs_transform(obs)
    path: list[tuple[float, float]] = [(float(env.start_xy[0]), float(env.start_xy[1]))]
    dt_s = float(_env_dt_s(env))

    t_series: list[float] | None = None
    v_series: list[float] | None = None
    delta_series: list[float] | None = None
    if bool(collect_controls) and isinstance(env, AMRBicycleEnv):
        t_series = [0.0]
        v_series = [float(getattr(env, "_v_m_s", 0.0))]
        delta_series = [float(getattr(env, "_delta_rad", 0.0))]

    trace_rows: list[dict[str, object]] | None = [] if trace_path is not None else None
    if trace_rows is not None:
        if isinstance(env, AMRBicycleEnv):
            d_goal0 = float(env._distance_to_goal_m())
            alpha0 = float(env._goal_relative_angle_rad())
            reached_pose0 = bool(env._goal_pose_reached(d_goal_m=d_goal0, alpha_rad=alpha0))
            reached_stop0 = bool(env._goal_stop_reached(v_m_s=float(env._v_m_s), delta_rad=float(env._delta_rad)))
            reached0 = bool(reached_pose0 and reached_stop0)
            trace_rows.append(
                {
                    "step": 0,
                    "x_m": float(info0.get("pose_m", (env._x_m, env._y_m, env._psi_rad))[0]),
                    "y_m": float(info0.get("pose_m", (env._x_m, env._y_m, env._psi_rad))[1]),
                    "theta_rad": float(info0.get("pose_m", (env._x_m, env._y_m, env._psi_rad))[2]),
                    "v_m_s": float(getattr(env, "_v_m_s", 0.0)),
                    "delta_rad": float(getattr(env, "_delta_rad", 0.0)),
                    "action_id": -1,
                    "delta_dot_rad_s": 0.0,
                    "a_m_s2": 0.0,
                    "od_m": float(getattr(env, "_last_od_m", float("nan"))),
                    "collision": bool(getattr(env, "_last_collision", False)),
                    "reached": bool(reached0),
                    "reached_pose": bool(reached_pose0),
                    "reached_stop": bool(reached_stop0),
                    "stuck": False,
                    "d_goal_m": float(d_goal0),
                    "alpha_rad": float(alpha0),
                    "cell_size_m": float(getattr(env, "cell_size_m", float("nan"))),
                    "start_x": int(getattr(env, "start_xy", (0, 0))[0]),
                    "start_y": int(getattr(env, "start_xy", (0, 0))[1]),
                    "goal_x": int(getattr(env, "goal_xy", (0, 0))[0]),
                    "goal_y": int(getattr(env, "goal_xy", (0, 0))[1]),
                }
            )
        else:
            trace_rows = None

    def sync_cuda() -> None:
        if agent.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

    time_mode = str(time_mode).lower().strip()
    if time_mode not in {"rollout", "policy"}:
        raise ValueError("time_mode must be one of: rollout, policy")

    inference_time_s = 0.0
    sync_cuda()
    t_rollout0 = time.perf_counter()
    done = False
    truncated = False
    steps = 0
    reached = False
    argmax_inadmissible_steps = 0
    replacement_topk_steps = 0
    replacement_mask_steps = 0
    fallback_steps = 0
    stop_override_steps = 0
    prefilter_active_steps = 0
    adm_h = max(1, int(forest_adm_horizon))
    adm_p = max(0, int(forest_adm_persistence))
    topk_k = max(1, int(forest_topk))
    strict_no_fallback = bool(forest_no_fallback)
    min_od = float(forest_min_od_m)
    min_prog = float(forest_min_progress_m)
    last_collision = False
    last_stuck = False

    while not (done or truncated) and steps < max_steps:
        steps += 1
        if time_mode == "policy":
            sync_cuda()
            t0 = time.perf_counter()
        if isinstance(env, AMRBicycleEnv):
            reached_pose = bool(
                env._goal_pose_reached(
                    d_goal_m=float(env._distance_to_goal_m()),
                    alpha_rad=float(env._goal_relative_angle_rad()),
                )
            )
            reached_stop = bool(env._goal_stop_reached(v_m_s=float(env._v_m_s), delta_rad=float(env._delta_rad)))
            if (not bool(strict_no_fallback)) and reached_pose and not reached_stop:
                a = int(forest_stop_action(env))
                stop_override_steps += 1
            else:
                # Forest policy rollout:
                # - strict_no_fallback=True: pure argmax(Q) inference (no masking/replacement/override).
                # - strict_no_fallback=False: keep admissibility-gated replacement logic.
                with torch.no_grad():
                    x = torch.from_numpy(obs.astype(np.float32, copy=False)).to(agent.device)
                    x_batch = x.unsqueeze(0)
                    q_primary = agent._q_values(agent.q(x_batch)).squeeze(0)
                    if ensemble_agents:
                        q_all = [q_primary]
                        for _ea in ensemble_agents:
                            q_i = _ea._q_values(_ea.q(x_batch.to(_ea.device))).squeeze(0)
                            q_all.append(q_i.to(agent.device))
                        # V24: pre-filter inadmissible actions before fusion
                        if ensemble_adm_prefilter and isinstance(env, AMRBicycleEnv):
                            _pf_mask = env.admissible_action_mask(
                                horizon_steps=adm_h, min_od_m=min_od,
                                min_progress_m=min_prog, fallback_to_safe=True,
                                action_persistence=int(adm_p),
                            )
                            _pf_t = torch.from_numpy(_pf_mask).to(q_primary.device)
                            if _pf_t.any():
                                _neg_inf = torch.finfo(q_primary.dtype).min
                                for _qi in range(len(q_all)):
                                    q_all[_qi] = q_all[_qi].clone()
                                    q_all[_qi][~_pf_t] = _neg_inf
                                prefilter_active_steps += 1
                        q = _ensemble_fuse(
                            q_all,
                            mode=ensemble_mode,
                            negotiate_k=negotiate_k,
                            negotiate_k_max=negotiate_k_max,
                            negsoft_alpha=negsoft_alpha,
                            negsoft_tau=negsoft_tau,
                        )
                    else:
                        q = q_primary

                a0 = int(torch.argmax(q).item())
                a = int(a0)

                a0_adm = bool(
                    env.is_action_admissible(int(a0), horizon_steps=adm_h, min_od_m=min_od, min_progress_m=min_prog, action_persistence=int(adm_p))
                )
                if not a0_adm:
                    argmax_inadmissible_steps += 1
                if (not bool(strict_no_fallback)) and (not a0_adm):
                    chosen: int | None = None
                    kk = int(min(int(topk_k), int(q.numel())))
                    topk = torch.topk(q, k=kk, dim=0).indices.detach().cpu().numpy()
                    for cand in topk.tolist():
                        cand_i = int(cand)
                        if cand_i == int(a0):
                            continue
                        if bool(
                            env.is_action_admissible(
                                cand_i, horizon_steps=adm_h, min_od_m=min_od, min_progress_m=min_prog, action_persistence=int(adm_p)
                            )
                        ):
                            chosen = int(cand_i)
                            replacement_topk_steps += 1
                            break

                    if chosen is None:
                        # If there are no admissible top-k alternatives, try masked argmax over
                        # admissible-progress actions. If mask is empty, keep original argmax(a0).
                        prog_mask = env.admissible_action_mask(
                            horizon_steps=adm_h,
                            min_od_m=min_od,
                            min_progress_m=min_prog,
                            fallback_to_safe=False,
                            action_persistence=int(adm_p),
                        )
                        if bool(prog_mask.any()):
                            q_masked = q.clone()
                            q_masked[torch.from_numpy(~prog_mask).to(q.device)] = torch.finfo(q_masked.dtype).min
                            chosen = int(torch.argmax(q_masked).item())
                            replacement_mask_steps += 1

                    if chosen is not None:
                        a = int(chosen)
        else:
            a = agent.act(obs, episode=0, explore=False)
        if time_mode == "policy":
            sync_cuda()
            inference_time_s += float(time.perf_counter() - t0)
        obs, _, done, truncated, info = env.step(a)
        last_collision = bool(info.get("collision", False))
        last_stuck = bool(info.get("stuck", False))
        if obs_transform is not None:
            obs = obs_transform(obs)
        x, y = info["agent_xy"]
        path.append((float(x), float(y)))
        if t_series is not None and v_series is not None and delta_series is not None:
            t_series.append(float(steps) * dt_s)
            v_series.append(float(info.get("v_m_s", float(getattr(env, "_v_m_s", 0.0)))))
            delta_series.append(float(info.get("delta_rad", float(getattr(env, "_delta_rad", 0.0)))))
        if trace_rows is not None and isinstance(env, AMRBicycleEnv):
            px, py, pth = info.get("pose_m", (env._x_m, env._y_m, env._psi_rad))
            dd = float(env.action_table[int(a), 0])
            aa = float(env.action_table[int(a), 1])
            trace_rows.append(
                {
                    "step": int(steps),
                    "x_m": float(px),
                    "y_m": float(py),
                    "theta_rad": float(pth),
                    "v_m_s": float(info.get("v_m_s", env._v_m_s)),
                    "delta_rad": float(info.get("delta_rad", env._delta_rad)),
                    "action_id": int(a),
                    "delta_dot_rad_s": float(dd),
                    "a_m_s2": float(aa),
                    "od_m": float(info.get("od_m", float("nan"))),
                    "collision": bool(info.get("collision", False)),
                    "reached": bool(info.get("reached", False)),
                    "reached_pose": bool(info.get("reached_pose", False)),
                    "reached_stop": bool(info.get("reached_stop", False)),
                    "stuck": bool(info.get("stuck", False)),
                    "d_goal_m": float(info.get("d_goal_m", float("nan"))),
                    "alpha_rad": float(info.get("alpha_rad", float("nan"))),
                    "cell_size_m": float(getattr(env, "cell_size_m", float("nan"))),
                    "start_x": int(getattr(env, "start_xy", (0, 0))[0]),
                    "start_y": int(getattr(env, "start_xy", (0, 0))[1]),
                    "goal_x": int(getattr(env, "goal_xy", (0, 0))[0]),
                    "goal_y": int(getattr(env, "goal_xy", (0, 0))[1]),
                }
            )
        if info.get("reached"):
            reached = True
            break

    if time_mode == "rollout":
        sync_cuda()
        inference_time_s = float(time.perf_counter() - t_rollout0)

    if bool(reached):
        failure_reason = "reached"
    elif bool(last_collision):
        failure_reason = "collision"
    elif bool(last_stuck):
        failure_reason = "stuck"
    elif bool(truncated) or int(steps) >= int(max_steps):
        failure_reason = "timeout"
    else:
        failure_reason = "not_reached"

    steps_safe = max(1, int(steps))
    debug = {
        "argmax_inadmissible_steps": int(argmax_inadmissible_steps),
        "replacement_topk_steps": int(replacement_topk_steps),
        "replacement_mask_steps": int(replacement_mask_steps),
        "fallback_steps": int(fallback_steps),
        "stop_override_steps": int(stop_override_steps),
        "prefilter_active_steps": int(prefilter_active_steps),
        "argmax_inadmissible_rate": float(argmax_inadmissible_steps) / float(steps_safe),
        "fallback_rate": float(fallback_steps) / float(steps_safe),
        "failure_reason": str(failure_reason),
    }

    if trace_path is not None and trace_rows is not None:
        trace_path = Path(trace_path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(trace_rows).to_csv(trace_path, index=False)
    controls = None
    if t_series is not None and v_series is not None and delta_series is not None:
        controls = ControlTrace(
            t_s=np.asarray(t_series, dtype=np.float64),
            v_m_s=np.asarray(v_series, dtype=np.float64),
            delta_rad=np.asarray(delta_series, dtype=np.float64),
        )
    return RolloutResult(
        path_xy_cells=path,
        compute_time_s=float(inference_time_s),
        reached=bool(reached),
        steps=int(steps),
        path_time_s=float(steps) * dt_s,
        controls=controls,
        collision=bool(last_collision),
        truncated=bool(truncated),
        debug=debug,
    )



def _estimate_path_yaw_rad(path_xy_cells: list[tuple[float, float]]) -> list[float]:
    if not path_xy_cells:
        return []
    if len(path_xy_cells) == 1:
        return [0.0]

    out: list[float] = []
    prev = 0.0
    for i in range(len(path_xy_cells)):
        x0, y0 = path_xy_cells[i]
        if i + 1 < len(path_xy_cells):
            x1, y1 = path_xy_cells[i + 1]
            dx = float(x1) - float(x0)
            dy = float(y1) - float(y0)
        else:
            x1, y1 = path_xy_cells[i - 1]
            dx = float(x0) - float(x1)
            dy = float(y0) - float(y1)
        if abs(float(dx)) + abs(float(dy)) < 1e-12:
            yaw = float(prev)
        else:
            yaw = float(math.atan2(float(dy), float(dx)))
        out.append(float(yaw))
        prev = float(yaw)
    return out


def rollout_td3_local(
    env: AMRBicycleEnv,
    agent,  # TD3Agent with local encoder
    *,
    max_steps: int,
    seed: int,
    reset_options: dict | None = None,
    time_mode: str = "rollout",
    collect_controls: bool = False,
) -> RolloutResult:
    """Rollout TD3-local with V16-C flat obs + continuous actions."""
    obs_raw, info0 = env.reset(seed=seed, options=reset_options)
    obs = env._observe()  # flat 154-dim
    path: list[tuple[float, float]] = []
    dt_s = float(env.model.dt)

    t_series: list[float] | None = None
    v_series: list[float] | None = None
    delta_series: list[float] | None = None
    if collect_controls:
        t_series = [0.0]
        v_series = [float(env._v_m_s)]
        delta_series = [float(env._delta_rad)]

    inference_time_s = 0.0
    if agent.device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    t_rollout0 = time.perf_counter()
    done = False
    truncated = False
    steps = 0
    last_collision = False
    last_stuck = False
    info: dict = {}

    while not (done or truncated) and steps < max_steps:
        steps += 1
        if time_mode == "policy":
            if agent.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

        action = agent.act_flat(obs, explore=False)  # deterministic

        if time_mode == "policy":
            if agent.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_time_s += time.perf_counter() - t0

        delta_dot = float(action[0]) * float(env.model.delta_dot_max_rad_s)
        accel = float(action[1]) * float(env.model.a_max_m_s2)
        _, reward, done, truncated, info = env.step_continuous(
            delta_dot_rad_s=delta_dot, a_m_s2=accel)
        obs = env._observe()

        ax, ay = env._agent_xy_for_plot()
        path.append((float(ax), float(ay)))
        last_collision = bool(info.get("collision", False))
        last_stuck = bool(info.get("stuck", False))

        if collect_controls and t_series is not None:
            t_series.append(float(steps) * dt_s)
            v_series.append(float(info.get("v_m_s", 0.0)))
            delta_series.append(float(info.get("delta_rad", 0.0)))

    if time_mode == "rollout":
        if agent.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_time_s = time.perf_counter() - t_rollout0

    reached = bool(info.get("reached", False)) if steps > 0 else False
    path_time_s = float(steps) * dt_s

    controls = None
    if collect_controls and t_series is not None:
        controls = ControlTrace(
            t_s=np.array(t_series), v_m_s=np.array(v_series),
            delta_rad=np.array(delta_series))

    failure_reason = "success"
    if not reached:
        if last_collision:
            failure_reason = "collision"
        elif last_stuck:
            failure_reason = "stuck"
        elif truncated or steps >= max_steps:
            failure_reason = "timeout"
        else:
            failure_reason = "unknown"

    return RolloutResult(
        path_xy_cells=path,
        compute_time_s=inference_time_s,
        reached=reached,
        steps=steps,
        path_time_s=path_time_s,
        controls=controls,
        collision=last_collision,
        truncated=truncated,
        debug={"failure_reason": failure_reason},
    )


def rollout_sac(
    env: AMRBicycleEnv,
    agent,  # SACAgent
    *,
    max_steps: int,
    seed: int,
    reset_options: dict[str, object] | None = None,
    time_mode: str = "rollout",
    global_map_size: int = 48,
    collect_controls: bool = False,
) -> RolloutResult:
    """Rollout a SAC agent with continuous actions."""
    obs_raw, info0 = env.reset(seed=seed, options=reset_options)
    obs = env.observe_sac(map_size=global_map_size)
    path: list[tuple[float, float]] = [
        (float(env.start_xy[0]), float(env.start_xy[1]))]
    dt_s = float(env.model.dt)

    t_series: list[float] | None = None
    v_series: list[float] | None = None
    delta_series: list[float] | None = None
    if collect_controls:
        t_series = [0.0]
        v_series = [float(env._v_m_s)]
        delta_series = [float(env._delta_rad)]

    inference_time_s = 0.0
    if agent.device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    t_rollout0 = time.perf_counter()
    done = False
    truncated = False
    steps = 0
    last_collision = False
    last_stuck = False

    while not (done or truncated) and steps < max_steps:
        steps += 1
        if time_mode == "policy":
            if agent.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

        action = agent.act(obs, explore=False)  # deterministic

        if time_mode == "policy":
            if agent.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_time_s += time.perf_counter() - t0

        delta_dot = float(action[0]) * float(env.model.delta_dot_max_rad_s)
        accel = float(action[1]) * float(env.model.a_max_m_s2)
        _, reward, done, truncated, info = env.step_continuous(
            delta_dot_rad_s=delta_dot, a_m_s2=accel)
        obs = env.observe_sac(map_size=global_map_size)

        ax, ay = env._agent_xy_for_plot()
        path.append((float(ax), float(ay)))
        last_collision = bool(info.get("collision", False))
        last_stuck = bool(info.get("stuck", False))

        if collect_controls and t_series is not None:
            t_series.append(float(steps) * dt_s)
            v_series.append(float(info.get("v_m_s", 0.0)))
            delta_series.append(float(info.get("delta_rad", 0.0)))

    if time_mode == "rollout":
        if agent.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_time_s = time.perf_counter() - t_rollout0

    reached = bool(info.get("reached", False)) if steps > 0 else False
    path_time_s = float(steps) * dt_s

    controls = None
    if collect_controls and t_series is not None:
        controls = ControlTrace(
            t_s=np.array(t_series), v_m_s=np.array(v_series),
            delta_rad=np.array(delta_series))

    failure_reason = "success"
    if not reached:
        if last_collision:
            failure_reason = "collision"
        elif last_stuck:
            failure_reason = "stuck"
        elif truncated or steps >= max_steps:
            failure_reason = "timeout"
        else:
            failure_reason = "unknown"

    return RolloutResult(
        path_xy_cells=path,
        compute_time_s=inference_time_s,
        reached=reached,
        steps=steps,
        path_time_s=path_time_s,
        controls=controls,
        collision=last_collision,
        truncated=truncated,
        debug={"failure_reason": failure_reason},
    )


def infer_checkpoint_obs_dim(path: Path) -> int:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict) or "q_state_dict" not in payload:
        raise ValueError(f"Unsupported checkpoint format: {path}")

    obs_dim = payload.get("obs_dim")
    if isinstance(obs_dim, (int, float)) and int(obs_dim) > 0:
        return int(obs_dim)

    sd = payload["q_state_dict"]
    w = sd.get("net.0.weight")
    if w is None:
        w = sd.get("feature.0.weight")
    if w is None:
        raise ValueError(f"Could not infer observation dim from checkpoint: {path}")
    return int(w.shape[1])


def forest_legacy_obs_transform(obs: np.ndarray) -> np.ndarray:
    """Map current forest observations (11+n_sectors) -> legacy (7+n_sectors)."""
    x = np.asarray(obs, dtype=np.float32).reshape(-1)
    if x.size < 11:
        return x
    return np.concatenate([x[:7], x[11:]]).astype(np.float32, copy=False)


def mean_kpi(kpis: list[KPI]) -> KPI:
    if not kpis:
        nan = float("nan")
        return KPI(
            avg_path_length=nan,
            path_time_s=nan,
            avg_curvature_1_m=nan,
            planning_time_s=nan,
            tracking_time_s=nan,
            num_corners=nan,
            inference_time_s=nan,
            max_corner_deg=nan,
        )
    return KPI(
        avg_path_length=float(np.mean([k.avg_path_length for k in kpis])),
        path_time_s=float(np.mean([k.path_time_s for k in kpis])),
        avg_curvature_1_m=float(np.mean([k.avg_curvature_1_m for k in kpis])),
        planning_time_s=float(np.mean([k.planning_time_s for k in kpis])),
        tracking_time_s=float(np.mean([k.tracking_time_s for k in kpis])),
        num_corners=float(np.mean([k.num_corners for k in kpis])),
        inference_time_s=float(np.mean([k.inference_time_s for k in kpis])),
        max_corner_deg=float(np.mean([k.max_corner_deg for k in kpis])),
    )


def smooth_path(path: list[tuple[float, float]], *, iterations: int) -> list[tuple[float, float]]:
    if not path:
        return []
    pts = np.array(path, dtype=np.float32)
    sm = chaikin_smooth(pts, iterations=max(0, int(iterations)))
    return [(float(x), float(y)) for x, y in sm]


def plot_env(ax: plt.Axes, grid: np.ndarray, *, title: str) -> None:
    ax.imshow(grid, origin="lower", cmap="gray_r", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlim(-0.5, grid.shape[1] - 0.5)
    ax.set_ylim(-0.5, grid.shape[0] - 0.5)
    ax.set_aspect("equal")
    h, w = grid.shape
    size = int(max(h, w))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=7, integer=True))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=7, integer=True))
    ax.tick_params(axis="both", labelsize=7)
    ax.tick_params(axis="x", labelrotation=45)

    if size <= 60:
        ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
        ax.grid(True, which="minor", alpha=0.18, linewidth=0.35)
        ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
    else:
        ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
        ax.grid(False, which="minor")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")


def draw_vehicle_boxes(
    ax: plt.Axes,
    trace: PathTrace,
    *,
    length_cells: float,
    width_cells: float,
    color: str,
) -> None:
    if not (float(length_cells) > 0.0 and float(width_cells) > 0.0):
        return
    path = trace.path_xy_cells
    if len(path) < 2:
        return

    stride = max(1, int(len(path) / 18))
    hl = 0.5 * float(length_cells)
    hw = 0.5 * float(width_cells)
    alpha = 0.28 if trace.success else 0.18
    ls = "-" if trace.success else ":"

    prev_theta: float | None = None
    for i in range(0, len(path), stride):
        x, y = path[i]
        if i < len(path) - 1:
            x2, y2 = path[i + 1]
            dx = float(x2) - float(x)
            dy = float(y2) - float(y)
        else:
            x2, y2 = path[i - 1]
            dx = float(x) - float(x2)
            dy = float(y) - float(y2)

        if abs(dx) + abs(dy) < 1e-9:
            theta = float(prev_theta) if prev_theta is not None else 0.0
        else:
            theta = float(math.atan2(dy, dx))
        prev_theta = float(theta)

        c = float(math.cos(theta))
        s = float(math.sin(theta))
        corners = [
            (float(x) + c * hl - s * hw, float(y) + s * hl + c * hw),
            (float(x) + c * hl - s * (-hw), float(y) + s * hl + c * (-hw)),
            (float(x) + c * (-hl) - s * (-hw), float(y) + s * (-hl) + c * (-hw)),
            (float(x) + c * (-hl) - s * hw, float(y) + s * (-hl) + c * hw),
        ]
        poly = mpatches.Polygon(
            corners,
            closed=True,
            fill=False,
            edgecolor=color,
            linewidth=0.6,
            alpha=float(alpha),
            linestyle=ls,
        )
        ax.add_patch(poly)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run inference and generate Fig.12 + Table II-style KPIs.")
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON config file. Supports a combined file with {train:{...}, infer:{...}}. CLI flags override config.",
    )
    ap.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Config profile name under configs/ (e.g. forest_a_3000 -> configs/forest_a_3000.json). Overrides configs/config.json.",
    )
    ap.add_argument(
        "--envs",
        nargs="*",
        default=list(FOREST_ENV_ORDER),
        help="Subset of envs: forest_a forest_b forest_c forest_d",
    )
    ap.add_argument(
        "--models",
        type=Path,
        default=Path("outputs"),
        help="Model source: experiment name/dir, run dir, or models dir.",
    )
    ap.add_argument(
        "--ensemble-models",
        type=Path,
        nargs="*",
        default=None,
        help="Additional model dirs for Q-value ensemble averaging (Averaged-DQN).",
    )
    ap.add_argument(
        "--ensemble-mode",
        type=str,
        default="qavg",
        choices=["qavg", "pavg", "vote", "conf", "rank", "qgap", "veto", "topk", "negotiate", "negrelax", "negsoft"],
        help="Ensemble fusion: qavg|pavg|vote|conf|rank|qgap|veto|topk|negotiate|negrelax|negsoft.",
    )
    ap.add_argument(
        "--negotiate-k",
        type=int,
        default=20,
        help="Top-K approval threshold for negotiate mode (default: 20).",
    )
    ap.add_argument(
        "--negotiate-k-max",
        type=int,
        default=30,
        help="Max K for negrelax progressive relaxation (default: 30).",
    )
    ap.add_argument(
        "--negsoft-alpha",
        type=float,
        default=0.7,
        help="DDQN rank weight for negsoft mode; 0=pure M-DQN, 1=pure DDQN (default: 0.7).",
    )
    ap.add_argument(
        "--negsoft-tau",
        type=float,
        default=0.1,
        help="Softmax temperature for M-DQN approval in negsoft mode (default: 0.1).",
    )
    ap.add_argument(
        "--ensemble-adm-prefilter",
        action="store_true",
        default=False,
        help="Pre-filter inadmissible actions before ensemble fusion (V24).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs"),
        help=(
            "Output experiment name/dir. If this resolves to the same experiment as --models, "
            "results are stored under that training run directory."
        ),
    )
    ap.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="If --out/--models is a bare name, store/read it under this directory.",
    )
    ap.add_argument(
        "--timestamp-runs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write into <experiment>/<timestamp>/ (or <train_run>/infer/<timestamp>/) to avoid mixing outputs.",
    )
    ap.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show an inference progress bar (default: on when running in a TTY).",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", type=int, default=5, help="Averaging runs for stochastic methods.")
    ap.add_argument(
        "--plot-run-idx",
        type=int,
        default=0,
        help=(
            "When --random-start-goal is enabled, plot this sample index in fig12_paths.png so all algorithms share "
            "the same (start,goal) pair (default: 0)."
        ),
    )
    ap.add_argument(
        "--plot-pair-runs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Forest-only: when --rand-two-suites is enabled, write one 2-panel path figure per run index "
            "(short + long) so each image contains a short and a long random pair."
        ),
    )
    ap.add_argument(
        "--plot-pair-runs-max",
        type=int,
        default=10,
        help="Maximum number of per-run short+long figures to write when --plot-pair-runs is enabled (<=0 disables cap).",
    )
    ap.add_argument(
        "--plot-run-groups",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Forest-only: write grouped random-run figures, each containing --plot-run-group-size panels "
            "for one environment (e.g., runs 00-03, 04-07, ...)."
        ),
    )
    ap.add_argument(
        "--plot-run-group-size",
        type=int,
        default=4,
        help="Number of run panels per grouped figure when --plot-run-groups is enabled (default: 4).",
    )
    ap.add_argument(
        "--plot-run-groups-with-controls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When --plot-run-groups is enabled, also write grouped control figures "
            "(fig13_controls_..._runs_XX_YY.png)."
        ),
    )
    ap.add_argument(
        "--baselines",
        nargs="*",
        default=[],
        help="Optional baselines to evaluate: astar hybrid_astar rrt_star astar_mpc hybrid_astar_mpc rrt_mpc (or 'all'). Default: none.",
    )
    ap.add_argument(
        "--rl-algos",
        nargs="+",
        default=["mlp-dqn"],
        help=(
            "RL algorithms to evaluate: mlp-dqn mlp-ddqn mlp-pddqn cnn-dqn cnn-ddqn cnn-pddqn (or 'all'). "
            "Legacy aliases: dqn ddqn iddqn cnn-iddqn. Default: mlp-dqn."
        ),
    )
    ap.add_argument(
        "--skip-rl",
        action="store_true",
        help="Skip loading/running RL agents (useful for baseline-only evaluation).",
    )
    ap.add_argument("--baseline-timeout", type=float, default=5.0, help="Planner timeout (seconds).")
    ap.add_argument("--hybrid-max-nodes", type=int, default=200_000, help="Hybrid A* node budget.")
    ap.add_argument("--rrt-max-iter", type=int, default=5_000, help="RRT* iteration budget.")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument(
        "--no-terminate-on-stuck",
        action="store_true",
        help="Forest-only: disable stuck termination (still reports stuck and applies stuck penalty).",
    )
    ap.add_argument("--sensor-range", type=int, default=6)
    ap.add_argument(
        "--n-sectors",
        type=int,
        default=36,
        help="Forest lidar sectors (kept for backwards compatibility; not used by the global-map observation).",
    )
    ap.add_argument(
        "--obs-map-size",
        type=int,
        default=12,
        help="Downsampled global-map observation size (applies to both grid and forest envs).",
    )
    ap.add_argument(
        "--obs-geodesic-goal-dist",
        action="store_true",
        default=False,
        help="Add a geodesic (Dijkstra) goal-distance channel to the map observation.",
    )
    ap.add_argument(
        "--goal-tolerance-m",
        type=float,
        default=1.0,
        help="Forest-only: positional tolerance (meters) to count as 'at goal'.",
    )
    ap.add_argument(
        "--goal-angle-tolerance-deg",
        type=float,
        default=180.0,
        help="Forest-only: heading tolerance (degrees) to count as 'at goal' (180 disables).",
    )
    ap.add_argument(
        "--goal-stop-speed-m-s",
        type=float,
        default=0.05,
        help="Forest-only: max |v| (m/s) required to count as 'stopped' at the goal.",
    )
    ap.add_argument(
        "--goal-stop-delta-deg",
        type=float,
        default=1.0,
        help="Forest-only: max |delta| (degrees) required to count as 'wheels straight' at the goal.",
    )
    ap.add_argument(
        "--forest-action-delta-dot-bins",
        type=int,
        default=7,
        help=(
            "Forest-only: number of discrete steering-rate (delta_dot) levels for the DQN action table "
            "(use an odd number to include 0, e.g. 15)."
        ),
    )
    ap.add_argument(
        "--forest-action-accel-bins",
        type=int,
        default=5,
        help=(
            "Forest-only: number of discrete acceleration (a) levels for the DQN action table "
            "(use an odd number to include 0, e.g. 15)."
        ),
    )
    ap.add_argument(
        "--forest-action-grid-power",
        type=float,
        default=1.0,
        help=(
            "Forest-only: symmetric action-grid shaping power (1.0=linear; >1.0 gives denser levels near 0)."
        ),
    )
    ap.add_argument("--cell-size", type=float, default=1.0, help="Grid cell size in meters.")
    ap.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Torch device selection (default: auto).",
    )
    ap.add_argument("--cuda-device", type=int, default=0, help="CUDA device index (when using --device=cuda).")
    ap.add_argument(
        "--score-time-weight",
        type=float,
        default=0.5,
        help=(
            "Time weight (m/s) for the composite planning_cost metric: "
            "planning_cost = (avg_path_length + w * inference_time_s) / max(success_rate, eps)."
        ),
    )
    ap.add_argument(
        "--composite-w-path-time",
        type=float,
        default=1.0,
        help="Weight for path_time_s in composite_score (default: 1.0).",
    )
    ap.add_argument(
        "--composite-w-avg-curvature",
        type=float,
        default=1.0,
        help="Weight for avg_curvature_1_m in composite_score (default: 1.0).",
    )
    ap.add_argument(
        "--composite-w-planning-time",
        type=float,
        default=1.0,
        help="Weight for planning_time_s in composite_score (default: 1.0).",
    )
    ap.add_argument(
        "--kpi-time-mode",
        choices=("rollout", "policy"),
        default="policy",
        help=(
            "How to measure inference_time_s for RL rollouts. "
            "'rollout' includes the full rollout wall-clock time (including env.step); "
            "'policy' measures only action-selection compute time (Q forward + admissibility checks)."
        ),
    )
    ap.add_argument(
        "--forest-adm-horizon",
        type=int,
        default=15,
        help="Forest-only: admissible-action horizon steps for safe/progress-gated rollouts.",
    )
    ap.add_argument(
        "--forest-adm-persistence",
        type=int,
        default=0,
        help="Forest-only: action persistence steps before decay (0=no decay, full constant rollout).",
    )
    ap.add_argument(
        "--forest-goal-admissible-relax-factor",
        type=float,
        default=1.5,
        help=(
            "Forest-only: when already in goal pose region, allow admissible actions that stay within "
            "factor*goal_tolerance_m (>=1.0)."
        ),
    )
    ap.add_argument(
        "--forest-no-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Forest-only strict mode: disable action replacements/fallback in RL rollout and run pure greedy policy."
        ),
    )
    ap.add_argument(
        "--forest-topk",
        type=int,
        default=10,
        help="Forest-only: try the top-k greedy actions before computing a full admissible-action mask.",
    )
    ap.add_argument(
        "--forest-min-progress-m",
        type=float,
        default=1e-4,
        help="Forest-only: minimum goal-distance progress required by admissible-action gating.",
    )
    ap.add_argument(
        "--forest-min-od-m",
        type=float,
        default=0.0,
        help="Forest-only: minimum clearance (OD) required by admissible-action gating.",
    )
    ap.add_argument(
        "--forest-baseline-hybrid-collision-padding-m",
        type=float,
        default=0.02,
        help="Forest-only: Hybrid A* collision padding (meters) for baseline planning.",
    )
    ap.add_argument(
        "--forest-baseline-rrt-collision-padding-m",
        type=float,
        default=0.0,
        help="Forest-only: RRT* collision padding (meters) for baseline planning (default: 0.0).",
    )
    ap.add_argument(
        "--forest_mpc_horizon_steps",
        type=int,
        default=12,
        help="Forest-only: MPC prediction horizon length (steps).",
    )
    ap.add_argument(
        "--forest_mpc_candidates",
        type=int,
        default=256,
        help="Forest-only: MPC candidate controls evaluated per step.",
    )
    ap.add_argument(
        "--forest_mpc_dt_s",
        type=float,
        default=0.0,
        help="Forest-only: MPC internal dt (<=0 uses env model dt).",
    )
    ap.add_argument(
        "--forest_mpc_w_u",
        type=float,
        default=0.05,
        help="Forest-only: MPC weight for ||u||^2 term.",
    )
    ap.add_argument(
        "--forest_mpc_w_du",
        type=float,
        default=0.5,
        help="Forest-only: MPC weight for ||Δu||^2 term.",
    )
    ap.add_argument(
        "--forest_mpc_w_pos",
        type=float,
        default=6.0,
        help="Forest-only: MPC weight for ||D-Z||^2 position error term.",
    )
    ap.add_argument(
        "--forest_mpc_w_yaw",
        type=float,
        default=1.2,
        help="Forest-only: MPC weight for near-goal yaw error term.",
    )
    ap.add_argument(
        "--forest_mpc_align_dist_m",
        type=float,
        default=1.5,
        help="Forest-only: distance threshold to activate near-goal yaw term.",
    )
    ap.add_argument(
        "--forest_mpc_collision_padding_m",
        type=float,
        default=0.0,
        help="Forest-only: MPC dual-circle collision padding (meters).",
    )
    ap.add_argument(
        "--forest_mpc_goal_lookahead_steps",
        type=int,
        default=2,
        help="Forest-only: MPC reference-path lookahead stride in steps.",
    )
    ap.add_argument(
        "--forest-astar-opt-enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forest-only: enable A* curve optimization for A*-MPC baseline.",
    )
    ap.add_argument(
        "--forest-astar-opt-max-replans",
        type=int,
        default=200,
        help="Forest-only: max replans when A* curve optimization fails.",
    )
    ap.add_argument(
        "--forest-astar-opt-resample-ds-m",
        type=float,
        default=0.2,
        help="Forest-only: resampling step (m) for A* curve optimization.",
    )
    ap.add_argument(
        "--forest-astar-opt-collision-step-m",
        type=float,
        default=0.1,
        help="Forest-only: collision sampling step (m) for A* curve optimization.",
    )
    ap.add_argument(
        "--forest-astar-opt-shortcut-passes",
        type=int,
        default=2,
        help="Forest-only: shortcut passes before curve smoothing.",
    )
    ap.add_argument(
        "--rand-pairs-json",
        type=Path,
        default=None,
        help=(
            "Forest-only: load fixed random start-goal pairs from JSON for fair cross-baseline comparison. "
            "Expected keys: {pairs:[{env,run_idx,start_xy,goal_xy}, ...]}"
        ),
    )
    ap.add_argument(
        "--forest-policy-save-traces",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Forest-only: save per-run executed policy trajectories (x,y,theta,v,delta,action,OD,flags) under "
            "<run_dir>/traces/ as CSV."
        ),
    )
    ap.add_argument(
        "--random-start-goal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forest-only: evaluate on random start/goal pairs (uses --runs samples per environment).",
    )
    ap.add_argument(
        "--rand-min-dist-m",
        type=float,
        default=6.0,
        help="Forest-only: minimum start→goal Euclidean distance (meters) when sampling random pairs.",
    )
    ap.add_argument(
        "--rand-max-dist-m",
        type=float,
        default=0.0,
        help="Forest-only: maximum start→goal Euclidean distance (meters) when sampling random pairs (<=0 disables).",
    )
    ap.add_argument(
        "--rand-edge-margin-m",
        type=float,
        default=0.0,
        help="Forest-only: minimum distance to map boundary (meters) for random start/goal sampling (<=0 disables).",
    )
    ap.add_argument(
        "--rand-fixed-prob",
        type=float,
        default=0.0,
        help="Forest-only: probability of using the canonical fixed start/goal instead of a random pair.",
    )
    ap.add_argument(
        "--rand-tries",
        type=int,
        default=200,
        help="Forest-only: rejection-sampling tries per sample when sampling random start/goal pairs.",
    )
    ap.add_argument(
        "--rand-reject-unreachable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Forest-only: when --random-start-goal is enabled, resample until a baseline planner succeeds "
            "(Hybrid A* by default; A* when only --baselines astar is selected)."
        ),
    )
    ap.add_argument(
        "--rand-reject-policy",
        type=str,
        default=None,
        help=(
            "Forest-only: when --rand-reject-unreachable is enabled, screen random (start,goal) pairs by rolling out a "
            "greedy RL policy instead of running the baseline reachability planner. "
            "Planner aliases 'hybrid_astar' / 'astar' are also accepted to force the screening planner. "
            "Example: --rand-reject-policy cnn-pddqn. "
            "Use 'none' to disable."
        ),
    )
    ap.add_argument(
        "--rand-reject-max-attempts",
        type=int,
        default=5000,
        help="Forest-only: maximum sampling attempts to find reachable random (start,goal) pairs.",
    )
    ap.add_argument(
        "--rand-two-suites",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Forest-only: when --random-start-goal is enabled, evaluate two random-pair suites (short + long) in one run. "
            "This adds '/short' and '/long' rows to KPI tables and plots."
        ),
    )
    ap.add_argument(
        "--rand-short-min-dist-m",
        type=float,
        default=6.0,
        help="Forest-only: minimum start→goal Euclidean distance (meters) for the 'short' random-pair suite.",
    )
    ap.add_argument(
        "--rand-short-max-dist-m",
        type=float,
        default=14.0,
        help="Forest-only: maximum start→goal Euclidean distance (meters) for the 'short' random-pair suite (<=0 disables).",
    )
    ap.add_argument(
        "--rand-long-min-dist-m",
        type=float,
        default=18.0,
        help="Forest-only: minimum start→goal Euclidean distance (meters) for the 'long' random-pair suite.",
    )
    ap.add_argument(
        "--rand-long-max-dist-m",
        type=float,
        default=0.0,
        help="Forest-only: maximum start→goal Euclidean distance (meters) for the 'long' random-pair suite (<=0 disables).",
    )
    ap.add_argument(
        "--rand-short-min-dist-ratio",
        type=float,
        default=0.20,
        help="Forest-only: short-suite lower bound as map diagonal ratio.",
    )
    ap.add_argument(
        "--rand-short-max-dist-ratio",
        type=float,
        default=0.40,
        help="Forest-only: short-suite upper bound as map diagonal ratio.",
    )
    ap.add_argument(
        "--rand-long-min-dist-ratio",
        type=float,
        default=0.60,
        help="Forest-only: long-suite lower bound as map diagonal ratio.",
    )
    ap.add_argument(
        "--rand-long-max-dist-ratio",
        type=float,
        default=0.00,
        help="Forest-only: long-suite upper bound as map diagonal ratio (<=0 disables upper bound).",
    )
    ap.add_argument(
        "--self-check",
        action="store_true",
        help="Print CUDA/runtime info and exit (use to verify CUDA setup).",
    )
    # SAC-specific args (used by v8 config)
    ap.add_argument("--forest-action-mode", type=str, default="discrete")
    ap.add_argument("--global-map-size", type=int, default=48)
    ap.add_argument("--global-map-channels", type=int, default=3)
    return ap


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = build_parser()

    pre_args, _ = ap.parse_known_args(argv)
    # Self-check should be independent of configs/config.json and profiles.
    # Users can still force CUDA validation via `--device cuda`.
    if bool(getattr(pre_args, "self_check", False)):
        args = ap.parse_args(argv)
        info = torch_runtime_info()
        print(f"torch={info.torch_version}")
        print(f"cuda_available={info.cuda_available}")
        print(f"torch_cuda_version={info.torch_cuda_version}")
        print(f"cuda_device_count={info.device_count}")
        if info.device_names:
            print("cuda_devices=" + ", ".join(info.device_names))
        arch = torch_cuda_arch_info(device_index=int(args.cuda_device))
        if arch is not None:
            print(f"cuda_device_sm={arch.device_sm}")
            if arch.build_arch_list:
                print("torch_cuda_arch_list=" + " ".join(arch.build_arch_list))
            print(f"cuda_arch_supported={arch.device_sm_in_build}")
            if not arch.device_sm_in_build:
                print(
                    f"[self-check] NOTE: GPU={arch.device_name}({arch.device_sm}) 不在当前 PyTorch 预编译架构列表中；"
                    "通常仍可运行（PTX/driver JIT），但首次运行可能更慢。若出现 CUDA kernel 相关报错，请升级 PyTorch/CUDA。",
                    file=sys.stderr,
                    flush=True,
                )
        try:
            device = select_device(device=args.device, cuda_device=args.cuda_device)
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(f"device_ok={device}")
        return 0
    try:
        config_path = resolve_config_path(config=getattr(pre_args, "config", None), profile=getattr(pre_args, "profile", None))
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc))
    if config_path is not None:
        cfg_raw = load_json(Path(config_path))
        cfg = select_section(cfg_raw, section="infer")
        deprecated_keys = (
            "forest_baseline_local_replan",
            "forest_baseline_replan_every_steps",
            "forest_baseline_replan_lookahead_m",
            "forest_baseline_replan_timeout_s",
            "forest_baseline_replan_max_nodes",
            "forest_baseline_rollout",
            "forest_baseline_controller",
            "forest_baseline_mpc_profile",
            "forest_baseline_curvature_smoothing_num",
            "forest_baseline_mpc_candidates",
            "forest_baseline_mpc_horizon_steps",
            "forest_baseline_mpc_dt_s",
            "forest_baseline_mpc_speed_scale",
            "forest_baseline_mpc_error_weight_scale",
            "forest_baseline_mpc_input_weight_scale",
            "forest_baseline_mpc_max_steer_deg",
            "forest_baseline_mpc_steer_rate_limit_deg_s",
            "forest_baseline_mpc_osqp_eps_abs",
            "forest_baseline_mpc_osqp_eps_rel",
            "forest_baseline_mpc_osqp_polish",
            "forest_baseline_mpc_reverse_recovery",
            "forest_baseline_mpc_reverse_min_od_m",
            "forest_baseline_mpc_reverse_no_progress_steps",
            "forest_baseline_mpc_reverse_hold_steps",
            "forest_baseline_mpc_reverse_speed_scale",
            "forest_baseline_mpc_reverse_cooldown_steps",
            "forest_baseline_mpc_reverse_predict_steps",
            "forest_baseline_planner_fallback_hybrid",
            "forest_baseline_save_traces",
        )
        _missing = object()
        removed: list[str] = []
        for k in deprecated_keys:
            if cfg.pop(k, _missing) is not _missing:
                removed.append(k)
        if removed:
            print(
                "[infer] NOTE: deprecated baseline config key(s) ignored.",
                file=sys.stderr,
                flush=True,
            )
        apply_config_defaults(ap, cfg, strict=True)

    args = ap.parse_args(argv)
    if int(getattr(args, "plot_run_idx", 0)) < 0:
        raise SystemExit("--plot-run-idx must be >= 0")
    if bool(getattr(args, "plot_run_groups", False)):
        if int(getattr(args, "plot_run_group_size", 0)) < 1:
            raise SystemExit("--plot-run-group-size must be >= 1.")
        if int(getattr(args, "runs", 0)) < 1:
            raise SystemExit("--plot-run-groups requires --runs >= 1.")
        if not bool(getattr(args, "random_start_goal", False)):
            raise SystemExit("--plot-run-groups requires --random-start-goal.")
    forest_envs = set(FOREST_ENV_ORDER)
    if int(args.max_steps) == 300 and args.envs and all(str(e) in forest_envs for e in args.envs):
        args.max_steps = 600
    canonical_all = ("mlp-dqn", "mlp-ddqn", "mlp-pddqn", "mlp-mdqn", "cnn-dqn", "cnn-ddqn", "cnn-pddqn", "cnn-mdqn", "cnn-sac", "cnn-td3", "cnn-td3-local")
    raw_algos = [str(a).lower().strip() for a in (args.rl_algos or [])]
    if any(a == "all" for a in raw_algos):
        raw_algos = list(canonical_all[:-3])  # exclude cnn-sac, cnn-td3, cnn-td3-local from "all"

    rl_algos: list[str] = []
    unknown = []
    for a in raw_algos:
        if a in ("cnn-sac", "cnn-td3", "cnn-td3-local"):
            if a not in rl_algos:
                rl_algos.append(a)
            continue
        try:
            canonical, _arch, _base, _legacy = parse_rl_algo(a)
        except ValueError:
            unknown.append(a)
            continue
        if canonical not in rl_algos:
            rl_algos.append(canonical)

    if unknown:
        raise SystemExit(
            f"Unknown --rl-algos value(s): {', '.join(unknown)}. Choose from: "
            f"{' '.join(canonical_all)} (or 'all'). Legacy aliases: dqn ddqn iddqn cnn-iddqn."
        )
    if not rl_algos:
        raise SystemExit(f"No RL algorithms selected (choose from: {' '.join(canonical_all)}).")
    args.rl_algos = rl_algos

    progress = bool(sys.stderr.isatty()) if getattr(args, "progress", None) is None else bool(args.progress)
    tqdm = None
    if progress:
        try:
            from tqdm import tqdm as _tqdm  # type: ignore
        except Exception:
            tqdm = None
        else:
            tqdm = _tqdm

    def progress_write(msg: str) -> None:
        if not progress:
            return
        if tqdm is not None:
            tqdm.write(str(msg), file=sys.stderr)
        else:
            print(str(msg), file=sys.stderr, flush=True)

    rand_reject_policy: str | None = None
    rand_reject_planner: str | None = None
    raw_policy = getattr(args, "rand_reject_policy", None)
    if raw_policy is not None:
        raw_policy = str(raw_policy).strip().lower()
        if raw_policy in {"none", "off", "false", "0", ""}:
            raw_policy = ""
    if raw_policy:
        planner_aliases = {
            "astar": "astar",
            "a*": "astar",
            "grid_astar": "astar",
            "grid-a-star": "astar",
            "hybrid_astar": "hybrid_astar",
            "hybrid-a-star": "hybrid_astar",
            "hybrid": "hybrid_astar",
            "ha": "hybrid_astar",
        }
        planner_mode = planner_aliases.get(str(raw_policy))
        if planner_mode is not None:
            rand_reject_planner = str(planner_mode)
        else:
            try:
                canonical, _arch, _base, _legacy = parse_rl_algo(str(raw_policy))
            except ValueError as exc:
                raise SystemExit(
                    f"Unknown --rand-reject-policy value {raw_policy!r}. Choose from: "
                    f"{' '.join(canonical_all)}, astar, hybrid_astar (or 'none')."
                ) from exc
            rand_reject_policy = str(canonical)
    args.rand_reject_policy = rand_reject_policy
    setattr(args, "rand_reject_planner", rand_reject_planner)

    baseline_aliases = {
        "astar": "astar",
        "a*": "astar",
        "grid_astar": "astar",
        "grid-a-star": "astar",
        "hybrid_astar": "hybrid_astar",
        "hybrid-a-star": "hybrid_astar",
        "hybrid": "hybrid_astar",
        "ha": "hybrid_astar",
        "rrt_star": "rrt_star",
        "rrt*": "rrt_star",
        "rrt": "rrt_star",
        "ss-rrt*": "rrt_star",
        "ss_rrt_star": "rrt_star",
        "ss_rrt*": "rrt_star",
        "astar_mpc": "astar_mpc",
        "a*-mpc": "astar_mpc",
        "a_star_mpc": "astar_mpc",
        "rrt_mpc": "rrt_mpc",
        "rrt-mpc": "rrt_mpc",
        "hybrid_astar_mpc": "hybrid_astar_mpc",
        "hybrid-a*-mpc": "hybrid_astar_mpc",
        "hybrid_mpc": "hybrid_astar_mpc",
        "all": "all",
    }
    baselines: list[str] = []
    has_all_baseline = False
    for raw in args.baselines:
        key = str(raw).strip().lower()
        if not key:
            continue
        mapped = baseline_aliases.get(key)
        if mapped is None:
            raise SystemExit(
                f"Unknown baseline {raw!r}. Options: astar, hybrid_astar, rrt_star, astar_mpc, hybrid_astar_mpc, rrt_mpc, all "
                "(aliases: a*, hybrid, rrt, rrt*, a*-mpc, rrt-mpc, hybrid-a*-mpc)."
            )
        if mapped == "all":
            has_all_baseline = True
            continue
        if mapped not in baselines:
            baselines.append(mapped)

    if has_all_baseline:
        merged = ["astar", "hybrid_astar", "rrt_star", "astar_mpc", "hybrid_astar_mpc", "rrt_mpc"]
        for b in baselines:
            if b not in merged:
                merged.append(b)
        baselines = merged

    if bool(args.skip_rl) and not baselines:
        raise SystemExit("--skip-rl requires at least one baseline via --baselines (e.g., --baselines all).")

    if bool(getattr(args, "rand_two_suites", False)):
        if not bool(getattr(args, "random_start_goal", False)):
            raise SystemExit("--rand-two-suites requires --random-start-goal.")
        if not args.envs or any(str(e) not in forest_envs for e in args.envs):
            raise SystemExit("--rand-two-suites is forest-only (use e.g. --envs forest_a ...).")
        if int(getattr(args, "runs", 0)) <= 0:
            raise SystemExit("--rand-two-suites requires --runs >= 1.")
        short_min = float(getattr(args, "rand_short_min_dist_m", 0.0))
        short_max = float(getattr(args, "rand_short_max_dist_m", 0.0))
        long_min = float(getattr(args, "rand_long_min_dist_m", 0.0))
        long_max = float(getattr(args, "rand_long_max_dist_m", 0.0))
        if short_max > 0.0 and short_min > short_max:
            raise SystemExit("--rand-short-min-dist-m must be <= --rand-short-max-dist-m (or disable max via <=0).")
        if long_max > 0.0 and long_min > long_max:
            raise SystemExit("--rand-long-min-dist-m must be <= --rand-long-max-dist-m (or disable max via <=0).")

    if bool(getattr(args, "rand_two_suites", False)):
        expanded_envs: list[str] = []
        for e in (args.envs or []):
            base = str(e).split("::", 1)[0].strip()
            if not base:
                continue
            expanded_envs.append(f"{base}::short")
            expanded_envs.append(f"{base}::long")
        args.envs = expanded_envs

    try:
        device = select_device(device=args.device, cuda_device=args.cuda_device)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    progress_write(
        f"[infer] device={device} envs={len(args.envs or [])} runs={int(getattr(args, 'runs', 0))} "
        f"rl_algos={len(args.rl_algos or [])} baselines={len(baselines)} "
        f"random_start_goal={bool(getattr(args, 'random_start_goal', False))} "
        f"rand_two_suites={bool(getattr(args, 'rand_two_suites', False))} "
        f"rand_reject_unreachable={bool(getattr(args, 'rand_reject_unreachable', False))} "
        f"rand_reject_policy={getattr(args, 'rand_reject_policy', None)!r} "
        f"rand_reject_planner={getattr(args, 'rand_reject_planner', None)!r}"
    )

    requested_experiment_dir = resolve_experiment_dir(args.out, runs_root=args.runs_root)
    models_dir: Path | None = None
    if not bool(args.skip_rl):
        models_dir = resolve_models_dir(args.models, runs_root=args.runs_root)

        models_run_dir = models_dir.parent
        models_experiment_dir = models_run_dir.parent

        # If the output points to the same experiment (timestamped runs) or the same run dir (no-timestamp runs),
        # keep inference outputs attached to the training run.
        requested_resolved = requested_experiment_dir.resolve(strict=False)
        models_run_resolved = models_run_dir.resolve(strict=False)
        models_experiment_resolved = models_experiment_dir.resolve(strict=False)

        if requested_resolved == models_run_resolved or requested_resolved == models_experiment_resolved:
            # Keep inference outputs attached to the training run to avoid creating a sibling timestamped run.
            experiment_dir = models_run_dir / "infer"
        else:
            experiment_dir = requested_experiment_dir
    else:
        experiment_dir = requested_experiment_dir

    run_paths = create_run_dir(experiment_dir, timestamp_runs=args.timestamp_runs)
    out_dir = run_paths.run_dir

    (out_dir / "configs").mkdir(parents=True, exist_ok=True)
    args_payload: dict[str, object] = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            args_payload[k] = str(v)
        elif isinstance(v, list) and v and isinstance(v[0], Path):
            args_payload[k] = [str(p) for p in v]
        else:
            args_payload[k] = v
    (out_dir / "configs" / "run.json").write_text(
        json.dumps(
            {
                "kind": "infer",
                "argv": list(sys.argv),
                "experiment_dir": str(run_paths.experiment_dir),
                "run_dir": str(run_paths.run_dir),
                "models_dir": (str(models_dir) if models_dir is not None else None),
                "args": args_payload,
                "torch": asdict(torch_runtime_info()),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    rows: list[dict[str, object]] = []
    rows_runs: list[dict[str, object]] = []
    paths_for_plot: dict[tuple[str, int], dict[str, PathTrace]] = {}
    controls_for_plot: dict[tuple[str, int], dict[str, ControlTrace]] = {}
    plot_meta: dict[tuple[str, int], dict[str, float]] = {}

    total_envs = int(len(args.envs or []))
    for env_idx, env_name in enumerate(args.envs):
        env_case = str(env_name)
        suite_tag: str | None = None
        env_base = str(env_case)
        if "::" in env_case:
            env_base, suite_tag_raw = env_case.split("::", 1)
            env_base = str(env_base).strip()
            suite_tag = str(suite_tag_raw).strip() or None

        env_label = f"Env. ({env_base})" if suite_tag is None else f"Env. ({env_base})/{suite_tag}"
        progress_write(f"[infer] {env_label} ({env_idx + 1}/{max(1, total_envs)})")

        rand_min_dist_m = float(getattr(args, "rand_min_dist_m", 0.0))
        rand_max_dist_m = float(getattr(args, "rand_max_dist_m", 0.0))
        if suite_tag == "short":
            rand_min_dist_m = float(getattr(args, "rand_short_min_dist_m", rand_min_dist_m))
            rand_max_dist_m = float(getattr(args, "rand_short_max_dist_m", rand_max_dist_m))
        elif suite_tag == "long":
            rand_min_dist_m = float(getattr(args, "rand_long_min_dist_m", rand_min_dist_m))
            rand_max_dist_m = float(getattr(args, "rand_long_max_dist_m", rand_max_dist_m))

        spec = get_map_spec(env_base)
        if env_base in FOREST_ENV_ORDER:
            env = AMRBicycleEnv(
                spec,
                max_steps=args.max_steps,
                cell_size_m=0.1,
                sensor_range_m=float(args.sensor_range),
                n_sectors=args.n_sectors,
                obs_map_size=int(args.obs_map_size),
                obs_geodesic_goal_dist=bool(getattr(args, "obs_geodesic_goal_dist", False)),
                goal_tolerance_m=float(args.goal_tolerance_m),
                goal_angle_tolerance_deg=float(args.goal_angle_tolerance_deg),
                goal_stop_speed_m_s=float(args.goal_stop_speed_m_s),
                goal_stop_delta_deg=float(args.goal_stop_delta_deg),
                goal_admissible_relax_factor=float(args.forest_goal_admissible_relax_factor),
                terminate_on_stuck=not bool(getattr(args, "no_terminate_on_stuck", False)),
                action_delta_dot_bins=int(args.forest_action_delta_dot_bins),
                action_accel_bins=int(args.forest_action_accel_bins),
                action_grid_power=float(args.forest_action_grid_power),
            )
            cell_size_m = 0.1
            diag_m = max(1e-6, float(getattr(env, "_diag_m", 1.0)))
            if suite_tag == "short":
                min_ratio = float(getattr(args, "rand_short_min_dist_ratio", 0.20))
                max_ratio = float(getattr(args, "rand_short_max_dist_ratio", 0.40))
                rand_min_dist_m = max(float(rand_min_dist_m), float(min_ratio) * float(diag_m))
                rand_max_dist_m = float(max_ratio) * float(diag_m)
            elif suite_tag == "long":
                min_ratio = float(getattr(args, "rand_long_min_dist_ratio", 0.60))
                max_ratio = float(getattr(args, "rand_long_max_dist_ratio", 0.00))
                rand_min_dist_m = max(float(rand_min_dist_m), float(min_ratio) * float(diag_m))
                if float(max_ratio) > 0.0:
                    rand_max_dist_m = float(max_ratio) * float(diag_m)
                else:
                    rand_max_dist_m = 0.0
        else:
            env = AMRGridEnv(
                spec,
                sensor_range=args.sensor_range,
                max_steps=args.max_steps,
                reward=RewardWeights(),
                cell_size=args.cell_size,
                safe_distance=0.6,
                obs_map_size=int(args.obs_map_size),
                terminate_on_collision=False,
            )
            cell_size_m = float(args.cell_size)
        grid = spec.obstacle_grid()

        env_paths_by_run: dict[int, dict[str, PathTrace]] = {}
        base_meta: dict[str, float] = {"cell_size_m": float(cell_size_m)}
        if isinstance(env, AMRBicycleEnv):
            base_meta["goal_tol_cells"] = float(env.goal_tolerance_m) / float(cell_size_m)
            fp = forest_two_circle_footprint()
            base_meta["veh_length_cells"] = float(fp.length) / float(cell_size_m)
            base_meta["veh_width_cells"] = float(fp.width) / float(cell_size_m)
        else:
            base_meta["goal_tol_cells"] = 0.5
            base_meta["veh_length_cells"] = 0.0
            base_meta["veh_width_cells"] = 0.0

        plot_run_idx = int(getattr(args, "plot_run_idx", 0))
        plot_run_indices: list[int] = [int(plot_run_idx)]
        multi_pair_plot = (
            bool(getattr(args, "random_start_goal", False))
            and isinstance(env, AMRBicycleEnv)
            and int(args.runs) >= 4
            and int(len(args.envs)) == 1
        )
        if multi_pair_plot:
            plot_run_indices = [(int(plot_run_idx) + k) % int(args.runs) for k in range(4)]

        # Plotting: store path traces for specific run indices to keep memory bounded.
        # - `plot_run_indices` drives the main Fig.12/Fig.13 panels.
        # - `plot_pair_runs` wants per-run path figures, but doesn't require control traces.
        path_run_indices: set[int] = set(plot_run_indices)
        control_run_indices: set[int] = set(plot_run_indices)
        if bool(getattr(args, "plot_run_groups", False)) and int(args.runs) > 0:
            all_runs = range(int(args.runs))
            path_run_indices.update(all_runs)
            control_run_indices.update(all_runs)
        if (
            bool(getattr(args, "plot_pair_runs", False))
            and bool(getattr(args, "random_start_goal", False))
            and bool(getattr(args, "rand_two_suites", False))
            and int(args.runs) > 0
        ):
            per_run_cap = int(getattr(args, "plot_pair_runs_max", 10))
            per_run_n = int(args.runs)
            if per_run_cap > 0:
                per_run_n = min(int(per_run_n), int(per_run_cap))
            path_run_indices.update(range(int(per_run_n)))

        for idx in sorted(path_run_indices):
            env_paths_by_run.setdefault(int(idx), {})

        # Optional: sample a fixed set of (start, goal) pairs for fair random-start/goal evaluation.
        reset_options_list: list[dict[str, object] | None] = [None] * int(max(0, int(args.runs)))
        precomputed_hybrid_paths: list[PlannerResult] | None = None
        precomputed_grid_paths: list[PlannerResult] | None = None
        plot_start_xy = tuple(spec.start_xy)
        plot_goal_xy = tuple(spec.goal_xy)
        if bool(getattr(args, "random_start_goal", False)) and isinstance(env, AMRBicycleEnv) and int(args.runs) > 0:
            rand_max = None if float(rand_max_dist_m) <= 0.0 else float(rand_max_dist_m)
            if plot_run_idx >= int(args.runs):
                raise SystemExit(
                    f"--plot-run-idx={plot_run_idx} must be < --runs={int(args.runs)} when --random-start-goal is enabled."
                )
            reset_options_list = []
            reject_unreachable = bool(getattr(args, "rand_reject_unreachable", False))
            max_attempts = max(1, int(getattr(args, "rand_reject_max_attempts", 5000)))
            reject_policy = getattr(args, "rand_reject_policy", None)
            reject_planner = getattr(args, "rand_reject_planner", None)
            use_policy_reject = bool(reject_unreachable) and isinstance(reject_policy, str) and bool(reject_policy)
            reject_with_grid_astar = bool(reject_unreachable) and (
                (str(reject_planner) == "astar")
                or (
                    (reject_planner is None)
                    and ("astar" in baselines)
                    and ("hybrid_astar" not in baselines)
                )
            )

            if use_policy_reject:
                progress_write(
                    f"[infer] Sampling {int(args.runs)} random pairs for {env_label} with policy screening "
                    f"({reject_policy}, max_attempts={max_attempts})."
                )
                if "hybrid_astar" in baselines or "astar" in baselines:
                    screening_name = "Hybrid A*" if "hybrid_astar" in baselines else "A*"
                    progress_write(
                        "[infer] Note: --rand-reject-policy screens reachability by an RL policy; "
                        f"{screening_name} may still fail on some sampled pairs. "
                        f"Use --rand-reject-policy none to screen with {screening_name} instead."
                    )
            elif reject_unreachable:
                screening_name = "A*" if bool(reject_with_grid_astar) else "Hybrid A*"
                progress_write(
                    f"[infer] Sampling {int(args.runs)} random pairs for {env_label} with {screening_name} screening "
                    f"(timeout={float(args.baseline_timeout):.2f}s, max_attempts={max_attempts})."
                )
            else:
                progress_write(f"[infer] Sampling {int(args.runs)} random pairs for {env_label} (no screening).")

            if getattr(args, "rand_pairs_json", None) is not None:
                progress_write(
                    f"[infer] Using fixed random pairs from {getattr(args, 'rand_pairs_json')} for {env_label}."
                )

            policy_reject_agent: DQNFamilyAgent | None = None
            if use_policy_reject:
                if bool(args.skip_rl) or models_dir is None:
                    raise SystemExit("--rand-reject-policy requires RL models (disable --skip-rl and set --models).")

                env_obs_dim = int(env.observation_space.shape[0])
                n_actions = int(env.action_space.n)
                agent_cfg = AgentConfig()

                p = models_dir / env_base / f"{reject_policy}.pt"
                if not p.exists():
                    legacy = {
                        "mlp-dqn": "dqn",
                        "mlp-ddqn": "ddqn",
                        "mlp-pddqn": "iddqn",
                        "cnn-pddqn": "cnn-iddqn",
                    }.get(str(reject_policy))
                    if legacy is not None:
                        p_legacy = models_dir / env_base / f"{legacy}.pt"
                        if p_legacy.exists():
                            p = p_legacy
                if not p.exists():
                    raise FileNotFoundError(f"Missing --rand-reject-policy model: {p}")

                ckpt_obs_dim = infer_checkpoint_obs_dim(p)
                if int(ckpt_obs_dim) != int(env_obs_dim):
                    raise RuntimeError(
                        f"--rand-reject-policy checkpoint expects obs_dim={ckpt_obs_dim} but env provides obs_dim={env_obs_dim} "
                        f"for {env_base!r}. Re-train models to match the environment observation space."
                    )

                policy_reject_agent = DQNFamilyAgent(
                    str(reject_policy),
                    env_obs_dim,
                    n_actions,
                    config=agent_cfg,
                    seed=int(args.seed),
                    device=device,
                )
                policy_reject_agent.load(p)

            if reject_unreachable and not use_policy_reject:
                grid_map = grid_map_from_obstacles(grid_y0_bottom=grid, cell_size_m=float(cell_size_m))
                if bool(reject_with_grid_astar):
                    precomputed_grid_paths = []
                else:
                    precomputed_hybrid_paths = []
                    params = default_ackermann_params(
                        wheelbase_m=float(env.model.wheelbase_m),
                        delta_max_rad=float(env.model.delta_max_rad),
                        v_max_m_s=float(env.model.v_max_m_s),
                    )
                    footprint = forest_two_circle_footprint()
                    goal_xy_tol_m = float(env.goal_tolerance_m)
                    goal_theta_tol_rad = float(env.goal_angle_tolerance_rad)

            fixed_pairs: list[dict[str, object]] | None = None
            fixed_pairs_path = getattr(args, "rand_pairs_json", None)
            if fixed_pairs_path is not None:
                pairs_payload = load_json(Path(fixed_pairs_path))
                raw_pairs = pairs_payload.get("pairs")
                if not isinstance(raw_pairs, list):
                    raise SystemExit("--rand-pairs-json must contain key 'pairs' as a list.")
                fixed_pairs = []
                for idx, item in enumerate(raw_pairs):
                    if not isinstance(item, dict):
                        raise SystemExit(f"--rand-pairs-json pairs[{idx}] must be an object.")
                    env_name_raw = str(item.get("env", "")).strip()
                    if env_name_raw != str(env_base):
                        continue
                    run_idx_raw = item.get("run_idx", idx)
                    sx = item.get("start_xy")
                    gx = item.get("goal_xy")
                    if not isinstance(sx, (list, tuple)) or len(sx) != 2:
                        raise SystemExit(f"--rand-pairs-json pairs[{idx}].start_xy must be [x, y].")
                    if not isinstance(gx, (list, tuple)) or len(gx) != 2:
                        raise SystemExit(f"--rand-pairs-json pairs[{idx}].goal_xy must be [x, y].")
                    fixed_pairs.append(
                        {
                            "env": str(env_name_raw),
                            "run_idx": int(run_idx_raw),
                            "start_xy": (int(sx[0]), int(sx[1])),
                            "goal_xy": (int(gx[0]), int(gx[1])),
                        }
                    )
                fixed_pairs.sort(key=lambda r: int(r["run_idx"]))
                if bool(getattr(args, "random_start_goal", False)) and int(args.runs) > 0:
                    if len(fixed_pairs) < int(args.runs):
                        raise SystemExit(
                            f"--rand-pairs-json provides {len(fixed_pairs)} usable pairs for env {env_base!r}, "
                            f"but --runs={int(args.runs)}."
                        )

            pair_pbar = None
            if tqdm is not None:
                pair_pbar = tqdm(
                    total=int(args.runs),
                    desc=f"Sample pairs {env_label}",
                    unit="pair",
                    dynamic_ncols=True,
                    leave=True,
                )

            attempts = 0
            while len(reset_options_list) < int(args.runs) and attempts < max_attempts:
                if fixed_pairs is not None:
                    item = fixed_pairs[len(reset_options_list)]
                    start_xy = (int(item["start_xy"][0]), int(item["start_xy"][1]))  # type: ignore[index]
                    goal_xy = (int(item["goal_xy"][0]), int(item["goal_xy"][1]))  # type: ignore[index]
                    opts = {"start_xy": start_xy, "goal_xy": goal_xy}
                    reset_options_list.append(opts)
                    attempts += 1
                    if pair_pbar is not None:
                        pair_pbar.update(1)
                        pair_pbar.set_postfix({"attempts": attempts, "mode": "fixed"})
                    continue

                if pair_pbar is None and progress and attempts > 0 and (attempts % 50 == 0):
                    progress_write(
                        f"[infer] Sampling {env_label}: accepted {len(reset_options_list)}/{int(args.runs)}, "
                        f"attempts {attempts}/{max_attempts}."
                    )
                env.reset(
                    seed=int(args.seed) + 90_000 + int(attempts),
                    options={
                        "random_start_goal": True,
                        "rand_min_dist_m": float(rand_min_dist_m),
                        "rand_max_dist_m": rand_max,
                        "rand_fixed_prob": float(args.rand_fixed_prob),
                        "rand_tries": int(args.rand_tries),
                        "rand_edge_margin_m": float(args.rand_edge_margin_m),
                    },
                )

                start_xy = (int(env.start_xy[0]), int(env.start_xy[1]))
                goal_xy = (int(env.goal_xy[0]), int(env.goal_xy[1]))

                # When the sampling constraints are too strict, the env falls back to the canonical
                # (start,goal) pair after exhausting `rand_tries`. That defeats the purpose of
                # random-pair evaluation and also breaks the short/long suite separation.
                if float(getattr(args, "rand_fixed_prob", 0.0)) <= 0.0:
                    if start_xy == (int(spec.start_xy[0]), int(spec.start_xy[1])) and goal_xy == (
                        int(spec.goal_xy[0]),
                        int(spec.goal_xy[1]),
                    ):
                        attempts += 1
                        if pair_pbar is not None and (attempts % 10 == 0):
                            pair_pbar.set_postfix({"accepted": len(reset_options_list), "attempts": attempts})
                        continue

                    dist0 = float(
                        math.hypot(
                            float(goal_xy[0] - start_xy[0]) * float(env.cell_size_m),
                            float(goal_xy[1] - start_xy[1]) * float(env.cell_size_m),
                        )
                    )
                    if not math.isfinite(dist0):
                        attempts += 1
                        if pair_pbar is not None and (attempts % 10 == 0):
                            pair_pbar.set_postfix({"accepted": len(reset_options_list), "attempts": attempts})
                        continue
                    if float(dist0) + 1e-6 < float(rand_min_dist_m):
                        attempts += 1
                        if pair_pbar is not None and (attempts % 10 == 0):
                            pair_pbar.set_postfix({"accepted": len(reset_options_list), "attempts": attempts})
                        continue
                    if rand_max is not None and float(rand_max) > 0.0 and float(dist0) - 1e-6 > float(rand_max):
                        attempts += 1
                        if pair_pbar is not None and (attempts % 10 == 0):
                            pair_pbar.set_postfix({"accepted": len(reset_options_list), "attempts": attempts})
                        continue

                if policy_reject_agent is not None:
                    roll = rollout_agent(
                        env,
                        policy_reject_agent,
                        max_steps=int(args.max_steps),
                        seed=int(args.seed) + 95_000 + int(attempts),
                        reset_options={"start_xy": start_xy, "goal_xy": goal_xy},
                        time_mode="rollout",
                        forest_adm_horizon=int(args.forest_adm_horizon),
                        forest_adm_persistence=int(args.forest_adm_persistence),
                        forest_topk=int(args.forest_topk),
                        forest_min_od_m=float(args.forest_min_od_m),
                        forest_min_progress_m=float(args.forest_min_progress_m),
                        forest_no_fallback=bool(getattr(args, "forest_no_fallback", False)),
                        collect_controls=False,
                    )
                    if not bool(roll.reached):
                        attempts += 1
                        if pair_pbar is not None and (attempts % 10 == 0):
                            pair_pbar.set_postfix({"accepted": len(reset_options_list), "attempts": attempts})
                        continue

                if reject_unreachable and not use_policy_reject:
                    if bool(reject_with_grid_astar):
                        res = plan_grid_astar(
                            grid_map=grid_map,
                            start_xy=start_xy,
                            goal_xy=goal_xy,
                            timeout_s=float(args.baseline_timeout),
                            seed=int(args.seed),
                            replan_attempt=int(attempts),
                        )
                        if not bool(res.success):
                            attempts += 1
                            if pair_pbar is not None and (attempts % 10 == 0):
                                pair_pbar.set_postfix({"accepted": len(reset_options_list), "attempts": attempts})
                            continue
                        if precomputed_grid_paths is not None:
                            precomputed_grid_paths.append(res)
                    else:
                        res = plan_hybrid_astar(
                            grid_map=grid_map,
                            footprint=footprint,
                            params=params,
                            start_xy=start_xy,
                            goal_xy=goal_xy,
                            goal_theta_rad=None,
                            start_theta_rad=None,
                            goal_xy_tol_m=goal_xy_tol_m,
                            goal_theta_tol_rad=goal_theta_tol_rad,
                            timeout_s=float(args.baseline_timeout),
                            max_nodes=int(args.hybrid_max_nodes),
                        )
                        if not bool(res.success):
                            attempts += 1
                            if pair_pbar is not None and (attempts % 10 == 0):
                                pair_pbar.set_postfix({"accepted": len(reset_options_list), "attempts": attempts})
                            continue
                        if precomputed_hybrid_paths is not None:
                            precomputed_hybrid_paths.append(res)

                opts: dict[str, object] = {"start_xy": start_xy, "goal_xy": goal_xy}
                reset_options_list.append(opts)
                attempts += 1
                if pair_pbar is not None:
                    pair_pbar.update(1)
                    pair_pbar.set_postfix({"attempts": attempts})
                else:
                    progress_write(
                        f"[infer] Sampled {len(reset_options_list)}/{int(args.runs)} pairs for {env_label} "
                        f"(attempts {attempts}/{max_attempts})."
                    )

            if pair_pbar is not None:
                pair_pbar.close()

            if len(reset_options_list) < int(args.runs):
                raise RuntimeError(
                    f"Could not sample {int(args.runs)} reachable random (start,goal) pairs for {env_name!r} "
                    f"after {attempts} attempts (rand_min_dist_m={float(rand_min_dist_m):.2f}, rand_max_dist_m={rand_max}). "
                    "Try increasing --rand-tries, adjusting the distance bounds, or disabling screening via --no-rand-reject-unreachable."
                )
            if reset_options_list:
                plot_start_xy = tuple(reset_options_list[plot_run_idx]["start_xy"])  # type: ignore[arg-type]
                plot_goal_xy = tuple(reset_options_list[plot_run_idx]["goal_xy"])  # type: ignore[arg-type]

        meta_run_indices = sorted(path_run_indices)
        if not reset_options_list or reset_options_list[0] is None:
            panel_start_goal: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {
                int(idx): ((int(spec.start_xy[0]), int(spec.start_xy[1])), (int(spec.goal_xy[0]), int(spec.goal_xy[1])))
                for idx in meta_run_indices
            }
        else:
            panel_start_goal = {
                int(idx): (
                    tuple(reset_options_list[int(idx)]["start_xy"]),  # type: ignore[arg-type]
                    tuple(reset_options_list[int(idx)]["goal_xy"]),  # type: ignore[arg-type]
                )
                for idx in meta_run_indices
            }

        for idx, (sp_xy, gp_xy) in panel_start_goal.items():
            meta = dict(base_meta)
            meta["plot_start_x"] = float(sp_xy[0])
            meta["plot_start_y"] = float(sp_xy[1])
            meta["plot_goal_x"] = float(gp_xy[0])
            meta["plot_goal_y"] = float(gp_xy[1])
            meta["plot_run_idx"] = float(idx)
            plot_meta[(env_name, int(idx))] = meta

        if not bool(args.skip_rl):
            # Load trained models
            env_obs_dim = int(env.observation_space.shape[0])
            n_actions = int(env.action_space.n)
            agent_cfg = AgentConfig()

            algo_label = {
                "mlp-dqn": "MLP-DQN",
                "mlp-ddqn": "MLP-DDQN",
                "mlp-pddqn": "MLP-PDDQN",
                "cnn-dqn": "CNN-DQN",
                "cnn-ddqn": "CNN-DDQN",
                "cnn-pddqn": "CNN-PDDQN",
                "mlp-mdqn": "MLP-MDQN",
                "cnn-mdqn": "CNN-MDQN",
                "cnn-sac": "CNN-SAC",
                "cnn-td3": "CNN-TD3",
                "cnn-td3-local": "CNN-TD3-Local",
            }
            algo_seed_offset = {
                "mlp-dqn": 20_000,
                "mlp-ddqn": 30_000,
                "mlp-pddqn": 60_000,
                "cnn-dqn": 40_000,
                "cnn-ddqn": 50_000,
                "cnn-pddqn": 70_000,
                "mlp-mdqn": 90_000,
                "cnn-mdqn": 100_000,
                "cnn-sac": 80_000,
                "cnn-td3": 110_000,
                "cnn-td3-local": 120_000,
            }

            def resolve_model_path(algo: str) -> Path:
                p = models_dir / env_base / f"{algo}.pt"
                if p.exists():
                    return p
                legacy = {
                    "mlp-dqn": "dqn",
                    "mlp-ddqn": "ddqn",
                    # Back-compat: older runs saved Polyak-DDQN as iddqn/cnn-iddqn.
                    "mlp-pddqn": "iddqn",
                    "cnn-pddqn": "cnn-iddqn",
                }.get(str(algo))
                if legacy is not None:
                    p_legacy = models_dir / env_base / f"{legacy}.pt"
                    if p_legacy.exists():
                        return p_legacy
                return p

            algo_paths = {str(a): resolve_model_path(str(a)) for a in args.rl_algos}
            missing = [str(p) for p in algo_paths.values() if not p.exists()]
            if missing:
                exp = ", ".join(str(p) for p in algo_paths.values())
                raise FileNotFoundError(
                    f"Missing model(s) for env {env_base!r}. Expected: {exp}. "
                    "Point --models at a training run (or an experiment name/dir with a latest run)."
                )

            # Check obs_dim consistency (skip SAC checkpoints which have different format)
            dqn_algo_paths = {a: p for a, p in algo_paths.items() if str(a) not in ("cnn-sac", "cnn-td3", "cnn-td3-local")}
            if dqn_algo_paths:
                ckpt_obs_dim = infer_checkpoint_obs_dim(next(iter(dqn_algo_paths.values())))
                for path in dqn_algo_paths.values():
                    if infer_checkpoint_obs_dim(path) != ckpt_obs_dim:
                        raise RuntimeError(f"Observation dim mismatch between checkpoints under: {models_dir / env_base}")

                obs_dim = env_obs_dim
                obs_transform = None
                if ckpt_obs_dim != env_obs_dim:
                    raise RuntimeError(
                        f"Checkpoint expects obs_dim={ckpt_obs_dim} but env provides obs_dim={env_obs_dim} for {env_base!r}. "
                        "Re-train models to match the environment observation space."
                    )
            else:
                obs_dim = env_obs_dim
                obs_transform = None

            agents: dict[str, object] = {}
            sac_agents: dict[str, object] = {}
            for algo, path in algo_paths.items():
                if str(algo) == "cnn-sac":
                    from forest_vehicle_dqn.sac_agent import SACAgent, SACConfig
                    sac_cfg = SACConfig(
                        map_size=int(getattr(args, "global_map_size", 48)),
                        map_channels=int(getattr(args, "global_map_channels", 3)),
                        scalar_dim=12,
                    )
                    sa = SACAgent(sac_cfg, device=str(device), seed=args.seed)
                    sa.load(path)
                    sac_agents[str(algo)] = sa
                    agents[str(algo)] = sa
                elif str(algo) == "cnn-td3":
                    from forest_vehicle_dqn.td3_agent import TD3Agent, TD3Config
                    td3_cfg = TD3Config(
                        map_size=int(getattr(args, "global_map_size", 48)),
                        map_channels=3, scalar_dim=12,
                    )
                    ta = TD3Agent(td3_cfg, device=str(device), seed=args.seed)
                    ta.load(path)
                    sac_agents[str(algo)] = ta
                    agents[str(algo)] = ta
                elif str(algo) == "cnn-td3-local":
                    from forest_vehicle_dqn.td3_agent import TD3Agent, TD3Config
                    obs_map_size_local = int(getattr(args, "obs_map_size", 12))
                    td3l_cfg = TD3Config(
                        map_size=obs_map_size_local, map_channels=1,
                        scalar_dim=10, use_local_encoder=True,
                    )
                    tl = TD3Agent(td3l_cfg, device=str(device), seed=args.seed)
                    tl.load(path)
                    agents[str(algo)] = tl  # stored separately, not in sac_agents
                else:
                    a = DQNFamilyAgent(str(algo), obs_dim, n_actions, config=agent_cfg, seed=args.seed, device=device)
                    a.load(path)
                    agents[str(algo)] = a

            # --- Ensemble model loading (Averaged-DQN / heterogeneous ensemble) ---
            _ENS_ALGOS = ["cnn-dqn", "cnn-ddqn", "cnn-mdqn", "cnn-pddqn"]
            ensemble_agents_map: dict[str, list[DQNFamilyAgent]] = {}
            if args.ensemble_models:
                primary_algo = str(args.rl_algos[0]) if args.rl_algos else "cnn-ddqn"
                for ens_src in args.ensemble_models:
                    ens_dir = resolve_models_dir(ens_src, runs_root=args.runs_root)
                    for ens_algo in _ENS_ALGOS:
                        ens_path = ens_dir / env_base / f"{ens_algo}.pt"
                        if ens_path.exists():
                            ea = DQNFamilyAgent(ens_algo, obs_dim, n_actions, config=agent_cfg, seed=args.seed, device=device)
                            ea.load(ens_path)
                            ensemble_agents_map.setdefault(primary_algo, []).append(ea)
                n_ens = sum(len(v) for v in ensemble_agents_map.values())
                progress_write(f"[infer] Loaded {n_ens} ensemble model(s) from {len(args.ensemble_models)} dir(s)")

            for algo in args.rl_algos:
                algo_key = str(algo)
                pretty = algo_label.get(algo_key, algo_key.upper())
                seed_base = int(algo_seed_offset.get(algo_key, 30_000))

                algo_kpis: list[KPI] = []
                algo_times: list[float] = []
                algo_success = 0
                algo_inadmissible_rates: list[float] = []
                algo_fallback_rates: list[float] = []
                algo_fail_reasons: dict[str, int] = {}
                run_iter = range(int(args.runs))
                if tqdm is not None:
                    run_iter = tqdm(
                        run_iter,
                        desc=f"{env_label} {pretty}",
                        unit="run",
                        dynamic_ncols=True,
                        leave=True,
                    )
                for i in run_iter:
                    if tqdm is None:
                        progress_write(f"[infer] {env_label} {pretty} run {int(i) + 1}/{int(args.runs)}")
                    trace_path = None
                    if bool(getattr(args, "forest_policy_save_traces", False)) and isinstance(env, AMRBicycleEnv):
                        trace_path = out_dir / "traces" / f"{_safe_slug(env_case)}__{_safe_slug(pretty)}__run{int(i)}.csv"
                    if algo_key == "cnn-td3-local" and isinstance(env, AMRBicycleEnv):
                        roll = rollout_td3_local(
                            env,
                            agents[algo_key],
                            max_steps=args.max_steps,
                            seed=int(args.seed) + seed_base + int(i),
                            reset_options=reset_options_list[i] if i < len(reset_options_list) else None,
                            time_mode=str(getattr(args, "kpi_time_mode", "rollout")),
                            collect_controls=bool(int(i) in control_run_indices),
                        )
                    elif algo_key in sac_agents and isinstance(env, AMRBicycleEnv):
                        roll = rollout_sac(
                            env,
                            sac_agents[algo_key],
                            max_steps=args.max_steps,
                            seed=int(args.seed) + seed_base + int(i),
                            reset_options=reset_options_list[i] if i < len(reset_options_list) else None,
                            time_mode=str(getattr(args, "kpi_time_mode", "rollout")),
                            global_map_size=int(getattr(args, "global_map_size", 48)),
                            collect_controls=bool(int(i) in control_run_indices),
                        )
                    else:
                        roll = rollout_agent(
                            env,
                            agents[algo_key],
                            max_steps=args.max_steps,
                            seed=int(args.seed) + seed_base + int(i),
                            reset_options=reset_options_list[i] if i < len(reset_options_list) else None,
                            time_mode=str(getattr(args, "kpi_time_mode", "rollout")),
                            obs_transform=obs_transform,
                            forest_adm_horizon=int(args.forest_adm_horizon),
                            forest_adm_persistence=int(args.forest_adm_persistence),
                            forest_topk=int(args.forest_topk),
                            forest_min_od_m=float(args.forest_min_od_m),
                            forest_min_progress_m=float(args.forest_min_progress_m),
                            forest_no_fallback=bool(getattr(args, "forest_no_fallback", False)),
                            collect_controls=bool(int(i) in control_run_indices),
                            trace_path=trace_path,
                            ensemble_agents=ensemble_agents_map.get(algo_key),
                            ensemble_mode=str(getattr(args, "ensemble_mode", "qavg")),
                            negotiate_k=int(getattr(args, "negotiate_k", 20)),
                            negotiate_k_max=int(getattr(args, "negotiate_k_max", 30)),
                            negsoft_alpha=float(getattr(args, "negsoft_alpha", 0.7)),
                            negsoft_tau=float(getattr(args, "negsoft_tau", 0.1)),
                            ensemble_adm_prefilter=bool(getattr(args, "ensemble_adm_prefilter", False)),
                        )
                    algo_times.append(float(roll.compute_time_s))
                    if int(i) in path_run_indices:
                        env_paths_by_run[int(i)][pretty] = PathTrace(path_xy_cells=roll.path_xy_cells, success=bool(roll.reached))
                    if roll.controls is not None and int(i) in control_run_indices:
                        controls_for_plot.setdefault((env_name, int(i)), {})[str(pretty)] = roll.controls

                    start_xy = (int(spec.start_xy[0]), int(spec.start_xy[1]))
                    goal_xy = (int(spec.goal_xy[0]), int(spec.goal_xy[1]))
                    opts = reset_options_list[i] if i < len(reset_options_list) else None
                    if isinstance(opts, dict) and "start_xy" in opts and "goal_xy" in opts:
                        sx, sy = opts["start_xy"]  # type: ignore[misc]
                        gx, gy = opts["goal_xy"]  # type: ignore[misc]
                        start_xy = (int(sx), int(sy))
                        goal_xy = (int(gx), int(gy))

                    raw_corners = float(num_path_corners(roll.path_xy_cells, angle_threshold_deg=13.0))
                    smoothed = smooth_path(roll.path_xy_cells, iterations=2)
                    smoothed_m = [(float(x) * float(cell_size_m), float(y) * float(cell_size_m)) for x, y in smoothed]
                    run_kpi = KPI(
                        avg_path_length=float(path_length(smoothed)) * float(cell_size_m),
                        path_time_s=float(roll.path_time_s),
                        avg_curvature_1_m=float(avg_abs_curvature(smoothed_m)),
                        planning_time_s=float(roll.compute_time_s),
                        tracking_time_s=0.0,
                        inference_time_s=float(roll.compute_time_s),
                        num_corners=raw_corners,
                        max_corner_deg=float(max_corner_degree(smoothed)),
                    )
                    roll_debug = dict(roll.debug or {})
                    run_failure_reason = str(roll_debug.get("failure_reason", "unknown"))
                    run_inad_rate = float(roll_debug.get("argmax_inadmissible_rate", float("nan")))
                    run_fallback_rate = float(roll_debug.get("fallback_rate", float("nan")))
                    if math.isfinite(run_inad_rate):
                        algo_inadmissible_rates.append(float(run_inad_rate))
                    if math.isfinite(run_fallback_rate):
                        algo_fallback_rates.append(float(run_fallback_rate))
                    algo_fail_reasons[run_failure_reason] = int(algo_fail_reasons.get(run_failure_reason, 0)) + 1
                    rows_runs.append(
                        {
                            "Environment": str(env_label),
                            "Algorithm": str(pretty),
                            "run_idx": int(i),
                            "start_x": int(start_xy[0]),
                            "start_y": int(start_xy[1]),
                            "goal_x": int(goal_xy[0]),
                            "goal_y": int(goal_xy[1]),
                            "success_rate": 1.0 if bool(roll.reached) else 0.0,
                            "failure_reason": str(run_failure_reason),
                            "argmax_inadmissible_rate": float(run_inad_rate),
                            "fallback_rate": float(run_fallback_rate),
                            **dict(run_kpi.__dict__),
                        }
                    )
                    if bool(roll.reached):
                        algo_success += 1
                        algo_kpis.append(run_kpi)

                k = mean_kpi(algo_kpis)
                k_dict = dict(k.__dict__)
                if algo_times:
                    mean_plan = float(np.mean(algo_times))
                    k_dict["planning_time_s"] = mean_plan
                    k_dict["tracking_time_s"] = 0.0
                    k_dict["inference_time_s"] = mean_plan
                rows.append(
                    {
                        "Environment": str(env_label),
                        "Algorithm": str(pretty),
                        "success_rate": float(algo_success) / float(max(1, int(args.runs))),
                        "argmax_inadmissible_rate": (
                            float(np.mean(algo_inadmissible_rates)) if algo_inadmissible_rates else float("nan")
                        ),
                        "fallback_rate": float(np.mean(algo_fallback_rates)) if algo_fallback_rates else float("nan"),
                        **k_dict,
                    }
                )
                if algo_fail_reasons:
                    items = ", ".join(f"{k}={v}" for k, v in sorted(algo_fail_reasons.items()))
                    inad_mean = float(np.mean(algo_inadmissible_rates)) if algo_inadmissible_rates else float("nan")
                    fallback_mean = float(np.mean(algo_fallback_rates)) if algo_fallback_rates else float("nan")
                    progress_write(
                        f"[infer] {env_label} {pretty} debug: argmax_inadmissible_rate={inad_mean:.3f}, "
                        f"fallback_rate={fallback_mean:.3f}, failures: {items}"
                    )

        if baselines:
            grid_map = grid_map_from_obstacles(grid_y0_bottom=grid, cell_size_m=float(cell_size_m))
            params = default_ackermann_params()
            if env_base in FOREST_ENV_ORDER and isinstance(env, AMRBicycleEnv):
                footprint = forest_two_circle_footprint()
                goal_xy_tol_m = float(env.goal_tolerance_m)
                goal_theta_tol_rad = float(env.goal_angle_tolerance_rad)
                start_theta_rad = None
            else:
                footprint = point_footprint(cell_size_m=float(cell_size_m))
                goal_xy_tol_m = float(cell_size_m) * 0.5
                goal_theta_tol_rad = float(math.pi)
                start_theta_rad = 0.0

            use_random_pairs = bool(getattr(args, "random_start_goal", False)) and bool(reset_options_list) and reset_options_list[0] is not None

            def pair_for_run(i: int) -> tuple[tuple[int, int], tuple[int, int], dict[str, object] | None]:
                if use_random_pairs and i < len(reset_options_list) and reset_options_list[i] is not None:
                    opts = reset_options_list[i] or {}
                    sx, sy = opts["start_xy"]  # type: ignore[misc]
                    gx, gy = opts["goal_xy"]  # type: ignore[misc]
                    return (int(sx), int(sy)), (int(gx), int(gy)), opts
                return (int(spec.start_xy[0]), int(spec.start_xy[1])), (int(spec.goal_xy[0]), int(spec.goal_xy[1])), None

            hybrid_collision_padding_m = 0.0
            rrt_collision_padding_m = 0.0
            if env_base in FOREST_ENV_ORDER and isinstance(env, AMRBicycleEnv):
                hybrid_collision_padding_m = float(getattr(args, "forest_baseline_hybrid_collision_padding_m", 0.02))
                rrt_collision_padding_m = float(getattr(args, "forest_baseline_rrt_collision_padding_m", 0.0))

            mpc_cfg = MPCConfig(
                horizon_steps=max(1, int(getattr(args, "forest_mpc_horizon_steps", 12))),
                candidates=max(9, int(getattr(args, "forest_mpc_candidates", 256))),
                dt_s=float(getattr(args, "forest_mpc_dt_s", 0.0)),
                w_u=float(getattr(args, "forest_mpc_w_u", 0.05)),
                w_du=float(getattr(args, "forest_mpc_w_du", 0.5)),
                w_pos=float(getattr(args, "forest_mpc_w_pos", 6.0)),
                w_yaw=float(getattr(args, "forest_mpc_w_yaw", 1.2)),
                align_dist_m=float(getattr(args, "forest_mpc_align_dist_m", 1.5)),
                collision_padding_m=float(getattr(args, "forest_mpc_collision_padding_m", 0.0)),
                goal_lookahead_steps=max(1, int(getattr(args, "forest_mpc_goal_lookahead_steps", 2))),
            )
            astar_curve_cfg = AStarCurveOptConfig(
                enable=bool(getattr(args, "forest_astar_opt_enable", True)),
                max_replans=max(1, int(getattr(args, "forest_astar_opt_max_replans", 200))),
                resample_ds_m=max(1e-4, float(getattr(args, "forest_astar_opt_resample_ds_m", 0.2))),
                collision_step_m=max(1e-4, float(getattr(args, "forest_astar_opt_collision_step_m", 0.1))),
                shortcut_passes=max(0, int(getattr(args, "forest_astar_opt_shortcut_passes", 2))),
            )

            def run_mpc_combo_baseline(*, algo_name: str, planner_kind: str) -> None:
                combo_kpis: list[KPI] = []
                combo_plan_times: list[float] = []
                combo_total_times: list[float] = []
                combo_fail_reasons: dict[str, int] = {}
                combo_success = 0

                n_runs = int(args.runs) if use_random_pairs else 1
                progress_write(f"[infer] {env_label} baseline {algo_name} ({n_runs} run(s))")
                base_iter = range(n_runs)
                if tqdm is not None:
                    base_iter = tqdm(
                        base_iter,
                        desc=f"{env_label} {algo_name}",
                        unit="run",
                        dynamic_ncols=True,
                        leave=True,
                    )

                for i in base_iter:
                    if tqdm is None:
                        progress_write(f"[infer] {env_label} {algo_name} run {int(i) + 1}/{int(n_runs)}")
                    start_xy, goal_xy, r_opts = pair_for_run(int(i))

                    if planner_kind == "astar":
                        if precomputed_grid_paths is not None and use_random_pairs and int(i) < len(precomputed_grid_paths):
                            res = precomputed_grid_paths[int(i)]
                        else:
                            res = plan_grid_astar(
                                grid_map=grid_map,
                                start_xy=start_xy,
                                goal_xy=goal_xy,
                                timeout_s=float(args.baseline_timeout),
                                seed=int(args.seed),
                                replan_attempt=int(i),
                                curve_opt_cfg=astar_curve_cfg,
                                footprint=footprint,
                                ackermann_params=params,
                            )
                    elif planner_kind == "hybrid_astar":
                        if precomputed_hybrid_paths is not None and use_random_pairs and int(i) < len(precomputed_hybrid_paths):
                            res = precomputed_hybrid_paths[int(i)]
                        else:
                            res = plan_hybrid_astar(
                                grid_map=grid_map,
                                footprint=footprint,
                                params=params,
                                start_xy=start_xy,
                                goal_xy=goal_xy,
                                goal_theta_rad=None,
                                start_theta_rad=start_theta_rad,
                                goal_xy_tol_m=goal_xy_tol_m,
                                goal_theta_tol_rad=goal_theta_tol_rad,
                                timeout_s=float(args.baseline_timeout),
                                max_nodes=int(args.hybrid_max_nodes),
                                collision_padding_m=float(hybrid_collision_padding_m),
                            )
                    elif planner_kind == "rrt_star":
                        res = plan_rrt_star(
                            grid_map=grid_map,
                            footprint=footprint,
                            params=params,
                            start_xy=start_xy,
                            goal_xy=goal_xy,
                            goal_theta_rad=None,
                            start_theta_rad=start_theta_rad,
                            goal_xy_tol_m=goal_xy_tol_m,
                            goal_theta_tol_rad=goal_theta_tol_rad,
                            timeout_s=float(args.baseline_timeout),
                            max_iter=int(args.rrt_max_iter),
                            seed=args.seed + 40_000 + int(i),
                            collision_padding_m=float(rrt_collision_padding_m),
                        )
                    else:
                        raise RuntimeError(f"unknown planner_kind: {planner_kind}")

                    planner_path = list(res.path_xy_cells)
                    planner_theta = list(res.path_theta_rad) if res.path_theta_rad is not None else None
                    plan_success = bool(res.success)

                    exec_path = list(planner_path)
                    failure_reason = "planner_fail"
                    reached_flag = False
                    tracking_time_s = 0.0
                    infer_time_s = float(res.time_s)
                    path_time_s = float("nan")
                    mpc_out = None

                    if plan_success:
                        if isinstance(env, AMRBicycleEnv):
                            if isinstance(r_opts, dict):
                                reset_opts = dict(r_opts)
                            else:
                                reset_opts = {"start_xy": start_xy, "goal_xy": goal_xy}
                            mpc_out = run_mpc_local_planning(
                                env=env,
                                plan_path_xy_cells=planner_path,
                                plan_path_theta_rad=planner_theta,
                                max_steps=int(args.max_steps),
                                seed=int(args.seed) + 45_000 + int(i),
                                config=mpc_cfg,
                                reset_options=reset_opts,
                            )
                            exec_path = list(mpc_out.path_xy_cells)
                            reached_flag = bool(mpc_out.reached)
                            failure_reason = "reached" if reached_flag else str(mpc_out.failure_reason)
                            tracking_time_s = float(mpc_out.compute_time_s)
                            infer_time_s = float(res.time_s) + float(tracking_time_s)
                            path_time_s = float(mpc_out.path_time_s)
                        else:
                            failure_reason = "unsupported_env"

                    if not math.isfinite(float(path_time_s)) and isinstance(env, AMRBicycleEnv):
                        sm_for_time = smooth_path(exec_path, iterations=2)
                        sm_for_time_m = [(float(x) * float(cell_size_m), float(y) * float(cell_size_m)) for x, y in sm_for_time]
                        path_time_s = float(path_length(sm_for_time_m)) / max(1e-9, float(env.model.v_max_m_s))

                    exec_path_sm = smooth_path(exec_path, iterations=2)
                    exec_path_sm_m = [(float(x) * float(cell_size_m), float(y) * float(cell_size_m)) for x, y in exec_path_sm]
                    combo_kpi = KPI(
                        avg_path_length=float(path_length(exec_path_sm)) * float(cell_size_m),
                        path_time_s=float(path_time_s),
                        avg_curvature_1_m=float(avg_abs_curvature(exec_path_sm_m)),
                        planning_time_s=float(res.time_s),
                        tracking_time_s=float(tracking_time_s),
                        inference_time_s=float(infer_time_s),
                        num_corners=float(num_path_corners(exec_path, angle_threshold_deg=13.0)),
                        max_corner_deg=float(max_corner_degree(exec_path)),
                    )

                    rows_runs.append(
                        {
                            "Environment": str(env_label),
                            "Algorithm": str(algo_name),
                            "run_idx": int(i),
                            "start_x": int(start_xy[0]),
                            "start_y": int(start_xy[1]),
                            "goal_x": int(goal_xy[0]),
                            "goal_y": int(goal_xy[1]),
                            "success_rate": 1.0 if bool(reached_flag) else 0.0,
                            "failure_reason": str(failure_reason),
                            **dict(combo_kpi.__dict__),
                        }
                    )

                    combo_plan_times.append(float(res.time_s))
                    combo_total_times.append(float(infer_time_s))
                    if bool(reached_flag):
                        combo_success += 1
                        combo_kpis.append(combo_kpi)
                    if not bool(reached_flag):
                        combo_fail_reasons[str(failure_reason)] = int(combo_fail_reasons.get(str(failure_reason), 0)) + 1

                    if int(i) in path_run_indices:
                        env_paths_by_run[int(i)][str(algo_name)] = PathTrace(path_xy_cells=exec_path, success=bool(reached_flag))
                    if int(i) in control_run_indices and mpc_out is not None:
                        controls_for_plot.setdefault((env_name, int(i)), {})[str(algo_name)] = ControlTrace(
                            t_s=np.asarray(mpc_out.t_s, dtype=np.float64),
                            v_m_s=np.asarray(mpc_out.v_m_s, dtype=np.float64),
                            delta_rad=np.asarray(mpc_out.delta_rad, dtype=np.float64),
                        )

                k = mean_kpi(combo_kpis)
                k_dict = dict(k.__dict__)
                if combo_plan_times:
                    k_dict["planning_time_s"] = float(np.mean(combo_plan_times))
                if combo_total_times:
                    k_dict["inference_time_s"] = float(np.mean(combo_total_times))
                rows.append(
                    {
                        "Environment": str(env_label),
                        "Algorithm": str(algo_name),
                        "success_rate": float(combo_success) / float(max(1, int(n_runs))),
                        **k_dict,
                    }
                )

                if combo_fail_reasons:
                    items = ", ".join(f"{k}={v}" for k, v in sorted(combo_fail_reasons.items()))
                    progress_write(f"[infer] {env_label} {algo_name} failures: {items}")


            if "astar" in baselines:
                astar_plan_kpis: list[KPI] = []
                astar_plan_plan_times: list[float] = []
                astar_plan_total_times: list[float] = []
                astar_plan_fail_reasons: dict[str, int] = {}
                astar_plan_success = 0

                n_runs = int(args.runs) if use_random_pairs else 1
                progress_write(f"[infer] {env_label} baseline A* ({n_runs} run(s))")
                base_iter = range(n_runs)
                if tqdm is not None:
                    base_iter = tqdm(
                        base_iter,
                        desc=f"{env_label} A*",
                        unit="run",
                        dynamic_ncols=True,
                        leave=True,
                    )
                for i in base_iter:
                    if tqdm is None:
                        progress_write(f"[infer] {env_label} A* run {int(i) + 1}/{int(n_runs)}")
                    start_xy, goal_xy, r_opts = pair_for_run(int(i))
                    if precomputed_grid_paths is not None and use_random_pairs and int(i) < len(precomputed_grid_paths):
                        res = precomputed_grid_paths[int(i)]
                    else:
                        res = plan_grid_astar(
                            grid_map=grid_map,
                            start_xy=start_xy,
                            goal_xy=goal_xy,
                            timeout_s=float(args.baseline_timeout),
                            seed=int(args.seed),
                            replan_attempt=int(i),
                        )
                    astar_plan_path = list(res.path_xy_cells)
                    astar_plan_success_flag = bool(res.success)
                    if not bool(astar_plan_success_flag):
                        reason = res.stats.get("failure_reason", "unknown")
                        reason = "unknown" if reason is None else str(reason)
                        astar_plan_fail_reasons[reason] = int(astar_plan_fail_reasons.get(reason, 0)) + 1

                    astar_plan_failure_reason = "reached" if bool(astar_plan_success_flag) else "planner_fail"
                    astar_plan_path_time_s = float("nan")
                    astar_plan_smoothed = smooth_path(astar_plan_path, iterations=2)
                    astar_plan_smoothed_m = [
                        (float(x) * float(cell_size_m), float(y) * float(cell_size_m)) for x, y in astar_plan_smoothed
                    ]
                    if not math.isfinite(float(astar_plan_path_time_s)) and isinstance(env, AMRBicycleEnv):
                        astar_plan_path_time_s = float(path_length(astar_plan_smoothed_m)) / max(1e-9, float(env.model.v_max_m_s))
                    astar_plan_kpi = KPI(
                        avg_path_length=float(path_length(astar_plan_smoothed)) * float(cell_size_m),
                        path_time_s=float(astar_plan_path_time_s),
                        avg_curvature_1_m=float(avg_abs_curvature(astar_plan_smoothed_m)),
                        planning_time_s=float(res.time_s),
                        tracking_time_s=0.0,
                        inference_time_s=float(res.time_s),
                        num_corners=float(num_path_corners(astar_plan_path, angle_threshold_deg=13.0)),
                        max_corner_deg=float(max_corner_degree(astar_plan_path)),
                    )
                    rows_runs.append(
                        {
                            "Environment": str(env_label),
                            "Algorithm": "A*",
                            "run_idx": int(i),
                            "start_x": int(start_xy[0]),
                            "start_y": int(start_xy[1]),
                            "goal_x": int(goal_xy[0]),
                            "goal_y": int(goal_xy[1]),
                            "success_rate": 1.0 if bool(astar_plan_success_flag) else 0.0,
                            "failure_reason": str(astar_plan_failure_reason),
                            **dict(astar_plan_kpi.__dict__),
                        }
                    )
                    astar_plan_plan_times.append(float(res.time_s))
                    astar_plan_total_times.append(float(res.time_s))
                    if bool(astar_plan_success_flag) and astar_plan_path:
                        astar_plan_success += 1
                        astar_plan_kpis.append(astar_plan_kpi)

                    if int(i) in path_run_indices:
                        env_paths_by_run[int(i)]["A*"] = PathTrace(
                            path_xy_cells=astar_plan_path, success=bool(astar_plan_success_flag)
                        )

                k = mean_kpi(astar_plan_kpis)
                k_dict = dict(k.__dict__)
                if astar_plan_plan_times:
                    k_dict["planning_time_s"] = float(np.mean(astar_plan_plan_times))
                k_dict["tracking_time_s"] = 0.0
                if astar_plan_total_times:
                    k_dict["inference_time_s"] = float(np.mean(astar_plan_total_times))
                rows.append(
                    {
                        "Environment": str(env_label),
                        "Algorithm": "A*",
                        "success_rate": float(astar_plan_success) / float(max(1, int(n_runs))),
                        **k_dict,
                    }
                )
                if astar_plan_fail_reasons:
                    items = ", ".join(f"{k}={v}" for k, v in sorted(astar_plan_fail_reasons.items()))
                    progress_write(
                        f"[infer] {env_label} A* failures: {items} "
                        f"(timeout={float(args.baseline_timeout):.2f}s)."
                    )

            if "hybrid_astar" in baselines:
                ha_plan_kpis: list[KPI] = []
                ha_plan_plan_times: list[float] = []
                ha_plan_total_times: list[float] = []
                ha_plan_fail_reasons: dict[str, int] = {}
                ha_plan_success = 0

                n_runs = int(args.runs) if use_random_pairs else 1
                progress_write(f"[infer] {env_label} baseline Hybrid A* ({n_runs} run(s))")
                base_iter = range(n_runs)
                if tqdm is not None:
                    base_iter = tqdm(
                        base_iter,
                        desc=f"{env_label} Hybrid A*",
                        unit="run",
                        dynamic_ncols=True,
                        leave=True,
                    )
                for i in base_iter:
                    if tqdm is None:
                        progress_write(f"[infer] {env_label} Hybrid A* run {int(i) + 1}/{int(n_runs)}")
                    start_xy, goal_xy, r_opts = pair_for_run(int(i))
                    if precomputed_hybrid_paths is not None and use_random_pairs and int(i) < len(precomputed_hybrid_paths):
                        res = precomputed_hybrid_paths[int(i)]
                    else:
                        res = plan_hybrid_astar(
                            grid_map=grid_map,
                            footprint=footprint,
                            params=params,
                            start_xy=start_xy,
                            goal_xy=goal_xy,
                            goal_theta_rad=None,
                            start_theta_rad=start_theta_rad,
                            goal_xy_tol_m=goal_xy_tol_m,
                            goal_theta_tol_rad=goal_theta_tol_rad,
                            timeout_s=float(args.baseline_timeout),
                            max_nodes=int(args.hybrid_max_nodes),
                            collision_padding_m=float(hybrid_collision_padding_m),
                        )
                    ha_plan_path = list(res.path_xy_cells)
                    ha_plan_success_flag = bool(res.success)
                    if not bool(ha_plan_success_flag):
                        reason = res.stats.get("failure_reason", "unknown")
                        reason = "unknown" if reason is None else str(reason)
                        ha_plan_fail_reasons[reason] = int(ha_plan_fail_reasons.get(reason, 0)) + 1

                    ha_plan_failure_reason = "reached" if bool(ha_plan_success_flag) else "planner_fail"
                    ha_plan_path_time_s = float("nan")
                    ha_plan_smoothed = smooth_path(ha_plan_path, iterations=2)
                    ha_plan_smoothed_m = [
                        (float(x) * float(cell_size_m), float(y) * float(cell_size_m)) for x, y in ha_plan_smoothed
                    ]
                    if not math.isfinite(float(ha_plan_path_time_s)) and isinstance(env, AMRBicycleEnv):
                        ha_plan_path_time_s = float(path_length(ha_plan_smoothed_m)) / max(1e-9, float(env.model.v_max_m_s))
                    ha_plan_kpi = KPI(
                        avg_path_length=float(path_length(ha_plan_smoothed)) * float(cell_size_m),
                        path_time_s=float(ha_plan_path_time_s),
                        avg_curvature_1_m=float(avg_abs_curvature(ha_plan_smoothed_m)),
                        planning_time_s=float(res.time_s),
                        tracking_time_s=0.0,
                        inference_time_s=float(res.time_s),
                        num_corners=float(num_path_corners(ha_plan_path, angle_threshold_deg=13.0)),
                        max_corner_deg=float(max_corner_degree(ha_plan_path)),
                    )
                    rows_runs.append(
                        {
                            "Environment": str(env_label),
                            "Algorithm": "Hybrid A*",
                            "run_idx": int(i),
                            "start_x": int(start_xy[0]),
                            "start_y": int(start_xy[1]),
                            "goal_x": int(goal_xy[0]),
                            "goal_y": int(goal_xy[1]),
                            "success_rate": 1.0 if bool(ha_plan_success_flag) else 0.0,
                            "failure_reason": str(ha_plan_failure_reason),
                            **dict(ha_plan_kpi.__dict__),
                        }
                    )
                    ha_plan_plan_times.append(float(res.time_s))
                    ha_plan_total_times.append(float(res.time_s))
                    if bool(ha_plan_success_flag) and ha_plan_path:
                        ha_plan_success += 1
                        ha_plan_kpis.append(ha_plan_kpi)

                    if int(i) in path_run_indices:
                        env_paths_by_run[int(i)]["Hybrid A*"] = PathTrace(
                            path_xy_cells=ha_plan_path, success=bool(ha_plan_success_flag)
                        )

                k = mean_kpi(ha_plan_kpis)
                k_dict = dict(k.__dict__)
                if ha_plan_plan_times:
                    k_dict["planning_time_s"] = float(np.mean(ha_plan_plan_times))
                k_dict["tracking_time_s"] = 0.0
                if ha_plan_total_times:
                    k_dict["inference_time_s"] = float(np.mean(ha_plan_total_times))
                rows.append(
                    {
                        "Environment": str(env_label),
                        "Algorithm": "Hybrid A*",
                        "success_rate": float(ha_plan_success) / float(max(1, int(n_runs))),
                        **k_dict,
                    }
                )
                if ha_plan_fail_reasons:
                    items = ", ".join(f"{k}={v}" for k, v in sorted(ha_plan_fail_reasons.items()))
                    progress_write(
                        f"[infer] {env_label} Hybrid A* failures: {items} "
                        f"(timeout={float(args.baseline_timeout):.2f}s, max_nodes={int(args.hybrid_max_nodes)})."
                    )

            if "rrt_star" in baselines:
                rrt_plan_kpis: list[KPI] = []
                rrt_plan_plan_times: list[float] = []
                rrt_plan_total_times: list[float] = []
                rrt_plan_fail_reasons: dict[str, int] = {}
                rrt_plan_success = 0

                n_runs = int(args.runs)
                progress_write(f"[infer] {env_label} baseline RRT* ({n_runs} run(s))")
                base_iter = range(n_runs)
                if tqdm is not None:
                    base_iter = tqdm(
                        base_iter,
                        desc=f"{env_label} RRT*",
                        unit="run",
                        dynamic_ncols=True,
                        leave=True,
                    )
                for i in base_iter:
                    if tqdm is None:
                        progress_write(f"[infer] {env_label} RRT* run {int(i) + 1}/{int(n_runs)}")
                    start_xy, goal_xy, r_opts = pair_for_run(int(i))
                    res = plan_rrt_star(
                        grid_map=grid_map,
                        footprint=footprint,
                        params=params,
                        start_xy=start_xy,
                        goal_xy=goal_xy,
                        goal_theta_rad=None,
                        start_theta_rad=start_theta_rad,
                        goal_xy_tol_m=goal_xy_tol_m,
                        goal_theta_tol_rad=goal_theta_tol_rad,
                        timeout_s=float(args.baseline_timeout),
                        max_iter=int(args.rrt_max_iter),
                        seed=args.seed + 30_000 + i,
                        collision_padding_m=float(rrt_collision_padding_m),
                    )
                    plan_path = list(res.path_xy_cells)
                    plan_success = bool(res.success)
                    if not bool(plan_success):
                        reason = res.stats.get("failure_reason", "unknown")
                        reason = "unknown" if reason is None else str(reason)
                        rrt_plan_fail_reasons[reason] = int(rrt_plan_fail_reasons.get(reason, 0)) + 1

                    rrt_plan_failure_reason = "reached" if bool(plan_success) else "planner_fail"
                    rrt_plan_path_time_s = float("nan")
                    rrt_plan_smoothed = smooth_path(plan_path, iterations=2)
                    rrt_plan_smoothed_m = [
                        (float(x) * float(cell_size_m), float(y) * float(cell_size_m)) for x, y in rrt_plan_smoothed
                    ]
                    if not math.isfinite(float(rrt_plan_path_time_s)) and isinstance(env, AMRBicycleEnv):
                        rrt_plan_path_time_s = float(path_length(rrt_plan_smoothed_m)) / max(1e-9, float(env.model.v_max_m_s))
                    rrt_plan_kpi = KPI(
                        avg_path_length=float(path_length(rrt_plan_smoothed)) * float(cell_size_m),
                        path_time_s=float(rrt_plan_path_time_s),
                        avg_curvature_1_m=float(avg_abs_curvature(rrt_plan_smoothed_m)),
                        planning_time_s=float(res.time_s),
                        tracking_time_s=0.0,
                        inference_time_s=float(res.time_s),
                        num_corners=float(num_path_corners(plan_path, angle_threshold_deg=13.0)),
                        max_corner_deg=float(max_corner_degree(plan_path)),
                    )
                    rows_runs.append(
                        {
                            "Environment": str(env_label),
                            "Algorithm": "RRT*",
                            "run_idx": int(i),
                            "start_x": int(start_xy[0]),
                            "start_y": int(start_xy[1]),
                            "goal_x": int(goal_xy[0]),
                            "goal_y": int(goal_xy[1]),
                            "success_rate": 1.0 if bool(plan_success) else 0.0,
                            "failure_reason": str(rrt_plan_failure_reason),
                            **dict(rrt_plan_kpi.__dict__),
                        }
                    )
                    rrt_plan_plan_times.append(float(res.time_s))
                    rrt_plan_total_times.append(float(res.time_s))
                    if bool(plan_success) and plan_path:
                        rrt_plan_success += 1
                        rrt_plan_kpis.append(rrt_plan_kpi)

                    if int(i) in path_run_indices:
                        env_paths_by_run[int(i)]["RRT"] = PathTrace(path_xy_cells=plan_path, success=bool(plan_success))

                k = mean_kpi(rrt_plan_kpis)
                k_dict = dict(k.__dict__)
                if rrt_plan_plan_times:
                    k_dict["planning_time_s"] = float(np.mean(rrt_plan_plan_times))
                k_dict["tracking_time_s"] = 0.0
                if rrt_plan_total_times:
                    k_dict["inference_time_s"] = float(np.mean(rrt_plan_total_times))
                rows.append(
                    {
                        "Environment": str(env_label),
                        "Algorithm": "RRT*",
                        "success_rate": float(rrt_plan_success) / float(max(1, int(args.runs))),
                        **k_dict,
                    }
                )
                if rrt_plan_fail_reasons:
                    items = ", ".join(f"{k}={v}" for k, v in sorted(rrt_plan_fail_reasons.items()))
                    progress_write(
                        f"[infer] {env_label} RRT* failures: {items} "
                        f"(timeout={float(args.baseline_timeout):.2f}s, max_iter={int(args.rrt_max_iter)})."
                    )

            if "astar_mpc" in baselines:
                run_mpc_combo_baseline(algo_name="A*-MPC", planner_kind="astar")

            if "hybrid_astar_mpc" in baselines:
                run_mpc_combo_baseline(
                    algo_name="Hybrid A*-MPC",
                    planner_kind="hybrid_astar",
                )

            if "rrt_mpc" in baselines:
                run_mpc_combo_baseline(algo_name="RRT-MPC", planner_kind="rrt_star")

        for run_idx, run_paths in env_paths_by_run.items():
            paths_for_plot[(env_name, int(run_idx))] = dict(run_paths)

    table = pd.DataFrame(rows_runs)
    failure_reason_series = table.get("failure_reason")
    argmax_inadmissible_rate_series = table.get("argmax_inadmissible_rate")
    fallback_rate_series = table.get("fallback_rate")
    # Pretty column order
    table_cols = [
        "Environment",
        "Algorithm",
        "run_idx",
        "start_x",
        "start_y",
        "goal_x",
        "goal_y",
        "success_rate",
        "avg_path_length",
        "path_time_s",
        "avg_curvature_1_m",
        "planning_time_s",
        "tracking_time_s",
        "num_corners",
        "inference_time_s",
        "max_corner_deg",
    ]
    if argmax_inadmissible_rate_series is not None:
        table_cols.append("argmax_inadmissible_rate")
    if fallback_rate_series is not None:
        table_cols.append("fallback_rate")
    table = table[table_cols]
    table = table.copy()

    # Composite metric (lower is better): combines path length and compute time,
    # then penalizes non-reaching behavior via success_rate.
    w_t = float(args.score_time_weight)
    sr_raw = pd.to_numeric(table["success_rate"], errors="coerce").astype(float)
    denom = sr_raw.clip(lower=1e-6)
    base = pd.to_numeric(table["avg_path_length"], errors="coerce").astype(float) + w_t * pd.to_numeric(
        table["inference_time_s"], errors="coerce"
    ).astype(float)
    planning_cost = (base / denom).astype(float)
    planning_cost = planning_cost.where((sr_raw > 0.0) & np.isfinite(base.to_numpy()), other=float("inf"))
    table["planning_cost"] = planning_cost

    # Composite score (lower is better): combines path time, curvature, and planning compute time,
    # then penalizes non-reaching behavior via success_rate.
    w_pt = float(getattr(args, "composite_w_path_time", 1.0))
    w_k = float(getattr(args, "composite_w_avg_curvature", 1.0))
    w_pl = float(getattr(args, "composite_w_planning_time", 1.0))
    w_sum = max(1e-12, float(w_pt + w_k + w_pl))

    def _minmax_norm(s: pd.Series) -> pd.Series:
        x = pd.to_numeric(s, errors="coerce").astype(float)
        v = x.to_numpy(dtype=float, copy=False)
        finite = np.isfinite(v)
        if not bool(finite.any()):
            return pd.Series(np.zeros_like(v, dtype=float), index=x.index)
        mn = float(np.min(v[finite]))
        mx = float(np.max(v[finite]))
        d = float(mx - mn)
        if not math.isfinite(d) or d < 1e-12:
            return pd.Series(np.zeros_like(v, dtype=float), index=x.index)
        out = (v - mn) / d
        out = np.where(finite, out, np.nan)
        return pd.Series(out.astype(float, copy=False), index=x.index)

    group_keys = ["Environment", "run_idx"]
    n_pt = table.groupby(group_keys, sort=False)["path_time_s"].transform(_minmax_norm).fillna(0.0)
    n_k = table.groupby(group_keys, sort=False)["avg_curvature_1_m"].transform(_minmax_norm).fillna(0.0)
    n_pl = table.groupby(group_keys, sort=False)["planning_time_s"].transform(_minmax_norm).fillna(0.0)
    base_score = (w_pt * n_pt + w_k * n_k + w_pl * n_pl) / w_sum
    sr_denom2 = sr_raw.clip(lower=1e-6)
    composite_score = (base_score / sr_denom2).astype(float)
    composite_score = composite_score.where(sr_raw > 0.0, other=float("inf"))
    table["composite_score"] = composite_score

    table["success_rate"] = pd.to_numeric(table["success_rate"], errors="coerce").astype(float).round(3)
    table["avg_path_length"] = table["avg_path_length"].astype(float).round(4)
    table["path_time_s"] = pd.to_numeric(table["path_time_s"], errors="coerce").astype(float).round(4)
    table["avg_curvature_1_m"] = pd.to_numeric(table["avg_curvature_1_m"], errors="coerce").astype(float).round(6)
    table["planning_time_s"] = pd.to_numeric(table["planning_time_s"], errors="coerce").astype(float).round(5)
    table["tracking_time_s"] = pd.to_numeric(table["tracking_time_s"], errors="coerce").astype(float).round(5)
    table["num_corners"] = pd.to_numeric(table["num_corners"], errors="coerce").round(0).astype("Int64")
    table["inference_time_s"] = table["inference_time_s"].astype(float).round(5)
    table["max_corner_deg"] = pd.to_numeric(table["max_corner_deg"], errors="coerce").round(0).astype("Int64")
    table["planning_cost"] = pd.to_numeric(table["planning_cost"], errors="coerce").astype(float).round(3)
    table["composite_score"] = pd.to_numeric(table["composite_score"], errors="coerce").astype(float).round(3)
    if argmax_inadmissible_rate_series is not None:
        table["argmax_inadmissible_rate"] = (
            pd.to_numeric(argmax_inadmissible_rate_series, errors="coerce").astype(float).round(3)
        )
    if fallback_rate_series is not None:
        table["fallback_rate"] = pd.to_numeric(fallback_rate_series, errors="coerce").astype(float).round(3)
    if failure_reason_series is not None:
        # Keep legacy column order unchanged; append as the last column for debugging.
        table["failure_reason"] = failure_reason_series
    table.to_csv(out_dir / "table2_kpis_raw.csv", index=False)

    table_pretty = table.rename(
        columns={
            "Algorithm": "Algorithm name",
            "run_idx": "Run index",
            "start_x": "Start x",
            "start_y": "Start y",
            "goal_x": "Goal x",
            "goal_y": "Goal y",
            "success_rate": "Success rate",
            "avg_path_length": "Average path length (m)",
            "path_time_s": "Path time (s)",
            "avg_curvature_1_m": "Average curvature (1/m)",
            "planning_time_s": "Planning time (s)",
            "tracking_time_s": "Tracking time (s)",
            "num_corners": "Number of path corners",
            "inference_time_s": "Compute time (s)",
            "max_corner_deg": "Max corner degree (°)",
            "planning_cost": "Planning cost (m)",
            "composite_score": "Composite score",
            "argmax_inadmissible_rate": "Argmax inadmissible rate",
            "fallback_rate": "Fallback rate",
            "failure_reason": "Failure reason",
        }
    )
    table_pretty.to_csv(out_dir / "table2_kpis.csv", index=False)
    table_pretty.to_markdown(out_dir / "table2_kpis.md", index=False)

    # Also write the mean KPI table (previous default behavior).
    table_mean = pd.DataFrame(rows)
    table_mean_cols = [
        "Environment",
        "Algorithm",
        "success_rate",
        "avg_path_length",
        "path_time_s",
        "avg_curvature_1_m",
        "planning_time_s",
        "tracking_time_s",
        "num_corners",
        "inference_time_s",
        "max_corner_deg",
    ]
    if "argmax_inadmissible_rate" in table_mean.columns:
        table_mean_cols.append("argmax_inadmissible_rate")
    if "fallback_rate" in table_mean.columns:
        table_mean_cols.append("fallback_rate")
    table_mean = table_mean[table_mean_cols]
    table_mean = table_mean.copy()

    w_t = float(args.score_time_weight)
    sr_raw = pd.to_numeric(table_mean["success_rate"], errors="coerce").astype(float)
    denom = sr_raw.clip(lower=1e-6)
    base = pd.to_numeric(table_mean["avg_path_length"], errors="coerce").astype(float) + w_t * pd.to_numeric(
        table_mean["inference_time_s"], errors="coerce"
    ).astype(float)
    planning_cost = (base / denom).astype(float)
    planning_cost = planning_cost.where((sr_raw > 0.0) & np.isfinite(base.to_numpy()), other=float("inf"))
    table_mean["planning_cost"] = planning_cost

    group_keys = ["Environment"]
    n_pt = table_mean.groupby(group_keys, sort=False)["path_time_s"].transform(_minmax_norm).fillna(0.0)
    n_k = table_mean.groupby(group_keys, sort=False)["avg_curvature_1_m"].transform(_minmax_norm).fillna(0.0)
    n_pl = table_mean.groupby(group_keys, sort=False)["planning_time_s"].transform(_minmax_norm).fillna(0.0)
    base_score = (w_pt * n_pt + w_k * n_k + w_pl * n_pl) / w_sum
    sr_denom2 = sr_raw.clip(lower=1e-6)
    composite_score = (base_score / sr_denom2).astype(float)
    composite_score = composite_score.where(sr_raw > 0.0, other=float("inf"))
    table_mean["composite_score"] = composite_score

    table_mean["success_rate"] = pd.to_numeric(table_mean["success_rate"], errors="coerce").astype(float).round(3)
    table_mean["avg_path_length"] = table_mean["avg_path_length"].astype(float).round(4)
    table_mean["path_time_s"] = pd.to_numeric(table_mean["path_time_s"], errors="coerce").astype(float).round(4)
    table_mean["avg_curvature_1_m"] = pd.to_numeric(table_mean["avg_curvature_1_m"], errors="coerce").astype(float).round(6)
    table_mean["planning_time_s"] = pd.to_numeric(table_mean["planning_time_s"], errors="coerce").astype(float).round(5)
    table_mean["tracking_time_s"] = pd.to_numeric(table_mean["tracking_time_s"], errors="coerce").astype(float).round(5)
    table_mean["num_corners"] = pd.to_numeric(table_mean["num_corners"], errors="coerce").round(0).astype("Int64")
    table_mean["inference_time_s"] = table_mean["inference_time_s"].astype(float).round(5)
    table_mean["max_corner_deg"] = pd.to_numeric(table_mean["max_corner_deg"], errors="coerce").round(0).astype("Int64")
    table_mean["planning_cost"] = pd.to_numeric(table_mean["planning_cost"], errors="coerce").astype(float).round(3)
    table_mean["composite_score"] = pd.to_numeric(table_mean["composite_score"], errors="coerce").astype(float).round(3)
    if "argmax_inadmissible_rate" in table_mean.columns:
        table_mean["argmax_inadmissible_rate"] = (
            pd.to_numeric(table_mean["argmax_inadmissible_rate"], errors="coerce").astype(float).round(3)
        )
    if "fallback_rate" in table_mean.columns:
        table_mean["fallback_rate"] = pd.to_numeric(table_mean["fallback_rate"], errors="coerce").astype(float).round(3)
    table_mean.to_csv(out_dir / "table2_kpis_mean_raw.csv", index=False)

    table_mean_pretty = table_mean.rename(
        columns={
            "Algorithm": "Algorithm name",
            "success_rate": "Success rate",
            "avg_path_length": "Average path length (m)",
            "path_time_s": "Path time (s)",
            "avg_curvature_1_m": "Average curvature (1/m)",
            "planning_time_s": "Planning time (s)",
            "tracking_time_s": "Tracking time (s)",
            "num_corners": "Number of path corners",
            "inference_time_s": "Compute time (s)",
            "max_corner_deg": "Max corner degree (掳)",
            "planning_cost": "Planning cost (m)",
            "composite_score": "Composite score",
            "argmax_inadmissible_rate": "Argmax inadmissible rate",
            "fallback_rate": "Fallback rate",
        }
    )
    table_mean_pretty.to_csv(out_dir / "table2_kpis_mean.csv", index=False)
    table_mean_pretty.to_markdown(out_dir / "table2_kpis_mean.md", index=False)

    if plt is None or mpatches is None or mticker is None:
        progress_write("[infer] NOTE: matplotlib not available; skipping path/control figures.")
        print(f"Wrote: {out_dir / 'table2_kpis.csv'}")
        print(f"Wrote: {out_dir / 'table2_kpis_raw.csv'}")
        print(f"Wrote: {out_dir / 'table2_kpis.md'}")
        print(f"Wrote: {out_dir / 'table2_kpis_mean.csv'}")
        print(f"Wrote: {out_dir / 'table2_kpis_mean_raw.csv'}")
        print(f"Wrote: {out_dir / 'table2_kpis_mean.md'}")
        print(f"Run dir: {out_dir}")
        return 0

    # Plot Fig. 12-style paths
    styles = {
        "MLP-DQN": dict(color="tab:blue", linestyle="-", linewidth=2.0),
        "MLP-DDQN": dict(color="tab:orange", linestyle="-", linewidth=2.0),
        "CNN-DQN": dict(color="tab:green", linestyle="-", linewidth=2.0),
        "CNN-DDQN": dict(color="tab:red", linestyle="-", linewidth=2.0),
        "MLP-PDDQN": dict(color="tab:cyan", linestyle="-", linewidth=2.0),
        "CNN-PDDQN": dict(color="tab:pink", linestyle="-", linewidth=2.0),
        # Legacy short labels (treated as MLP variants).
        "DQN": dict(color="tab:blue", linestyle="-", linewidth=2.0),
        "DDQN": dict(color="tab:orange", linestyle="-", linewidth=2.0),
        # Baselines.
        "A*": dict(color="tab:gray", linestyle="-", linewidth=2.0),
        "Hybrid A*": dict(color="tab:purple", linestyle="-", linewidth=2.0),
        "RRT": dict(color="tab:brown", linestyle="-", linewidth=2.0),
        "A*-MPC": dict(color="black", linestyle="--", linewidth=2.0),
        "Hybrid A*-MPC": dict(color="tab:purple", linestyle="--", linewidth=2.0),
        "RRT-MPC": dict(color="tab:brown", linestyle="--", linewidth=2.0),
        # Back-compat label.
        "RRT*": dict(color="tab:brown", linestyle="-", linewidth=2.0),
    }

    def write_paths_figure(
        *,
        panels: list[tuple[str, int]],
        out_path: Path,
        suptitle: str,
        multi_pair_titles: bool = False,
    ) -> None:
        n_panels = int(len(panels))
        if n_panels <= 0:
            return
        cols = 1 if n_panels <= 1 else 2
        rows_n = int(math.ceil(float(n_panels) / float(cols)))
        fig, axes = plt.subplots(rows_n, cols, figsize=(5.2 * cols, 5.2 * rows_n))
        axes = np.atleast_1d(axes).ravel()

        for i, (env_name, run_idx) in enumerate(panels):
            ax = axes[i]
            env_base = str(env_name).split("::", 1)[0]
            suite = str(env_name).split("::", 1)[1] if "::" in str(env_name) else ""
            spec = get_map_spec(env_base)
            grid = spec.obstacle_grid()
            title = f"Env. ({env_base})"
            if suite:
                title = f"Env. ({env_base})/{suite}"
            if multi_pair_titles:
                title = f"Env. ({env_base}) #{int(run_idx)}"
            plot_env(ax, grid, title=title)

            meta = plot_meta.get((env_name, int(run_idx))) or plot_meta.get((env_name, 0), {})

            spx = float(meta.get("plot_start_x", float(spec.start_xy[0])))
            spy = float(meta.get("plot_start_y", float(spec.start_xy[1])))
            gpx = float(meta.get("plot_goal_x", float(spec.goal_xy[0])))
            gpy = float(meta.get("plot_goal_y", float(spec.goal_xy[1])))

            ax.scatter(
                [spx],
                [spy],
                marker="*",
                s=140,
                color="blue",
                label="Start",
            )
            ax.text(spx - 1.0, spy - 1.0, "SP", fontsize=9, color="black")
            ax.scatter(
                [gpx],
                [gpy],
                marker="*",
                s=140,
                color="red",
                label="Goal",
            )
            ax.text(gpx - 1.0, gpy - 1.0, "TP", fontsize=9, color="black")

            tol = float(meta.get("goal_tol_cells", 0.0))
            if tol > 0.0:
                ax.add_patch(
                    mpatches.Circle(
                        (float(gpx), float(gpy)),
                        radius=float(tol),
                        fill=False,
                        edgecolor="crimson",
                        linestyle="--",
                        linewidth=1.8,
                        alpha=0.95,
                        zorder=6,
                    )
                )

            env_paths = paths_for_plot.get((env_name, int(run_idx)), {})
            for algo_name, trace in env_paths.items():
                path = trace.path_xy_cells
                if not path:
                    continue
                pts = np.array(path, dtype=np.float32)
                pts_s = chaikin_smooth(pts, iterations=2)
                style = styles.get(algo_name, dict(color="black", linestyle="-", linewidth=1.5))
                label = algo_name if trace.success else f"{algo_name} (fail)"
                alpha = 1.0 if trace.success else 0.55
                ax.plot(pts_s[:, 0], pts_s[:, 1], label=label, alpha=alpha, **style)
                end_marker = "o" if trace.success else "x"
                ax.scatter(
                    [float(pts_s[-1, 0])],
                    [float(pts_s[-1, 1])],
                    marker=end_marker,
                    s=28,
                    color=style["color"],
                    label="_nolegend_",
                )

                if float(meta.get("veh_length_cells", 0.0)) > 0.0 and float(meta.get("veh_width_cells", 0.0)) > 0.0:
                    draw_vehicle_boxes(
                        ax,
                        trace,
                        length_cells=float(meta["veh_length_cells"]),
                        width_cells=float(meta["veh_width_cells"]),
                        color=str(style["color"]),
                    )

            ax.legend(fontsize=8, loc="lower right")

        for ax in axes[n_panels:]:
            ax.axis("off")

        fig.suptitle(str(suptitle))
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    def write_controls_figure(
        *,
        panels: list[tuple[str, int]],
        out_path: Path,
        suptitle: str,
        multi_pair_titles: bool = False,
    ) -> None:
        n_panels = int(len(panels))
        if n_panels <= 0:
            return
        if not any(bool(controls_for_plot.get((env_name, int(run_idx)), {})) for env_name, run_idx in panels):
            return

        fig, axes = plt.subplots(n_panels, 2, figsize=(10.8, 3.2 * n_panels), squeeze=False)

        for i, (env_name, run_idx) in enumerate(panels):
            ax_v = axes[i, 0]
            ax_d = axes[i, 1]

            env_base = str(env_name).split("::", 1)[0]
            suite = str(env_name).split("::", 1)[1] if "::" in str(env_name) else ""
            title = f"Env. ({env_base})"
            if suite:
                title = f"Env. ({env_base})/{suite}"
            if multi_pair_titles:
                title = f"Env. ({env_base}) #{int(run_idx)}"

            ctrl = controls_for_plot.get((env_name, int(run_idx)), {})
            env_paths = paths_for_plot.get((env_name, int(run_idx)), {})
            if not ctrl:
                ax_v.axis("off")
                ax_d.axis("off")
                ax_v.text(0.5, 0.5, f"{title}\n(no control traces)", ha="center", va="center", fontsize=9)
                continue

            for algo_name, tr in ctrl.items():
                style = styles.get(algo_name, dict(color="black", linestyle="-", linewidth=1.5))
                ok = True
                if algo_name in env_paths:
                    ok = bool(env_paths[algo_name].success)
                label = algo_name if ok else f"{algo_name} (fail)"
                alpha = 1.0 if ok else 0.55
                ax_v.plot(tr.t_s, tr.v_m_s, label=label, alpha=alpha, **style)
                ax_d.plot(tr.t_s, np.degrees(tr.delta_rad), label=label, alpha=alpha, **style)

            ax_v.set_title(f"{title}: Speed")
            ax_v.set_xlabel("t (s)")
            ax_v.set_ylabel("v (m/s)")
            ax_v.grid(True, alpha=0.22, linewidth=0.6)

            ax_d.set_title(f"{title}: Steering")
            ax_d.set_xlabel("t (s)")
            ax_d.set_ylabel("delta (deg)")
            ax_d.grid(True, alpha=0.22, linewidth=0.6)
            ax_d.legend(fontsize=8, loc="best")

        fig.suptitle(str(suptitle))
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    envs_to_plot = list(args.envs)[:4]
    panels: list[tuple[str, int]] = []
    env0_base = str(envs_to_plot[0]).split("::", 1)[0] if envs_to_plot else ""
    multi_pair_fig = (
        bool(getattr(args, "random_start_goal", False))
        and int(len(args.envs)) == 1
        and int(args.runs) >= 4
        and str(env0_base) in set(FOREST_ENV_ORDER)
    )
    if envs_to_plot and multi_pair_fig:
        base = int(getattr(args, "plot_run_idx", 0))
        panels = [(str(envs_to_plot[0]), (base + k) % int(args.runs)) for k in range(4)]
    else:
        for env_name in envs_to_plot:
            env_base = str(env_name).split("::", 1)[0]
            run_idx = 0
            if bool(getattr(args, "random_start_goal", False)) and str(env_base) in set(FOREST_ENV_ORDER):
                run_idx = int(getattr(args, "plot_run_idx", 0))
            panels.append((str(env_name), int(run_idx)))

    fig12_path = out_dir / "fig12_paths.png"
    write_paths_figure(
        panels=panels,
        out_path=fig12_path,
        suptitle="Simulation results of different path-planning methods",
        multi_pair_titles=bool(multi_pair_fig),
    )

    print(f"Wrote: {fig12_path}")

    fig13_path = out_dir / "fig13_controls.png"
    write_controls_figure(
        panels=panels,
        out_path=fig13_path,
        suptitle="Speed and steering of different path-planning methods",
        multi_pair_titles=bool(multi_pair_fig),
    )
    if fig13_path.exists():
        print(f"Wrote: {fig13_path}")

    if bool(getattr(args, "plot_run_groups", False)) and bool(getattr(args, "random_start_goal", False)):
        group_size = max(1, int(getattr(args, "plot_run_group_size", 4)))
        with_controls = bool(getattr(args, "plot_run_groups_with_controls", True))
        n_runs = int(getattr(args, "runs", 0))

        for env_name in list(args.envs):
            env_label = str(env_name)
            env_base = str(env_label).split("::", 1)[0]
            base_slug = _safe_slug(env_base)

            for group_start in range(0, int(n_runs), int(group_size)):
                group_end = min(int(n_runs), int(group_start) + int(group_size)) - 1
                group_panels = [(str(env_label), int(i)) for i in range(int(group_start), int(group_end) + 1)]
                if not group_panels:
                    continue

                grouped_path = out_dir / f"fig12_paths_{base_slug}_runs_{int(group_start):02d}_{int(group_end):02d}.png"
                write_paths_figure(
                    panels=group_panels,
                    out_path=grouped_path,
                    suptitle=(
                        "Simulation results of different path-planning methods "
                        f"({env_base}, runs {int(group_start)}-{int(group_end)})"
                    ),
                    multi_pair_titles=True,
                )
                print(f"Wrote: {grouped_path}")

                if with_controls:
                    grouped_controls = (
                        out_dir / f"fig13_controls_{base_slug}_runs_{int(group_start):02d}_{int(group_end):02d}.png"
                    )
                    write_controls_figure(
                        panels=group_panels,
                        out_path=grouped_controls,
                        suptitle=(
                            "Speed and steering of different path-planning methods "
                            f"({env_base}, runs {int(group_start)}-{int(group_end)})"
                        ),
                        multi_pair_titles=True,
                    )
                    if grouped_controls.exists():
                        print(f"Wrote: {grouped_controls}")

    # Optional: one figure per run index (short + long in the same image).
    if bool(getattr(args, "plot_pair_runs", False)) and bool(getattr(args, "random_start_goal", False)) and bool(
        getattr(args, "rand_two_suites", False)
    ):
        per_run_cap = int(getattr(args, "plot_pair_runs_max", 10))
        per_run_n = int(args.runs)
        if per_run_cap > 0:
            per_run_n = min(int(per_run_n), int(per_run_cap))

        pairs_by_base: dict[str, dict[str, str]] = {}
        for env_name in args.envs:
            env_case = str(env_name)
            if "::" not in env_case:
                continue
            base, suite = env_case.split("::", 1)
            base = str(base).strip()
            suite = str(suite).strip()
            if suite not in {"short", "long"}:
                continue
            pairs_by_base.setdefault(base, {})[suite] = env_case

        for base, suite_map in pairs_by_base.items():
            if "short" not in suite_map or "long" not in suite_map:
                continue
            base_slug = _safe_slug(base)
            for run_idx in range(int(per_run_n)):
                out_path = out_dir / f"fig12_paths_{base_slug}_run_{run_idx:02d}.png"
                write_paths_figure(
                    panels=[(suite_map["short"], int(run_idx)), (suite_map["long"], int(run_idx))],
                    out_path=out_path,
                    suptitle=f"Simulation results of different path-planning methods (run {run_idx})",
                )
                print(f"Wrote: {out_path}")

    print(f"Wrote: {out_dir / 'table2_kpis.csv'}")
    print(f"Wrote: {out_dir / 'table2_kpis_raw.csv'}")
    print(f"Wrote: {out_dir / 'table2_kpis.md'}")
    print(f"Wrote: {out_dir / 'table2_kpis_mean.csv'}")
    print(f"Wrote: {out_dir / 'table2_kpis_mean_raw.csv'}")
    print(f"Wrote: {out_dir / 'table2_kpis_mean.md'}")
    print(f"Run dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
