from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import TextIO

from forest_vehicle_dqn.runtime import configure_runtime, select_device, torch_runtime_info
from forest_vehicle_dqn.runs import create_run_dir, resolve_experiment_dir

configure_runtime()

import matplotlib.pyplot as plt
try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    import gym  # type: ignore
import numpy as np
import pandas as pd
import torch

from forest_vehicle_dqn.agents import AgentConfig, DQNFamilyAgent, parse_rl_algo
from forest_vehicle_dqn.baselines.mpc_local_planner import MPCConfig
from forest_vehicle_dqn.baselines.pathplan import AStarCurveOptConfig, grid_map_from_obstacles, plan_grid_astar
from forest_vehicle_dqn.config_io import apply_config_defaults, load_json, resolve_config_path, select_section
from forest_vehicle_dqn.env import AMRBicycleEnv, AMRGridEnv, RewardWeights, bicycle_integrate_one_step
from forest_vehicle_dqn.live_view_pygame import TrainLiveViewer
from forest_vehicle_dqn.maps import FOREST_ENV_ORDER, get_map_spec
from forest_vehicle_dqn.replay_buffer import FLAG_HAZARD, FLAG_NEAR_GOAL, FLAG_STUCK


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    w = int(window)
    if x.size < w:
        return x
    kernel = np.ones((w,), dtype=np.float32) / float(w)
    return np.convolve(x, kernel, mode="same")


def format_elapsed_s(seconds: float) -> str:
    total = max(0.0, float(seconds))
    if total < 60.0:
        return f"{total:.1f}s"
    minutes, sec = divmod(total, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m{sec:04.1f}s"
    hours, rem_minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h{int(rem_minutes):02d}m{sec:04.1f}s"


def make_progress_writer(progress: bool, *, flow_log_fp: TextIO | None = None) -> Callable[[str], None]:
    tqdm_cls = None
    if progress:
        try:
            from tqdm import tqdm as _tqdm  # type: ignore
        except Exception:
            tqdm_cls = None
        else:
            tqdm_cls = _tqdm

    def progress_write(msg: str) -> None:
        if flow_log_fp is not None:
            try:
                flow_log_fp.write(f"{str(msg)}\n")
                flow_log_fp.flush()
            except Exception:
                pass
        if not bool(progress):
            return
        if tqdm_cls is not None:
            tqdm_cls.write(str(msg), file=sys.stderr)
            return
        print(str(msg), file=sys.stderr, flush=True)

    return progress_write


def plot_training_eval_metrics(df_eval: pd.DataFrame, *, out_path: Path) -> None:
    if df_eval.empty:
        return

    if "phase" in df_eval.columns:
        phase_series = df_eval["phase"].astype(str)
        if bool((phase_series == "train").any()):
            df_eval = df_eval[phase_series == "train"].copy()
    if df_eval.empty:
        return

    metrics_all: list[tuple[str, str]] = [
        ("sr_short", "SR short"),
        ("sr_long", "SR long"),
        ("sr_all", "SR all"),
        ("ratio_short", "Distance ratio short"),
        ("ratio_long", "Distance ratio long"),
    ]
    metrics: list[tuple[str, str]] = [(col, title) for col, title in metrics_all if col in df_eval.columns]
    if not metrics:
        return

    df_eval = df_eval.copy()
    if "episode" in df_eval.columns:
        df_eval["episode"] = pd.to_numeric(df_eval["episode"], errors="coerce")

    envs = [str(x) for x in df_eval["env"].drop_duplicates().tolist()]
    if not envs:
        return

    rows_n = int(len(envs))
    cols_n = int(len(metrics))
    fig, axes = plt.subplots(
        rows_n,
        cols_n,
        figsize=(4.3 * cols_n, 2.8 * rows_n),
        sharex=False,
        sharey=False,
    )
    axes_arr = np.atleast_2d(axes)

    algo_label = {
        "mlp-dqn": "MLP-DQN",
        "mlp-ddqn": "MLP-DDQN",
        "mlp-pddqn": "MLP-PDDQN",
        "cnn-dqn": "CNN-DQN",
        "cnn-ddqn": "CNN-DDQN",
        "cnn-pddqn": "CNN-PDDQN",
        # Legacy (older runs)
        "dqn": "DQN",
        "ddqn": "DDQN",
        "iddqn": "MLP-PDDQN",
        "cnn-iddqn": "CNN-PDDQN",
    }
    present = [str(x) for x in df_eval["algo"].dropna().drop_duplicates().tolist()]
    pref = ("mlp-dqn", "mlp-ddqn", "mlp-pddqn", "cnn-dqn", "cnn-ddqn", "cnn-pddqn", "dqn", "ddqn")
    ordered = [a for a in pref if a in present] + [a for a in present if a not in pref]
    algo_defs: list[tuple[str, str]] = [(a, algo_label.get(a, a.upper())) for a in ordered]
    for i, env_name in enumerate(envs):
        for j, (col, title) in enumerate(metrics):
            ax = axes_arr[i, j]
            for algo, label in algo_defs:
                sub = df_eval[(df_eval["env"] == env_name) & (df_eval["algo"] == algo)].copy()
                if sub.empty:
                    continue
                sub = sub.sort_values("episode")
                x = sub["episode"].to_numpy()
                y = pd.to_numeric(sub[col], errors="coerce").astype(float).to_numpy()
                if col in ("ratio_short", "ratio_long"):
                    y = np.where(np.isfinite(y), y, np.nan)
                ax.plot(x, y, label=label, linewidth=1.0)

            if i == 0:
                ax.set_title(title)
            if j == 0:
                ax.set_ylabel(str(env_name))
            if i == rows_n - 1:
                ax.set_xlabel("Episodes")
            if col in ("sr_short", "sr_long", "sr_all"):
                ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.25)

    handles, labels = axes_arr[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), fontsize=9, frameon=False)
    fig.suptitle("Training progress metrics (fixed suites)", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def forest_demo_target(
    *,
    learning_starts: int,
    batch_size: int,
    target_mult: float = 40.0,
    target_cap: int = 20_000,
) -> int:
    # Forest long-horizon runs can fall into a "stop until stuck" local optimum unless the replay
    # is initially dominated by successful expert trajectories.
    mult = max(1.0, float(target_mult))
    target = max(int(float(learning_starts) * float(mult)), int(batch_size))
    cap = int(target_cap)
    if cap > 0:
        return int(min(int(target), int(cap)))
    return int(target)


def forest_expert_action(
    env: AMRBicycleEnv,
    *,
    forest_expert: str,
    horizon_steps: int,
    w_clearance: float,
    mpc_cfg: MPCConfig,
    astar_curve_cfg: AStarCurveOptConfig,
    astar_seed: int,
    astar_replan_attempt: int,
    astar_timeout_s: float,
    astar_max_expanded: int,
    stage: str = "unknown",
) -> int:
    h = max(1, int(horizon_steps))
    expert = str(forest_expert).lower().strip()
    if expert == "auto":
        expert = "hybrid_astar"

    if expert == "hybrid_astar":
        # Safer Hybrid A* tracking for demonstrations / guided exploration.
        # The shorter-horizon aggressive tracker can collide on harder maps (notably forest_a).
        return env.expert_action_hybrid_astar(
            lookahead_points=5,
            horizon_steps=max(15, h),
            w_target=0.2,
            w_heading=0.2,
            w_clearance=float(w_clearance),
            w_speed=0.0,
        )

    if expert == "astar_mpc":
        return env.expert_action_astar_mpc(
            mpc_cfg=mpc_cfg,
            curve_cfg=astar_curve_cfg,
            seed=int(astar_seed),
            replan_attempt=int(astar_replan_attempt),
            timeout_s=float(astar_timeout_s),
            max_expanded=int(astar_max_expanded),
        )

    if expert == "hybrid_astar_mpc":
        return env.expert_action_hybrid_astar_mpc(
            mpc_cfg=mpc_cfg,
        )

    raise ValueError("forest_expert must be one of: auto, hybrid_astar, astar_mpc, hybrid_astar_mpc")


def forest_stop_action(env: AMRBicycleEnv) -> int:
    """Pick a stop-oriented action near the goal to drive v, delta toward 0 safely."""
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


def _forest_policy_action_from_q(
    env: AMRBicycleEnv,
    q: torch.Tensor,
    *,
    forest_adm_horizon: int,
    forest_topk: int,
    forest_min_od_m: float,
    forest_min_progress_m: float,
    forest_no_fallback: bool,
) -> tuple[int, bool]:
    """Infer-style forest policy action from Q values; returns (action, argmax_inadmissible)."""
    adm_h = max(1, int(forest_adm_horizon))
    topk_k = max(1, int(forest_topk))
    strict_no_fallback = bool(forest_no_fallback)
    min_od = float(forest_min_od_m)
    min_prog = float(forest_min_progress_m)

    if not bool(strict_no_fallback):
        reached_pose = bool(
            env._goal_pose_reached(
                d_goal_m=float(env._distance_to_goal_m()),
                alpha_rad=float(env._goal_relative_angle_rad()),
            )
        )
        reached_stop = bool(env._goal_stop_reached(v_m_s=float(env._v_m_s), delta_rad=float(env._delta_rad)))
        if bool(reached_pose) and (not bool(reached_stop)):
            return int(forest_stop_action(env)), False

    a0 = int(torch.argmax(q).item())
    a = int(a0)
    a0_adm = bool(
        env.is_action_admissible(
            int(a0),
            horizon_steps=int(adm_h),
            min_od_m=float(min_od),
            min_progress_m=float(min_prog),
        )
    )

    if (not bool(strict_no_fallback)) and (not bool(a0_adm)):
        chosen: int | None = None
        kk = int(min(int(topk_k), int(q.numel())))
        topk = torch.topk(q, k=kk, dim=0).indices.detach().cpu().numpy()
        for cand in topk.tolist():
            cand_i = int(cand)
            if cand_i == int(a0):
                continue
            if bool(
                env.is_action_admissible(
                    int(cand_i),
                    horizon_steps=int(adm_h),
                    min_od_m=float(min_od),
                    min_progress_m=float(min_prog),
                )
            ):
                chosen = int(cand_i)
                break

        if chosen is None:
            prog_mask = env.admissible_action_mask(
                horizon_steps=int(adm_h),
                min_od_m=float(min_od),
                min_progress_m=float(min_prog),
                fallback_to_safe=False,
            )
            if bool(prog_mask.any()):
                q_masked = q.clone()
                q_masked[torch.from_numpy(~prog_mask).to(q.device)] = torch.finfo(q_masked.dtype).min
                chosen = int(torch.argmax(q_masked).item())

        if chosen is not None:
            a = int(chosen)

    return int(a), (not bool(a0_adm))


def _forest_pair_dist_ratio(env: AMRBicycleEnv) -> float:
    sx, sy = int(env.start_xy[0]), int(env.start_xy[1])
    gx, gy = int(env.goal_xy[0]), int(env.goal_xy[1])
    d_m = float(math.hypot(float(gx - sx) * float(env.cell_size_m), float(gy - sy) * float(env.cell_size_m)))
    diag_m = max(1e-6, float(getattr(env, "_diag_m", 1.0)))
    return float(np.clip(float(d_m) / float(diag_m), 0.0, 1.0))


def _forest_effective_rand_max_dist_m(
    env: AMRBicycleEnv,
    *,
    rand_min_dist_m: float,
    rand_max_dist_m: float | None,
    curriculum_progress: float | None,
) -> float | None:
    diag_m = max(1e-6, float(getattr(env, "_diag_m", 1.0)))
    min_dist = max(0.0, float(rand_min_dist_m))

    hard_max: float | None = None
    if rand_max_dist_m is not None:
        raw_max = float(rand_max_dist_m)
        if raw_max > 0.0:
            hard_max = float(raw_max)

    full_max = float(diag_m if hard_max is None else min(float(hard_max), float(diag_m)))
    if float(full_max) <= float(min_dist) + 1e-6:
        return None if hard_max is None else float(full_max)

    if curriculum_progress is None:
        return None if hard_max is None else float(full_max)

    p = float(np.clip(float(curriculum_progress), 0.0, 1.0))
    short_cap = min(float(full_max), max(float(min_dist) + 8.0, float(min_dist) * 1.8))
    if float(short_cap) <= float(min_dist) + 1e-6:
        short_cap = float(min(float(full_max), float(min_dist) + 2.0))
    eff_max = float(short_cap + (float(full_max) - float(short_cap)) * float(p))
    if eff_max <= float(min_dist) + 1e-6:
        eff_max = float(full_max)
    return float(eff_max)


def _astar_path_length_m_for_pair(
    env: AMRBicycleEnv,
    *,
    start_xy: tuple[int, int],
    goal_xy: tuple[int, int],
    timeout_s: float,
    max_expanded: int,
) -> float | None:
    grid_map = grid_map_from_obstacles(grid_y0_bottom=env.grid, cell_size_m=float(env.cell_size_m))
    res = plan_grid_astar(
        grid_map=grid_map,
        start_xy=(int(start_xy[0]), int(start_xy[1])),
        goal_xy=(int(goal_xy[0]), int(goal_xy[1])),
        timeout_s=float(timeout_s),
        max_expanded=int(max_expanded),
        seed=0,
        replan_attempt=0,
        curve_opt_cfg=None,
        footprint=None,
        ackermann_params=None,
    )
    if not bool(res.success):
        return None
    raw_cost = res.stats.get("path_cost")
    if raw_cost is None:
        return None
    try:
        path_cost_cells = float(raw_cost)
    except Exception:
        return None
    if not math.isfinite(path_cost_cells) or path_cost_cells <= 0.0:
        return None
    return float(path_cost_cells) * float(env.cell_size_m)


def _sample_fixed_train_eval_pairs(
    env: AMRBicycleEnv,
    *,
    runs_per_suite: int,
    seed_base: int,
    short_min_dist_m: float,
    short_max_dist_m: float | None,
    long_min_dist_m: float,
    long_max_dist_m: float | None,
    rand_tries: int,
    rand_edge_margin_m: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    short_pairs: list[dict[str, object]] = []
    long_pairs: list[dict[str, object]] = []
    short_max = 0.0 if short_max_dist_m is None else float(short_max_dist_m)
    long_max = 0.0 if long_max_dist_m is None else float(long_max_dist_m)

    for i in range(max(1, int(runs_per_suite))):
        env.reset(
            seed=int(seed_base) + 10_000 + int(i),
            options={
                "random_start_goal": True,
                "rand_min_dist_m": float(short_min_dist_m),
                "rand_max_dist_m": float(short_max),
                "rand_fixed_prob": 0.0,
                "rand_tries": int(rand_tries),
                "rand_edge_margin_m": float(rand_edge_margin_m),
            },
        )
        short_pairs.append(
            {
                "start_xy": (int(env.start_xy[0]), int(env.start_xy[1])),
                "goal_xy": (int(env.goal_xy[0]), int(env.goal_xy[1])),
            }
        )

        env.reset(
            seed=int(seed_base) + 20_000 + int(i),
            options={
                "random_start_goal": True,
                "rand_min_dist_m": float(long_min_dist_m),
                "rand_max_dist_m": float(long_max),
                "rand_fixed_prob": 0.0,
                "rand_tries": int(rand_tries),
                "rand_edge_margin_m": float(rand_edge_margin_m),
            },
        )
        long_pairs.append(
            {
                "start_xy": (int(env.start_xy[0]), int(env.start_xy[1])),
                "goal_xy": (int(env.goal_xy[0]), int(env.goal_xy[1])),
            }
        )

    return short_pairs, long_pairs


def _eval_train_progress_suites(
    env: AMRBicycleEnv,
    *,
    agent: DQNFamilyAgent,
    pairs_short: list[dict[str, object]],
    pairs_long: list[dict[str, object]],
    seed_base: int,
    max_steps: int,
    adm_horizon: int,
    topk: int,
    min_od_m: float,
    min_progress_m: float,
    forest_no_fallback: bool,
    astar_timeout_s: float,
    astar_max_expanded: int,
) -> dict[str, float | int]:
    def extract_xy_m(info: dict[str, object]) -> tuple[float, float] | None:
        if "pose_m" in info:
            try:
                x_m, y_m, _ = info["pose_m"]  # type: ignore[misc]
                return (float(x_m), float(y_m))
            except Exception:
                return None
        if "agent_xy" in info:
            try:
                ax, ay = info["agent_xy"]  # type: ignore[misc]
                return (float(ax) * float(env.cell_size_m), float(ay) * float(env.cell_size_m))
            except Exception:
                return None
        return None

    def eval_one_pair(pair: dict[str, object], idx: int, suite_name: str) -> dict[str, float | bool | int]:
        obs_eval, info0 = env.reset(
            seed=int(seed_base) + (30_000 if suite_name == "short" else 40_000) + int(idx),
            options={
                "start_xy": pair["start_xy"],
                "goal_xy": pair["goal_xy"],
            },
        )
        done_eval = False
        trunc_eval = False
        steps_eval = 0
        path_len_m = 0.0
        last_xy_m = extract_xy_m(dict(info0))
        last_info_eval: dict[str, object] = {}
        argmax_inadmissible_steps = 0
        decision_steps = 0

        while not (done_eval or trunc_eval) and int(steps_eval) < int(max_steps):
            steps_eval += 1
            with torch.no_grad():
                x = torch.from_numpy(obs_eval.astype(np.float32, copy=False)).to(agent.device)
                q = agent._q_values(agent.q(x.unsqueeze(0))).squeeze(0)
                a, argmax_inadmissible = _forest_policy_action_from_q(
                    env,
                    q,
                    forest_adm_horizon=int(adm_horizon),
                    forest_topk=int(topk),
                    forest_min_od_m=float(min_od_m),
                    forest_min_progress_m=float(min_progress_m),
                    forest_no_fallback=bool(forest_no_fallback),
                )
            decision_steps += 1
            if bool(argmax_inadmissible):
                argmax_inadmissible_steps += 1
            obs_eval, _, done_eval, trunc_eval, info_eval = env.step(int(a))
            last_info_eval = dict(info_eval)
            xy_m = extract_xy_m(last_info_eval)
            if last_xy_m is not None and xy_m is not None:
                path_len_m += float(math.hypot(float(xy_m[0]) - float(last_xy_m[0]), float(xy_m[1]) - float(last_xy_m[1])))
            last_xy_m = xy_m

        reached = bool(last_info_eval.get("reached", False))
        astar_len_m = _astar_path_length_m_for_pair(
            env,
            start_xy=(int(pair["start_xy"][0]), int(pair["start_xy"][1])),
            goal_xy=(int(pair["goal_xy"][0]), int(pair["goal_xy"][1])),
            timeout_s=float(astar_timeout_s),
            max_expanded=int(astar_max_expanded),
        )
        valid_ratio = bool(reached) and (astar_len_m is not None) and float(astar_len_m) > 1e-6 and math.isfinite(float(path_len_m))
        ratio = (float(path_len_m) / float(astar_len_m)) if valid_ratio else float("nan")
        return {
            "reached": bool(reached),
            "ratio_valid": bool(valid_ratio),
            "ratio": float(ratio),
            "argmax_inadmissible_steps": int(argmax_inadmissible_steps),
            "decision_steps": int(decision_steps),
        }

    short_results = [eval_one_pair(p, i, "short") for i, p in enumerate(pairs_short)]
    long_results = [eval_one_pair(p, i, "long") for i, p in enumerate(pairs_long)]

    def aggregate(res: list[dict[str, float | bool | int]]) -> tuple[float, float, int, int, float]:
        n = max(1, int(len(res)))
        succ = int(sum(1 for r in res if bool(r["reached"])))
        ratios = [float(r["ratio"]) for r in res if bool(r["ratio_valid"]) and math.isfinite(float(r["ratio"]))]
        ratio_mean = float(np.mean(ratios)) if ratios else float("nan")
        inadmissible_steps = int(sum(int(r.get("argmax_inadmissible_steps", 0)) for r in res))
        decision_steps = int(sum(int(r.get("decision_steps", 0)) for r in res))
        inadmissible_rate = float(inadmissible_steps) / float(max(1, decision_steps))
        return float(succ) / float(n), float(ratio_mean), int(len(ratios)), int(n), float(inadmissible_rate)

    sr_short, ratio_short, n_valid_short, n_short, inad_rate_short = aggregate(short_results)
    sr_long, ratio_long, n_valid_long, n_long, inad_rate_long = aggregate(long_results)
    sr_all = float((sr_short * float(n_short) + sr_long * float(n_long)) / float(max(1, n_short + n_long)))
    inad_rate_all = float(
        (float(inad_rate_short) * float(n_short) + float(inad_rate_long) * float(n_long))
        / float(max(1, n_short + n_long))
    )

    return {
        "sr_short": float(sr_short),
        "sr_long": float(sr_long),
        "sr_all": float(sr_all),
        "ratio_short": float(ratio_short),
        "ratio_long": float(ratio_long),
        "n_valid_short": int(n_valid_short),
        "n_valid_long": int(n_valid_long),
        "n_short": int(n_short),
        "n_long": int(n_long),
        "argmax_inadmissible_rate_short": float(inad_rate_short),
        "argmax_inadmissible_rate_long": float(inad_rate_long),
        "argmax_inadmissible_rate_all": float(inad_rate_all),
    }


def _train_progress_primary_ratio(row: dict[str, float | int]) -> float:
    ratio_short = float(row.get("ratio_short", float("nan")))
    ratio_long = float(row.get("ratio_long", float("nan")))
    valid_short = int(row.get("n_valid_short", 0))
    valid_long = int(row.get("n_valid_long", 0))
    vals: list[float] = []
    if valid_short > 0 and math.isfinite(ratio_short):
        vals.append(float(ratio_short))
    if valid_long > 0 and math.isfinite(ratio_long):
        vals.append(float(ratio_long))
    if not vals:
        return float("inf")
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def collect_forest_demos(
    env: AMRBicycleEnv,
    *,
    target: int,
    seed: int,
    forest_curriculum: bool,
    curriculum_band_m: float,
    forest_random_start_goal: bool,
    forest_rand_min_dist_m: float,
    forest_rand_max_dist_m: float | None,
    forest_rand_fixed_prob: float,
    forest_rand_tries: int,
    forest_rand_edge_margin_m: float,
    forest_expert: str,
    forest_demo_horizon: int,
    forest_adm_horizon: int,
    forest_min_od_m: float,
    forest_min_progress_m: float,
    forest_use_admissible_next_mask: bool,
    forest_demo_w_clearance: float,
    mpc_cfg: MPCConfig,
    astar_curve_cfg: AStarCurveOptConfig,
    astar_seed: int,
    astar_timeout_s: float,
    astar_max_expanded: int,
    forest_demo_filter_min_progress_ratio: float,
    forest_demo_filter_min_progress_per_step_m: float,
    forest_demo_filter_max_steps: int,
    progress_write: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    expert = str(forest_expert).lower().strip()
    if expert == "auto":
        expert = "hybrid_astar"

    obs_dim = int(env.observation_space.shape[0])
    n = max(0, int(target))
    obs_buf = np.zeros((n, obs_dim), dtype=np.float32)
    next_obs_buf = np.zeros((n, obs_dim), dtype=np.float32)
    act_buf = np.zeros((n,), dtype=np.int64)
    rew_buf = np.zeros((n,), dtype=np.float32)
    next_mask_buf = np.ones((n, int(env.action_space.n)), dtype=np.bool_)
    done_buf = np.zeros((n,), dtype=np.float32)
    trunc_buf = np.zeros((n,), dtype=np.float32)

    def log(msg: str) -> None:
        if progress_write is not None:
            progress_write(str(msg))

    added = 0
    demo_ep = 0
    adm_h = max(1, int(forest_adm_horizon))
    min_od_m = float(forest_min_od_m)
    min_prog_m = float(forest_min_progress_m)
    use_adm_next_mask = bool(forest_use_admissible_next_mask)
    full_next_mask = np.ones((int(env.action_space.n),), dtype=np.bool_)
    t_collect_start = time.perf_counter()
    t_last_log = t_collect_start
    added_last_log = -1
    demo_prog = np.linspace(0.0, 1.0, num=5, dtype=np.float32)
    bucket_added = np.zeros((3,), dtype=np.int64)
    bucket_target = np.array([n // 3, n // 3, n - 2 * (n // 3)], dtype=np.int64)
    bucket_ratio_bounds: tuple[tuple[float, float], ...] = (
        (0.00, 0.33),
        (0.33, 0.66),
        (0.66, 1.01),
    )

    def choose_bucket() -> int | None:
        remaining = bucket_target - bucket_added
        remaining = np.maximum(remaining, 0)
        if int(np.sum(remaining)) <= 0:
            return None
        return int(np.argmax(remaining))

    # Only keep demonstrations from successful (goal-reaching) episodes.
    # Otherwise, failed expert rollouts can dominate DQfD losses + the demo-preserving replay buffer
    # and lock the policy into the degenerate "stop until stuck" behavior.
    max_demo_eps = 2000
    keep_min_ratio = max(0.0, float(forest_demo_filter_min_progress_ratio))
    keep_min_prog_per_step = max(0.0, float(forest_demo_filter_min_progress_per_step_m))
    keep_max_steps = max(0, int(forest_demo_filter_max_steps))
    log(f"[train] Demo collect start: target={int(n)}, expert={expert}, max_eps={int(max_demo_eps)}")
    while added < n and demo_ep < int(max_demo_eps):
        opts = None
        if forest_random_start_goal:
            bucket_idx: int | None = None
            if bool(forest_curriculum):
                min_dist = float(forest_rand_min_dist_m)
                p_demo = float(np.clip(float(demo_ep) / float(max(1, int(max_demo_eps) - 1)), 0.0, 1.0))
                max_dist = _forest_effective_rand_max_dist_m(
                    env,
                    rand_min_dist_m=float(min_dist),
                    rand_max_dist_m=forest_rand_max_dist_m,
                    curriculum_progress=float(p_demo),
                )
            else:
                bucket_idx = choose_bucket()
                diag_m = max(1e-6, float(getattr(env, "_diag_m", 1.0)))
                bmin_ratio = 0.0
                bmax_ratio = 0.0
                if bucket_idx is not None:
                    bmin_ratio, bmax_ratio = bucket_ratio_bounds[int(bucket_idx)]

                min_dist = max(float(forest_rand_min_dist_m), float(bmin_ratio) * float(diag_m))
                hard_max = None if forest_rand_max_dist_m is None else float(forest_rand_max_dist_m)
                bucket_max = None if float(bmax_ratio) >= 1.0 else float(bmax_ratio) * float(diag_m)
                max_dist: float | None
                if hard_max is None and bucket_max is None:
                    max_dist = None
                elif hard_max is None:
                    max_dist = float(bucket_max)
                elif bucket_max is None:
                    max_dist = float(hard_max)
                else:
                    max_dist = float(min(float(hard_max), float(bucket_max)))

                if max_dist is not None and float(max_dist) <= float(min_dist) + 1e-6:
                    max_dist = None

            opts = {
                "random_start_goal": True,
                "rand_min_dist_m": float(min_dist),
                "rand_max_dist_m": max_dist,
                "rand_fixed_prob": float(forest_rand_fixed_prob),
                "rand_tries": int(forest_rand_tries),
                "rand_edge_margin_m": float(forest_rand_edge_margin_m),
            }
        elif forest_curriculum:
            p = float(demo_prog[demo_ep % int(demo_prog.size)])
            opts = {"curriculum_progress": p, "curriculum_band_m": float(curriculum_band_m)}
        obs, _ = env.reset(seed=int(seed) + 50_000 + int(demo_ep), options=opts)
        done = False
        truncated = False
        reached = False
        expert_step_failed = False
        start_goal_dist_m = float(getattr(env, "_distance_to_goal_m")())
        last_info: dict[str, object] = {}
        ep: list[tuple[np.ndarray, int, float, np.ndarray, bool, bool, np.ndarray]] = []
        while not (done or truncated):
            try:
                a = forest_expert_action(
                    env,
                    forest_expert=str(expert),
                    horizon_steps=int(forest_demo_horizon),
                    w_clearance=float(forest_demo_w_clearance),
                    mpc_cfg=mpc_cfg,
                    astar_curve_cfg=astar_curve_cfg,
                    astar_seed=int(astar_seed + demo_ep),
                    astar_replan_attempt=0,
                    astar_timeout_s=float(astar_timeout_s),
                    astar_max_expanded=int(astar_max_expanded),
                    stage="prefill_collect",
                )
            except Exception:
                expert_step_failed = True
                break
            next_obs, reward, done, truncated, info = env.step(int(a))
            last_info = dict(info)
            next_mask = (
                env.admissible_action_mask(
                    horizon_steps=adm_h,
                    min_od_m=float(min_od_m),
                    min_progress_m=min_prog_m,
                    fallback_to_safe=True,
                )
                if use_adm_next_mask
                else full_next_mask
            )
            reached = bool(reached or bool(info.get("reached", False)))
            ep.append((obs, int(a), float(reward), next_obs, bool(done), bool(truncated), next_mask))
            obs = next_obs

        if bool(expert_step_failed):
            demo_ep += 1
            continue

        if bool(reached):
            steps_ep = int(len(ep))
            end_goal_dist_m = float(last_info.get("d_goal_m", 0.0))
            goal_progress_m = max(0.0, float(start_goal_dist_m) - float(end_goal_dist_m))
            progress_ratio = float(goal_progress_m) / float(max(1e-6, float(start_goal_dist_m)))
            progress_per_step_m = float(goal_progress_m) / float(max(1, int(steps_ep)))
            if (keep_max_steps > 0 and int(steps_ep) > int(keep_max_steps)) or (
                float(progress_ratio) < float(keep_min_ratio)
            ) or (float(progress_per_step_m) < float(keep_min_prog_per_step)):
                demo_ep += 1
                continue

            if bool(forest_random_start_goal) and (not bool(forest_curriculum)):
                ratio = _forest_pair_dist_ratio(env)
                if ratio < float(bucket_ratio_bounds[0][1]):
                    bucket_idx = 0
                elif ratio < float(bucket_ratio_bounds[1][1]):
                    bucket_idx = 1
                else:
                    bucket_idx = 2
                if int(bucket_added[bucket_idx]) >= int(bucket_target[bucket_idx]):
                    demo_ep += 1
                    continue

            if int(added + len(ep)) > int(n):
                break
            for o, a, r, no, d, tr, nm in ep:
                obs_buf[added] = o
                next_obs_buf[added] = no
                act_buf[added] = int(a)
                rew_buf[added] = float(r)
                next_mask_buf[added] = nm
                done_buf[added] = 1.0 if bool(d) else 0.0
                trunc_buf[added] = 1.0 if bool(tr) else 0.0
                added += 1
                if added >= n:
                    break
            if bool(forest_random_start_goal) and (not bool(forest_curriculum)):
                bucket_added[int(bucket_idx)] += int(len(ep))
        demo_ep += 1

        now = time.perf_counter()
        should_log = False
        if added >= n or demo_ep >= int(max_demo_eps):
            should_log = True
        elif (demo_ep % 20) == 0:
            should_log = True
        elif (now - t_last_log) >= 5.0 and int(added) != int(added_last_log):
            should_log = True

        if should_log:
            pct = 100.0 * float(added) / float(max(1, n))
            log(
                f"[train] Demo collect progress: {int(added)}/{int(n)} "
                f"({pct:.1f}%), episodes={int(demo_ep)}, elapsed={format_elapsed_s(now - t_collect_start)}"
            )
            t_last_log = now
            added_last_log = int(added)

    t_collect_elapsed = float(time.perf_counter() - t_collect_start)
    rate = float(added) / float(max(t_collect_elapsed, 1e-6))
    if int(added) < int(n):
        log(
            f"[train] Demo collect stopped early: got={int(added)}/{int(n)}, "
            f"episodes={int(demo_ep)}, elapsed={format_elapsed_s(t_collect_elapsed)}, rate={rate:.1f} trans/s"
        )
    else:
        log(
            f"[train] Demo collect done: got={int(added)}/{int(n)}, "
            f"episodes={int(demo_ep)}, elapsed={format_elapsed_s(t_collect_elapsed)}, rate={rate:.1f} trans/s"
        )

    return (
        obs_buf[:added],
        act_buf[:added],
        rew_buf[:added],
        next_obs_buf[:added],
        next_mask_buf[:added],
        done_buf[:added],
        trunc_buf[:added],
    )


def train_one(
    env: gym.Env,
    algo: str,
    *,
    episodes: int,
    seed: int,
    out_dir: Path,
    agent_cfg: AgentConfig,
    replay_near_goal_factor: float,
    replay_hazard_od_m: float,
    train_freq: int,
    learning_starts: int,
    forest_demo_target_mult: float,
    forest_demo_target_cap: int,
    save_ckpt: str,
    forest_curriculum: bool,
    curriculum_band_m: float,
    curriculum_ramp: float,
    forest_demo_prefill: bool,
    forest_demo_pretrain_steps: int,
    forest_demo_pretrain_eval_every: int,
    forest_demo_pretrain_val_runs: int,
    forest_demo_pretrain_early_stop_sr: float,
    forest_demo_pretrain_early_stop_patience: int,
    forest_demo_horizon: int,
    forest_demo_w_clearance: float,
    forest_demo_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
    forest_expert: str,
    forest_expert_exploration: bool,
    forest_action_shield: bool,
    forest_adm_horizon: int,
    forest_topk: int,
    forest_min_od_m: float,
    forest_min_progress_m: float,
    forest_no_fallback: bool,
    forest_expert_prob_start: float,
    forest_expert_prob_final: float,
    forest_expert_prob_decay: float,
    forest_expert_adapt_k: float,
    forest_expert_recent_window: int,
    forest_random_start_goal: bool,
    forest_rand_min_dist_m: float,
    forest_rand_max_dist_m: float | None,
    forest_rand_fixed_prob: float,
    forest_rand_tries: int,
    forest_rand_edge_margin_m: float,
    forest_train_two_suites: bool,
    forest_train_short_prob: float,
    forest_train_short_prob_ramp: float,
    forest_train_dynamic_curriculum: bool,
    forest_train_dynamic_target_sr_short: float,
    forest_train_dynamic_target_sr_long: float,
    forest_train_dynamic_k: float,
    forest_train_dynamic_min_short_prob: float,
    forest_train_dynamic_max_short_prob: float,
    forest_train_short_min_dist_m: float,
    forest_train_short_max_dist_m: float | None,
    forest_train_long_min_dist_m: float,
    forest_train_long_max_dist_m: float | None,
    astar_curve_cfg: AStarCurveOptConfig,
    astar_timeout_s: float,
    astar_max_expanded: int,
    mpc_cfg: MPCConfig,
    eval_every: int,
    eval_runs: int,
    train_eval_every: int,
    train_eval_short_min_dist_m: float,
    train_eval_short_max_dist_m: float | None,
    train_eval_long_min_dist_m: float,
    train_eval_long_max_dist_m: float | None,
    train_eval_seed_base: int,
    train_eval_astar_timeout_s: float,
    train_eval_astar_max_expanded: int,
    rl_early_stop_warmup_episodes: int,
    rl_early_stop_patience_points: int,
    rl_early_stop_min_delta_sr: float,
    rl_early_stop_min_delta_ratio: float,
    throughput_abort_min_episodes: int,
    throughput_abort_max_minutes: float,
    eval_score_time_weight: float,
    save_ckpt_joint_short_long: bool,
    save_ckpt_suite_runs: int,
    save_ckpt_short_min_dist_m: float,
    save_ckpt_short_max_dist_m: float | None,
    save_ckpt_long_min_dist_m: float,
    save_ckpt_long_max_dist_m: float | None,
    save_ckpt_long_sr_floor: float,
    periodic_ckpt_every: int = 0,
    periodic_ckpt_keep: int = 0,
    progress: bool = False,
    device: torch.device,
    live_viewer: TrainLiveViewer | None = None,
    progress_write: Callable[[str], None] | None = None,
) -> tuple[DQNFamilyAgent, np.ndarray, list[dict[str, float | int]], dict[str, object]]:
    if progress_write is None:
        progress_write = make_progress_writer(progress)

    def log(msg: str) -> None:
        if progress_write is not None:
            progress_write(str(msg))

    run_label = f"{env.map_spec.name}/{algo}"
    t_train_one_start = time.perf_counter()
    adm_h = max(1, int(forest_adm_horizon))
    topk_k = max(1, int(forest_topk))
    min_od_m = float(forest_min_od_m)
    min_prog_m = float(forest_min_progress_m)
    strict_no_fallback = bool(forest_no_fallback)
    ckpt_joint_short_long = bool(save_ckpt_joint_short_long) and bool(forest_random_start_goal) and isinstance(env, AMRBicycleEnv)
    ckpt_suite_runs = max(1, int(save_ckpt_suite_runs))

    def _maybe_max_dist(raw_max_dist_m: float | None) -> float | None:
        if raw_max_dist_m is None:
            return None
        if float(raw_max_dist_m) <= 0.0:
            return None
        return float(raw_max_dist_m)

    ckpt_short_max_dist_m = _maybe_max_dist(save_ckpt_short_max_dist_m)
    ckpt_long_max_dist_m = _maybe_max_dist(save_ckpt_long_max_dist_m)
    ckpt_long_sr_floor = float(np.clip(float(save_ckpt_long_sr_floor), 0.0, 1.0))
    progress_short_max_dist_m = _maybe_max_dist(train_eval_short_max_dist_m)
    progress_long_max_dist_m = _maybe_max_dist(train_eval_long_max_dist_m)
    train_two_suites = bool(forest_train_two_suites) and bool(forest_random_start_goal) and isinstance(env, AMRBicycleEnv)
    train_short_prob = float(np.clip(float(forest_train_short_prob), 0.0, 1.0))
    train_short_prob_ramp = float(np.clip(float(forest_train_short_prob_ramp), 0.0, 1.0))
    train_dynamic_curriculum = bool(forest_train_dynamic_curriculum) and bool(train_two_suites)
    train_dynamic_target_sr_short = float(np.clip(float(forest_train_dynamic_target_sr_short), 0.0, 1.0))
    train_dynamic_target_sr_long = float(np.clip(float(forest_train_dynamic_target_sr_long), 0.0, 1.0))
    train_dynamic_k = max(0.0, float(forest_train_dynamic_k))
    train_dynamic_min_short_prob = float(np.clip(float(forest_train_dynamic_min_short_prob), 0.0, 1.0))
    train_dynamic_max_short_prob = float(np.clip(float(forest_train_dynamic_max_short_prob), 0.0, 1.0))
    if float(train_dynamic_max_short_prob) < float(train_dynamic_min_short_prob):
        train_dynamic_min_short_prob, train_dynamic_max_short_prob = train_dynamic_max_short_prob, train_dynamic_min_short_prob
    dynamic_short_prob = float(np.clip(float(train_short_prob), train_dynamic_min_short_prob, train_dynamic_max_short_prob))
    dynamic_last_update_ep = -1
    train_short_max_dist_m = _maybe_max_dist(forest_train_short_max_dist_m)
    train_long_max_dist_m = _maybe_max_dist(forest_train_long_max_dist_m)
    train_eval_short_long_enabled = bool(isinstance(env, AMRBicycleEnv) and bool(forest_random_start_goal) and train_two_suites)
    train_eval_every_eff = max(0, int(train_eval_every))
    early_stop_warmup = max(0, int(rl_early_stop_warmup_episodes))
    early_stop_patience = max(1, int(rl_early_stop_patience_points))
    early_stop_delta_sr = max(0.0, float(rl_early_stop_min_delta_sr))
    early_stop_delta_ratio = max(0.0, float(rl_early_stop_min_delta_ratio))
    throughput_abort_min_eps_eff = max(1, int(throughput_abort_min_episodes))
    throughput_abort_max_s = max(0.0, float(throughput_abort_max_minutes)) * 60.0

    train_progress_pairs_short: list[dict[str, object]] = []
    train_progress_pairs_long: list[dict[str, object]] = []
    if train_eval_short_long_enabled and train_eval_every_eff > 0:
        train_progress_pairs_short, train_progress_pairs_long = _sample_fixed_train_eval_pairs(
            env,
            runs_per_suite=max(1, int(eval_runs)),
            seed_base=int(train_eval_seed_base),
            short_min_dist_m=float(train_eval_short_min_dist_m),
            short_max_dist_m=progress_short_max_dist_m,
            long_min_dist_m=float(train_eval_long_min_dist_m),
            long_max_dist_m=progress_long_max_dist_m,
            rand_tries=int(forest_rand_tries),
            rand_edge_margin_m=float(forest_rand_edge_margin_m),
        )
        log(
            f"[train] [{run_label}] Fixed train-eval suites prepared: "
            f"short={int(len(train_progress_pairs_short))}, long={int(len(train_progress_pairs_long))}, "
            f"every={int(train_eval_every_eff)}"
        )

    if bool(train_dynamic_curriculum):
        log(
            f"[train] [{run_label}] Dynamic two-suite curriculum enabled: "
            f"target_sr_short={float(train_dynamic_target_sr_short):.2f}, "
            f"target_sr_long={float(train_dynamic_target_sr_long):.2f}, "
            f"k={float(train_dynamic_k):.3f}, "
            f"short_prob_range=[{float(train_dynamic_min_short_prob):.2f},{float(train_dynamic_max_short_prob):.2f}], "
            f"init_short_prob={float(dynamic_short_prob):.2f}"
        )

    def random_reset_options_for_training(
        *,
        suite: str | None,
        curriculum_progress: float | None,
    ) -> dict[str, object]:
        if suite == "short":
            min_dist = float(forest_train_short_min_dist_m)
            max_dist = train_short_max_dist_m
            fixed_prob = 0.0
        elif suite == "long":
            min_dist = float(forest_train_long_min_dist_m)
            max_dist = train_long_max_dist_m
            fixed_prob = 0.0
        else:
            min_dist = float(forest_rand_min_dist_m)
            max_dist = _forest_effective_rand_max_dist_m(
                env,
                rand_min_dist_m=float(forest_rand_min_dist_m),
                rand_max_dist_m=forest_rand_max_dist_m,
                curriculum_progress=curriculum_progress,
            )
            fixed_prob = float(forest_rand_fixed_prob)
        return {
            "random_start_goal": True,
            "rand_min_dist_m": float(min_dist),
            "rand_max_dist_m": (0.0 if max_dist is None else float(max_dist)),
            "rand_fixed_prob": float(fixed_prob),
            "rand_tries": int(forest_rand_tries),
            "rand_edge_margin_m": float(forest_rand_edge_margin_m),
        }

    log(
        f"[train] [{run_label}] train_one start: episodes={int(episodes)}, "
        f"learning_starts={int(learning_starts)}, demo_prefill={bool(forest_demo_prefill)}, "
        f"demo_target_mult={float(forest_demo_target_mult):.1f}, demo_target_cap={int(forest_demo_target_cap)}, "
        f"demo_pretrain_steps={int(forest_demo_pretrain_steps)}, forest_adm_horizon={int(adm_h)}, "
        f"forest_topk={int(topk_k)}, forest_min_od_m={float(min_od_m):.3f}, "
        f"forest_min_progress_m={float(min_prog_m):.4f}, strict_no_fallback={bool(strict_no_fallback)}"
    )

    obs_dim = int(env.observation_space.shape[0])
    n_actions = int(env.action_space.n)
    full_action_mask = np.ones((n_actions,), dtype=np.bool_)
    agent = DQNFamilyAgent(algo, obs_dim, n_actions, config=agent_cfg, seed=seed, device=device)

    near_goal_dist_m = 0.0
    hazard_od_m_thr = float(replay_hazard_od_m)
    if isinstance(env, AMRBicycleEnv):
        near_goal_factor = max(0.0, float(replay_near_goal_factor))
        near_goal_dist_m = float(near_goal_factor) * float(env.goal_tolerance_m)
        if hazard_od_m_thr <= 0.0:
            hazard_od_m_thr = max(
                0.0,
                2.0 * float(env.safe_distance_m),
                float(env.safe_speed_distance_m),
                0.4,
            )

    returns = np.zeros((episodes,), dtype=np.float32)
    global_step = 0
    eval_history: list[dict[str, float | int]] = []
    train_progress_history: list[dict[str, float | int | str]] = []

    episodes_completed = 0
    train_stop_reason = "completed"
    throughput_aborted = False
    rl_early_stopped = False
    early_stop_bad_points = 0
    early_stop_best_sr_all = -1.0
    early_stop_best_ratio = float("inf")
    stop_now = False

    best_score: tuple[int, int, int] = (-1, -10**18, 0)
    best_q: dict[str, torch.Tensor] | None = None
    best_q_target: dict[str, torch.Tensor] | None = None
    best_train_steps: int = 0
    pretrain_q: dict[str, torch.Tensor] | None = None
    pretrain_q_target: dict[str, torch.Tensor] | None = None
    pretrain_train_steps: int = 0
    explore_rng = np.random.default_rng(seed + 777)
    recent_fail_history: list[int] = []
    expert_takeovers = 0
    expert_decisions = 0
    inadmissible_count = 0
    action_decisions = 0

    def episode_score(*, reached: bool, collision: bool, steps: int, ret: float) -> tuple[int, int, int]:
        """Prefer reach > survive (timeout) > collision (then higher return)."""
        if bool(reached):
            # Prefer higher return, then fewer steps.
            return (2, int(1_000_000 * float(ret)), -int(steps))
        if bool(collision):
            return (0, int(1_000_000 * float(ret)), -int(steps))
        # Timeout / did not reach.
        return (1, int(1_000_000 * float(ret)), -int(steps))

    def clone_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in sd.items()}

    # Match checkpoint selection and metric logging to the evaluation distribution:
    # - fixed-start training => evaluate on canonical start/goal
    # - random-start/goal training => evaluate on a small fixed batch of sampled (start,goal) pairs
    eval_reset_options_list: list[dict[str, object] | None] = [None]
    if bool(forest_random_start_goal) and isinstance(env, AMRBicycleEnv):
        eval_reset_options_list = []
        n_eval = max(1, int(eval_runs))
        for i in range(n_eval):
            eval_suite = None
            p_eval = None
            if bool(train_two_suites):
                eval_suite = "short" if (int(i) % 2 == 0) else "long"
            elif bool(forest_curriculum):
                p_eval = float(np.clip(float(i) / float(max(1, n_eval - 1)), 0.0, 1.0))
            env.reset(
                seed=int(seed) + 90_000 + int(i),
                options=random_reset_options_for_training(suite=eval_suite, curriculum_progress=p_eval),
            )
            eval_reset_options_list.append(
                {
                    "start_xy": (int(env.start_xy[0]), int(env.start_xy[1])),
                    "goal_xy": (int(env.goal_xy[0]), int(env.goal_xy[1])),
                }
            )

    def sync_cuda() -> None:
        if agent.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

    def forest_expert_action_local() -> int:
        if not isinstance(env, AMRBicycleEnv):
            raise RuntimeError("forest_expert_action called for non-forest env")
        expert = str(forest_expert).lower().strip()
        if expert == "auto":
            expert = "hybrid_astar"
        replan_attempt = max(0, int(global_step // 200))
        return forest_expert_action(
            env,
            forest_expert=str(expert),
            horizon_steps=int(forest_demo_horizon),
            w_clearance=float(forest_demo_w_clearance),
            mpc_cfg=mpc_cfg,
            astar_curve_cfg=astar_curve_cfg,
            astar_seed=int(seed),
            astar_replan_attempt=int(replan_attempt),
            astar_timeout_s=float(astar_timeout_s),
            astar_max_expanded=int(astar_max_expanded),
            stage="train_episode",
        )

    def _forest_q_values(obs_arr: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            x = torch.from_numpy(obs_arr.astype(np.float32, copy=False)).to(agent.device)
            return agent._q_values(agent.q(x.unsqueeze(0))).squeeze(0)

    def _forest_greedy_action(obs_eval: np.ndarray) -> tuple[int, bool]:
        if isinstance(env, AMRBicycleEnv):
            q = _forest_q_values(obs_eval)
            return _forest_policy_action_from_q(
                env,
                q,
                forest_adm_horizon=int(adm_h),
                forest_topk=int(topk_k),
                forest_min_od_m=float(min_od_m),
                forest_min_progress_m=float(min_prog_m),
                forest_no_fallback=bool(strict_no_fallback),
            )
        return int(agent.act(obs_eval, episode=0, explore=False)), False

    # Forest-only: prefill replay buffer with a few short expert rollouts.
    # This is off-policy data (valid for Q-learning) and dramatically reduces the
    # chance of converging to the degenerate "stop until stuck" behavior.
    if forest_demo_prefill and isinstance(env, AMRBicycleEnv) and learning_starts > 0:
        t_prefill_start = time.perf_counter()
        demo_target = forest_demo_target(
            learning_starts=int(learning_starts),
            batch_size=int(agent_cfg.batch_size),
            target_mult=float(forest_demo_target_mult),
            target_cap=int(forest_demo_target_cap),
        )
        log(f"[train] [{run_label}] Demo prefill start: target={int(demo_target)}")
        if forest_demo_data is not None:
            obs_buf, act_buf, rew_buf, next_obs_buf, next_mask_buf, done_buf, trunc_buf = forest_demo_data
            n = int(min(int(demo_target), int(obs_buf.shape[0])))
            log(
                f"[train] [{run_label}] Use cached demos: available={int(obs_buf.shape[0])}, "
                f"load={int(n)}"
            )
            for i in range(n):
                agent.observe(
                    obs_buf[i],
                    int(act_buf[i]),
                    float(rew_buf[i]),
                    next_obs_buf[i],
                    bool(done_buf[i] > 0.5),
                    demo=True,
                    truncated=bool(trunc_buf[i] > 0.5),
                    next_action_mask=next_mask_buf[i],
                )
            global_step += int(n)
            log(
                f"[train] [{run_label}] Demo prefill done (cached): loaded={int(n)}, "
                f"elapsed={format_elapsed_s(time.perf_counter() - t_prefill_start)}"
            )
        else:
            # Collect *more* than just learning_starts transitions: imitation on static global maps
            # is very data-efficient, and preserving a diverse demo set improves robustness when the
            # learned policy slightly deviates from the reference trajectory (DAgger-like effect).
            demo_added = 0
            demo_ep = 0
            demo_prog = np.linspace(0.0, 1.0, num=5, dtype=np.float32)
            max_demo_eps = 2000
            t_last_prefill_log = time.perf_counter()
            demo_added_last_log = -1
            while demo_added < demo_target and demo_ep < int(max_demo_eps):
                opts = None
                if bool(forest_random_start_goal):
                    if bool(train_two_suites):
                        demo_suite = "short" if (int(demo_ep) % 2 == 0) else "long"
                        opts = random_reset_options_for_training(suite=demo_suite, curriculum_progress=None)
                    else:
                        p_demo = None
                        if bool(forest_curriculum):
                            p_demo = float(np.clip(float(demo_ep) / float(max(1, int(max_demo_eps) - 1)), 0.0, 1.0))
                        opts = random_reset_options_for_training(suite=None, curriculum_progress=p_demo)
                elif forest_curriculum:
                    # When curriculum is enabled, diversify demonstration starts to match the
                    # training start-state distribution.
                    p = float(demo_prog[demo_ep % int(demo_prog.size)])
                    opts = {"curriculum_progress": p, "curriculum_band_m": float(curriculum_band_m)}
                obs, _ = env.reset(seed=seed + 50_000 + demo_ep, options=opts)
                done = False
                truncated = False
                reached = False
                expert_step_failed = False
                ep: list[tuple[np.ndarray, int, float, np.ndarray, bool, bool, np.ndarray]] = []
                while not (done or truncated):
                    try:
                        a = forest_expert_action_local()
                    except Exception:
                        expert_step_failed = True
                        break
                    next_obs, reward, done, truncated, info = env.step(a)
                    next_mask = env.admissible_action_mask(
                        horizon_steps=adm_h,
                        min_od_m=float(min_od_m),
                        min_progress_m=min_prog_m,
                        fallback_to_safe=True,
                    )
                    reached = bool(reached or bool(info.get("reached", False)))
                    ep.append((obs, int(a), float(reward), next_obs, bool(done), bool(truncated), next_mask))
                    obs = next_obs
                if bool(expert_step_failed):
                    demo_ep += 1
                    continue
                if bool(reached):
                    for o, a, r, no, d, tr, nm in ep:
                        agent.observe(
                            o,
                            int(a),
                            float(r),
                            no,
                            bool(d),
                            demo=True,
                            truncated=bool(tr),
                            next_action_mask=nm,
                        )
                        demo_added += 1
                        global_step += 1
                    if demo_added >= demo_target:
                        break
                demo_ep += 1

                now = time.perf_counter()
                should_log = False
                if demo_added >= demo_target or demo_ep >= int(max_demo_eps):
                    should_log = True
                elif (demo_ep % 20) == 0:
                    should_log = True
                elif (now - t_last_prefill_log) >= 5.0 and int(demo_added) != int(demo_added_last_log):
                    should_log = True

                if should_log:
                    pct = 100.0 * float(demo_added) / float(max(1, demo_target))
                    log(
                        f"[train] [{run_label}] Demo prefill progress: {int(demo_added)}/{int(demo_target)} "
                        f"({pct:.1f}%), episodes={int(demo_ep)}, "
                        f"elapsed={format_elapsed_s(now - t_prefill_start)}"
                    )
                    t_last_prefill_log = now
                    demo_added_last_log = int(demo_added)

            log(
                f"[train] [{run_label}] Demo prefill done: added={int(demo_added)}/{int(demo_target)}, "
                f"episodes={int(demo_ep)}, elapsed={format_elapsed_s(time.perf_counter() - t_prefill_start)}"
            )

        # Supervised warm-start on demos before TD learning.
        pre_steps = int(max(0, int(forest_demo_pretrain_steps)))
        if pre_steps > 0:
            t_pretrain_start = time.perf_counter()
            done_steps = 0
            chunk = max(1, int(forest_demo_pretrain_eval_every))
            val_runs = max(1, int(forest_demo_pretrain_val_runs))
            sr_thr = float(np.clip(float(forest_demo_pretrain_early_stop_sr), 0.0, 1.0))
            patience = max(1, int(forest_demo_pretrain_early_stop_patience))
            sr_streak = 0
            log(
                f"[train] [{run_label}] Demo pretrain start: steps={int(pre_steps)}, "
                f"chunk={int(chunk)}, val_runs={int(val_runs)}, early_stop_sr={sr_thr:.2f}, "
                f"patience={int(patience)}"
            )

            def pretrain_validation_sr() -> float:
                succ = 0
                for i in range(int(val_runs)):
                    eval_opts = None
                    if bool(forest_random_start_goal) and isinstance(env, AMRBicycleEnv):
                        if eval_reset_options_list:
                            eval_opts = eval_reset_options_list[int(i) % int(len(eval_reset_options_list))]
                    obs_eval, _ = env.reset(seed=int(seed) + 199_999 + int(i), options=eval_opts)
                    done_eval = False
                    trunc_eval = False
                    info_eval: dict[str, object] = {}
                    while not (done_eval or trunc_eval):
                        a_eval, _ = _forest_greedy_action(obs_eval)
                        obs_eval, _r, done_eval, trunc_eval, info_eval = env.step(int(a_eval))
                    if bool(info_eval.get("reached", False)):
                        succ += 1
                return float(succ) / float(max(1, int(val_runs)))

            while done_steps < pre_steps:
                n = min(int(chunk), int(pre_steps - done_steps))
                agent.pretrain_on_demos(steps=int(n))
                done_steps += int(n)

                sr_val = pretrain_validation_sr()
                pct = 100.0 * float(done_steps) / float(max(1, pre_steps))
                log(
                    f"[train] [{run_label}] Demo pretrain progress: {int(done_steps)}/{int(pre_steps)} "
                    f"({pct:.1f}%), val_sr={float(sr_val):.3f}, streak={int(sr_streak)}/{int(patience)}, "
                    f"elapsed={format_elapsed_s(time.perf_counter() - t_pretrain_start)}"
                )
                if float(sr_val) >= float(sr_thr):
                    sr_streak += 1
                    if int(sr_streak) >= int(patience):
                        log(
                            f"[train] [{run_label}] Demo pretrain early-stop: val_sr={float(sr_val):.3f}, "
                            f"streak={int(sr_streak)}"
                        )
                        break
                else:
                    sr_streak = 0

            log(
                f"[train] [{run_label}] Demo pretrain done: steps={int(done_steps)}/{int(pre_steps)}, "
                f"elapsed={format_elapsed_s(time.perf_counter() - t_pretrain_start)}"
            )

            # Sync target net after imitation so subsequent TD updates start from a consistent pair.
            agent.q_target.load_state_dict(agent.q.state_dict())

            # Keep a snapshot of the post-imitation policy: it is often the most reliable
            # goal-reaching policy on static maps, while later TD updates can sometimes drift.
            pretrain_q = clone_state_dict(agent.q.state_dict())
            pretrain_q_target = clone_state_dict(agent.q_target.state_dict())
            pretrain_train_steps = int(agent._train_steps)

        # Start learning immediately once the buffer has useful transitions.
        global_step = max(int(global_step), int(learning_starts))
        log(
            f"[train] [{run_label}] Demo stages done: global_step={int(global_step)}, "
            f"elapsed={format_elapsed_s(time.perf_counter() - t_prefill_start)}"
        )

    def eval_action(obs_eval: np.ndarray) -> int:
        a_eval, _ = _forest_greedy_action(obs_eval)
        return int(a_eval)

    def eval_greedy_metrics() -> dict[str, float]:
        successes = 0
        collisions = 0
        total_ret = 0.0
        total_steps = 0
        times_s: list[float] = []
        succ_path_lens_m: list[float] = []

        def extract_xy_m(info: dict[str, object]) -> tuple[float, float] | None:
            if "pose_m" in info:
                try:
                    x_m, y_m, _ = info["pose_m"]  # type: ignore[misc]
                    return (float(x_m), float(y_m))
                except Exception:
                    return None
            if "agent_xy" in info:
                try:
                    ax, ay = info["agent_xy"]  # type: ignore[misc]
                    if isinstance(env, AMRBicycleEnv):
                        return (float(ax) * float(env.cell_size_m), float(ay) * float(env.cell_size_m))
                    if isinstance(env, AMRGridEnv):
                        return (float(ax) * float(env.cell_size), float(ay) * float(env.cell_size))
                except Exception:
                    return None
            return None

        for i, r_opts in enumerate(eval_reset_options_list):
            obs_eval, info0 = env.reset(seed=int(seed) + 99_999 + int(i), options=r_opts)
            done_eval = False
            trunc_eval = False
            steps_eval = 0
            ret_eval = 0.0
            t_eval = 0.0
            path_len_m = 0.0
            last_xy_m = extract_xy_m(dict(info0))
            last_info_eval: dict[str, object] = {}
            while not (done_eval or trunc_eval):
                steps_eval += 1
                sync_cuda()
                t0 = time.perf_counter()
                a_eval = eval_action(obs_eval)
                sync_cuda()
                t_eval += float(time.perf_counter() - t0)
                obs_eval, r, done_eval, trunc_eval, info_eval = env.step(int(a_eval))
                last_info_eval = dict(info_eval)
                ret_eval += float(r)
                xy_m = extract_xy_m(last_info_eval)
                if last_xy_m is not None and xy_m is not None:
                    path_len_m += float(math.hypot(float(xy_m[0]) - float(last_xy_m[0]), float(xy_m[1]) - float(last_xy_m[1])))
                last_xy_m = xy_m

            reached_eval = bool(last_info_eval.get("reached", False))
            collision_eval = bool(last_info_eval.get("collision", False) or last_info_eval.get("stuck", False))
            if reached_eval:
                successes += 1
                succ_path_lens_m.append(float(path_len_m))
            if collision_eval:
                collisions += 1
            total_ret += float(ret_eval)
            total_steps += int(steps_eval)
            times_s.append(float(t_eval))

        n = max(1, int(len(eval_reset_options_list)))
        sr = float(successes) / float(n)
        avg_path_length = float(np.mean(succ_path_lens_m)) if succ_path_lens_m else float("nan")
        inference_time_s = float(np.mean(times_s)) if times_s else 0.0

        w_t = float(eval_score_time_weight)
        base = float(avg_path_length) + float(w_t) * float(inference_time_s)
        denom = max(float(sr), 1e-6)
        planning_cost = float(base) / float(denom)
        if not (sr > 0.0 and math.isfinite(base)):
            planning_cost = float("inf")

        return {
            "success_rate": float(successes) / float(n),
            "collision_rate": float(collisions) / float(n),
            "avg_return": float(total_ret) / float(n),
            "avg_steps": float(total_steps) / float(n),
            "avg_path_length": float(avg_path_length),
            "inference_time_s": float(inference_time_s),
            "planning_cost": float(planning_cost),
        }

    pbar = None
    if progress:
        try:
            from tqdm import tqdm  # type: ignore
        except Exception:
            pbar = None
        else:
            pbar = tqdm(
                range(episodes),
                desc=f"Train {env.map_spec.name} {algo}",
                unit="ep",
                dynamic_ncols=True,
                leave=True,
            )

    t_rl_train_start = time.perf_counter()
    if pbar is None:
        log(f"[train] [{run_label}] RL training start: episodes={int(episodes)}")

    train_progress_t0 = time.perf_counter()

    def should_eval_train_progress(ep_idx: int) -> bool:
        if not bool(train_eval_short_long_enabled):
            return False
        if int(train_eval_every_eff) <= 0:
            return False
        ep1 = int(ep_idx) + 1
        return bool((ep1 % int(train_eval_every_eff) == 0) or ep_idx == 0 or ep_idx == int(episodes - 1))

    def maybe_abort_on_throughput(ep_idx: int) -> bool:
        nonlocal throughput_aborted, train_stop_reason, stop_now
        if float(throughput_abort_max_s) <= 0.0:
            return False
        ep1 = int(ep_idx) + 1
        if int(ep1) >= int(throughput_abort_min_eps_eff):
            return False
        elapsed_s = float(time.perf_counter() - t_rl_train_start)
        if float(elapsed_s) <= float(throughput_abort_max_s):
            return False
        throughput_aborted = True
        train_stop_reason = "training_throughput_abnormal"
        stop_now = True
        log(
            f"[train] [{run_label}] Throughput fuse triggered: ep={int(ep1)} < {int(throughput_abort_min_eps_eff)}, "
            f"elapsed={format_elapsed_s(elapsed_s)} > {format_elapsed_s(throughput_abort_max_s)}"
        )
        return True

    def maybe_record_train_progress(ep_idx: int) -> None:
        nonlocal early_stop_bad_points, early_stop_best_sr_all, early_stop_best_ratio
        nonlocal rl_early_stopped, train_stop_reason, stop_now
        nonlocal dynamic_short_prob, dynamic_last_update_ep

        if not bool(should_eval_train_progress(ep_idx)):
            return

        eval_max_steps = int(getattr(env, "max_steps", 0)) if hasattr(env, "max_steps") else 0
        if int(eval_max_steps) <= 0:
            eval_max_steps = 1200

        row = _eval_train_progress_suites(
            env,
            agent=agent,
            pairs_short=train_progress_pairs_short,
            pairs_long=train_progress_pairs_long,
            seed_base=int(train_eval_seed_base),
            max_steps=int(eval_max_steps),
            adm_horizon=int(adm_h),
            topk=int(topk_k),
            min_od_m=float(min_od_m),
            min_progress_m=float(min_prog_m),
            forest_no_fallback=bool(strict_no_fallback),
            astar_timeout_s=float(train_eval_astar_timeout_s),
            astar_max_expanded=int(train_eval_astar_max_expanded),
        )
        wall_clock_s = float(time.perf_counter() - train_progress_t0)
        row_out: dict[str, float | int | str] = {
            "episode": int(ep_idx + 1),
            "wall_clock_s": float(wall_clock_s),
            "sr_short": float(row["sr_short"]),
            "sr_long": float(row["sr_long"]),
            "sr_all": float(row["sr_all"]),
            "ratio_short": float(row["ratio_short"]),
            "ratio_long": float(row["ratio_long"]),
            "n_valid_short": int(row["n_valid_short"]),
            "n_valid_long": int(row["n_valid_long"]),
            "argmax_inadmissible_rate_short": float(row["argmax_inadmissible_rate_short"]),
            "argmax_inadmissible_rate_long": float(row["argmax_inadmissible_rate_long"]),
            "argmax_inadmissible_rate_all": float(row["argmax_inadmissible_rate_all"]),
            "phase": "train",
        }
        train_progress_history.append(row_out)

        sr_all = float(row_out["sr_all"])
        ratio_cur = _train_progress_primary_ratio({
            "ratio_short": row_out["ratio_short"],
            "ratio_long": row_out["ratio_long"],
            "n_valid_short": row_out["n_valid_short"],
            "n_valid_long": row_out["n_valid_long"],
        })
        log(
            f"[train] [{run_label}] Train-progress ep={int(ep_idx + 1)}: "
            f"sr(short/long/all)={float(row_out['sr_short']):.3f}/{float(row_out['sr_long']):.3f}/{sr_all:.3f}, "
            f"ratio(short/long)={float(row_out['ratio_short']):.3f}/{float(row_out['ratio_long']):.3f}, "
            f"argmax_inad(short/long/all)={float(row_out['argmax_inadmissible_rate_short']):.3f}/"
            f"{float(row_out['argmax_inadmissible_rate_long']):.3f}/"
            f"{float(row_out['argmax_inadmissible_rate_all']):.3f}, "
            f"n_valid={int(row_out['n_valid_short'])}/{int(row_out['n_valid_long'])}"
        )

        if bool(train_dynamic_curriculum):
            short_gap = float(train_dynamic_target_sr_short) - float(row_out["sr_short"])
            long_gap = float(train_dynamic_target_sr_long) - float(row_out["sr_long"])
            dynamic_err = float(short_gap - long_gap)
            if abs(float(dynamic_err)) <= 1e-9:
                mean_target = 0.5 * (float(train_dynamic_target_sr_short) + float(train_dynamic_target_sr_long))
                sr_all_cur = float(row_out["sr_all"])
                if float(sr_all_cur) < float(mean_target):
                    # When both suites are equally below target, bias toward long-suite recovery.
                    dynamic_err = -0.25 * float(mean_target - sr_all_cur)
            dynamic_short_prob = float(
                np.clip(
                    float(dynamic_short_prob) + float(train_dynamic_k) * float(dynamic_err),
                    float(train_dynamic_min_short_prob),
                    float(train_dynamic_max_short_prob),
                )
            )
            dynamic_last_update_ep = int(ep_idx) + 1
            log(
                f"[train] [{run_label}] Dynamic curriculum update: "
                f"short_prob={float(dynamic_short_prob):.3f}, err={float(dynamic_err):+.3f}, "
                f"target(short/long)={float(train_dynamic_target_sr_short):.2f}/{float(train_dynamic_target_sr_long):.2f}"
            )

        ep1 = int(ep_idx) + 1
        if int(ep1) < int(early_stop_warmup):
            return

        improved = False
        if sr_all > float(early_stop_best_sr_all) + float(early_stop_delta_sr):
            improved = True
        elif math.isfinite(ratio_cur) and (
            (not math.isfinite(float(early_stop_best_ratio)))
            or (ratio_cur < float(early_stop_best_ratio) - float(early_stop_delta_ratio))
        ):
            improved = True

        if improved:
            early_stop_best_sr_all = max(float(early_stop_best_sr_all), float(sr_all))
            if math.isfinite(ratio_cur):
                early_stop_best_ratio = min(float(early_stop_best_ratio), float(ratio_cur))
            early_stop_bad_points = 0
            return

        early_stop_bad_points += 1
        if int(early_stop_bad_points) >= int(early_stop_patience):
            rl_early_stopped = True
            train_stop_reason = "rl_early_stop_plateau"
            stop_now = True
            log(
                f"[train] [{run_label}] RL early-stop triggered: "
                f"patience={int(early_stop_patience)}, warmup={int(early_stop_warmup)}, "
                f"best_sr_all={float(early_stop_best_sr_all):.3f}, best_ratio={float(early_stop_best_ratio):.3f}"
            )

    ep_iter = pbar if pbar is not None else range(episodes)
    for ep in ep_iter:
        reset_options = None
        if bool(forest_random_start_goal) and isinstance(env, AMRBicycleEnv):
            if bool(train_two_suites):
                if bool(train_dynamic_curriculum) and int(dynamic_last_update_ep) > 0:
                    p_short = float(dynamic_short_prob)
                else:
                    p_short = float(train_short_prob)
                    ramp = float(train_short_prob_ramp)
                    if float(ramp) > 1e-9 and int(episodes) > 1:
                        ramp_episodes = max(1.0, float(ramp) * float(max(1, int(episodes) - 1)))
                        mix = float(np.clip(float(ep) / float(ramp_episodes), 0.0, 1.0))
                        p_short = float((1.0 - float(mix)) * 1.0 + float(mix) * float(train_short_prob))
                ep_suite = "short" if float(explore_rng.random()) < float(np.clip(p_short, 0.0, 1.0)) else "long"
                reset_options = random_reset_options_for_training(suite=ep_suite, curriculum_progress=None)
            else:
                p_curriculum = None
                if bool(forest_curriculum):
                    p_raw = float(ep) / float(max(1, episodes - 1))
                    ramp = max(1e-6, float(curriculum_ramp))
                    p_curriculum = float(np.clip(p_raw / ramp, 0.0, 1.0))
                reset_options = random_reset_options_for_training(suite=None, curriculum_progress=p_curriculum)
        elif forest_curriculum and isinstance(env, AMRBicycleEnv):
            p_raw = float(ep) / float(max(1, episodes - 1))
            ramp = max(1e-6, float(curriculum_ramp))
            p = float(np.clip(p_raw / ramp, 0.0, 1.0))
            reset_options = {"curriculum_progress": p, "curriculum_band_m": float(curriculum_band_m)}
        obs, reset_info = env.reset(seed=seed + ep, options=reset_options)
        if live_viewer is not None and isinstance(env, AMRBicycleEnv):
            live_viewer.start_episode(
                env=env,
                env_name=str(env.map_spec.name),
                algo=str(algo),
                episode=int(ep + 1),
                total_episodes=int(episodes),
                info=reset_info,
            )
        action_mask = None
        use_masked_policy = isinstance(env, AMRBicycleEnv) and (bool(forest_action_shield) or bool(strict_no_fallback))
        if use_masked_policy:
            # Strict no-fallback mode still allows hard admissibility masking as the legal
            # action set (no heuristic replacement). Non-strict mode keeps the previous
            # fallback-to-safe behavior.
            mask0 = env.admissible_action_mask(
                horizon_steps=adm_h,
                min_od_m=float(min_od_m),
                min_progress_m=min_prog_m,
                fallback_to_safe=True,
            )
            if bool(mask0.any()):
                action_mask = mask0
            elif not bool(strict_no_fallback):
                action_mask = mask0
        ep_return = 0.0
        done = False
        truncated = False
        ep_steps = 0
        last_info: dict[str, object] = {}
        ep_buffer: list[tuple[np.ndarray, int, float, np.ndarray, bool, bool, bool, np.ndarray | None, int]] = []
        pending_updates = 0

        while not (done or truncated):
            ep_steps += 1
            global_step += 1
            # Forest stabilizer: mix in an expert as the *behavior* policy early in training.
            # Off-policy Q-learning remains valid, while successful trajectories become frequent
            # enough for bootstrapping long-horizon returns.
            used_expert = False
            if forest_expert_exploration and isinstance(env, AMRBicycleEnv) and (not bool(strict_no_fallback)):
                ramp = max(1e-6, float(forest_expert_prob_decay))
                t = float(np.clip((float(ep) / float(max(1, episodes - 1))) / ramp, 0.0, 1.0))
                p_linear = float(forest_expert_prob_start) + (float(forest_expert_prob_final) - float(forest_expert_prob_start)) * t
                window = max(1, int(forest_expert_recent_window))
                fails = recent_fail_history[-window:]
                recent_fail_rate = float(np.mean(fails)) if fails else 0.5
                p_exp = float(p_linear) + float(forest_expert_adapt_k) * (float(recent_fail_rate) - 0.5)
                p_low = min(float(forest_expert_prob_final), float(forest_expert_prob_start))
                p_high = max(float(forest_expert_prob_final), float(forest_expert_prob_start))
                p_exp = float(np.clip(float(p_exp), float(p_low), float(p_high)))
                expert_decisions += 1
                if explore_rng.random() < p_exp:
                    try:
                        action = forest_expert_action_local()
                        used_expert = True
                        expert_takeovers += 1
                    except Exception:
                        if action_mask is not None:
                            action = agent.act_masked(obs, episode=ep, explore=True, action_mask=action_mask)
                        else:
                            action = int(env._fallback_action_short_rollout(horizon_steps=adm_h, min_od_m=float(min_od_m)))
                else:
                    if action_mask is not None:
                        action = agent.act_masked(obs, episode=ep, explore=True, action_mask=action_mask)
                    else:
                        action = agent.act(obs, episode=ep, explore=True)
            else:
                if action_mask is not None:
                    action = agent.act_masked(obs, episode=ep, explore=True, action_mask=action_mask)
                else:
                    action = agent.act(obs, episode=ep, explore=True)

            if isinstance(env, AMRBicycleEnv) and (not bool(used_expert)):
                if not bool(strict_no_fallback):
                    reached_pose = bool(
                        env._goal_pose_reached(
                            d_goal_m=float(env._distance_to_goal_m()),
                            alpha_rad=float(env._goal_relative_angle_rad()),
                        )
                    )
                    reached_stop = bool(env._goal_stop_reached(v_m_s=float(env._v_m_s), delta_rad=float(env._delta_rad)))
                    if bool(reached_pose) and (not bool(reached_stop)):
                        action = int(forest_stop_action(env))

                inadmissible = not bool(
                    env.is_action_admissible(
                        int(action),
                        horizon_steps=int(adm_h),
                        min_od_m=float(min_od_m),
                        min_progress_m=float(min_prog_m),
                    )
                )
                if bool(inadmissible):
                    inadmissible_count += 1
                    if not bool(strict_no_fallback):
                        q = _forest_q_values(obs)
                        action, _ = _forest_policy_action_from_q(
                            env,
                            q,
                            forest_adm_horizon=int(adm_h),
                            forest_topk=int(topk_k),
                            forest_min_od_m=float(min_od_m),
                            forest_min_progress_m=float(min_prog_m),
                            forest_no_fallback=bool(strict_no_fallback),
                        )
            action_decisions += 1
            next_obs, reward, done, truncated, info = env.step(action)
            last_info = dict(info)
            if live_viewer is not None and isinstance(env, AMRBicycleEnv):
                live_viewer.update_step(
                    env=env,
                    env_name=str(env.map_spec.name),
                    algo=str(algo),
                    episode=int(ep + 1),
                    total_episodes=int(episodes),
                    step=int(ep_steps),
                    action=int(action),
                    reward=float(reward),
                    done=bool(done),
                    truncated=bool(truncated),
                    info=info,
                )
            next_mask = None
            if isinstance(env, AMRBicycleEnv):
                if bool(strict_no_fallback):
                    next_mask_raw = env.admissible_action_mask(
                        horizon_steps=adm_h,
                        min_od_m=float(min_od_m),
                        min_progress_m=min_prog_m,
                        fallback_to_safe=True,
                    )
                    next_mask = next_mask_raw if bool(next_mask_raw.any()) else full_action_mask
                    action_mask = next_mask_raw if bool(next_mask_raw.any()) else None
                else:
                    next_mask = env.admissible_action_mask(
                        horizon_steps=adm_h,
                        min_od_m=float(min_od_m),
                        min_progress_m=min_prog_m,
                        fallback_to_safe=True,
                    )
                    if action_mask is not None:
                        action_mask = next_mask

            replay_flags = 0
            if isinstance(env, AMRBicycleEnv) and bool(getattr(agent_cfg, "replay_stratified", False)):
                d_goal_m = float(last_info.get("d_goal_m", float("inf")))
                if math.isfinite(d_goal_m) and (float(d_goal_m) <= float(near_goal_dist_m)):
                    replay_flags |= int(FLAG_NEAR_GOAL)

                if bool(last_info.get("stuck", False)):
                    replay_flags |= int(FLAG_STUCK)

                od_m = float(last_info.get("od_m", float("inf")))
                if math.isfinite(od_m) and (od_m >= 0.0) and (od_m <= float(hazard_od_m_thr)):
                    replay_flags |= int(FLAG_HAZARD)
                if bool(last_info.get("collision", False)):
                    replay_flags |= int(FLAG_HAZARD)
            # Time-limit truncation should not be treated as terminal for bootstrapping.
            # Only mark expert transitions as demos when the *episode* reaches the goal.
            # Failed expert steps are still useful off-policy data, but should not be imitated/preserved.
            ep_buffer.append(
                (
                    obs,
                    int(action),
                    float(reward),
                    next_obs,
                    bool(done),
                    bool(truncated),
                    bool(used_expert),
                    next_mask,
                    int(replay_flags),
                )
            )
            ep_return += float(reward)

            if global_step >= learning_starts and (global_step % max(1, train_freq) == 0):
                pending_updates += 1

            obs = next_obs

        reached_ep = bool(last_info.get("reached", False))
        recent_fail_history.append(0 if reached_ep else 1)
        keep_n = max(8, int(forest_expert_recent_window) * 3)
        if len(recent_fail_history) > keep_n:
            recent_fail_history = recent_fail_history[-keep_n:]
        for o, a, r, no, d, tr, ue, nm, rf in ep_buffer:
            agent.observe(
                o,
                int(a),
                float(r),
                no,
                bool(d),
                demo=bool(ue and reached_ep),
                truncated=bool(tr),
                next_action_mask=nm,
                replay_flags=int(rf),
            )
        for _ in range(int(pending_updates)):
            agent.update()

        returns[ep] = float(ep_return)
        episodes_completed = int(ep + 1)

        every = int(max(0, int(eval_every)))
        if every > 0 and ((ep + 1) % every == 0 or ep == 0 or ep == int(episodes - 1)):
            m = eval_greedy_metrics()
            eval_history.append(
                {
                    "episode": int(ep + 1),
                    "phase": "eval",
                    "success_rate": float(m["success_rate"]),
                    "collision_rate": float(m["collision_rate"]),
                    "avg_return": float(m["avg_return"]),
                    "avg_steps": float(m["avg_steps"]),
                    "avg_path_length": float(m["avg_path_length"]),
                    "inference_time_s": float(m["inference_time_s"]),
                    "planning_cost": float(m["planning_cost"]),
                    "expert_takeover_rate": float(expert_takeovers) / float(max(1, expert_decisions)),
                    "inadmissible_rate": float(inadmissible_count) / float(max(1, action_decisions)),
                }
            )

        if pbar is not None:
            pbar.set_postfix(
                {
                    "ret": f"{ep_return:.3f}",
                    "eps": f"{agent.epsilon(ep):.3f}",
                    "steps": global_step,
                    "updates": int(agent._train_steps),
                },
                refresh=False,
            )

        if (ep + 1) % max(1, int(episodes // 20), 10) == 0 or ep == 0 or ep == int(episodes - 1):
            pct = 100.0 * float(ep + 1) / float(max(1, episodes))
            log(
                f"[train] [{run_label}] RL progress: ep={int(ep + 1)}/{int(episodes)} "
                f"({pct:.1f}%), last_ret={float(ep_return):.3f}, train_steps={int(agent._train_steps)}, "
                f"elapsed={format_elapsed_s(time.perf_counter() - t_rl_train_start)}"
            )

        reached = bool(last_info.get("reached", False))
        collision = bool(last_info.get("collision", False) or last_info.get("stuck", False))
        score = episode_score(reached=reached, collision=collision, steps=ep_steps, ret=ep_return)
        if score > best_score:
            best_score = score
            best_q = clone_state_dict(agent.q.state_dict())
            best_q_target = clone_state_dict(agent.q_target.state_dict())
            best_train_steps = int(agent._train_steps)

        if int(periodic_ckpt_every) > 0 and (ep + 1) % int(periodic_ckpt_every) == 0:
            _pdir = out_dir / "models" / env.map_spec.name / "periodic"
            _pdir.mkdir(parents=True, exist_ok=True)
            _ppath = _pdir / f"ep{ep + 1:05d}.pt"
            agent.save(_ppath)
            log(f"[train] [{run_label}] periodic ckpt: ep={ep + 1} -> {_ppath}")
            if int(periodic_ckpt_keep) > 0:
                _existing = sorted(_pdir.glob("ep*.pt"))
                for _old in _existing[: -int(periodic_ckpt_keep)]:
                    _old.unlink(missing_ok=True)

        if maybe_abort_on_throughput(ep):
            break

        maybe_record_train_progress(ep)
        if bool(stop_now):
            break

    if pbar is not None:
        pbar.close()
    log(
        f"[train] [{run_label}] RL training done: episodes={int(episodes_completed)}/{int(episodes)}, "
        f"stop_reason={str(train_stop_reason)}, "
        f"elapsed={format_elapsed_s(time.perf_counter() - t_rl_train_start)}, "
        f"train_steps={int(agent._train_steps)}"
    )

    final_q = clone_state_dict(agent.q.state_dict())
    final_q_target = clone_state_dict(agent.q_target.state_dict())
    final_train_steps = int(agent._train_steps)

    def sample_joint_suite_reset_options(*, min_dist_m: float, max_dist_m: float | None, seed_offset: int) -> list[dict[str, object]]:
        if not isinstance(env, AMRBicycleEnv):
            return []

        options_list: list[dict[str, object]] = []
        max_dist_opt = 0.0 if max_dist_m is None else float(max_dist_m)
        for i in range(int(ckpt_suite_runs)):
            env.reset(
                seed=int(seed) + int(seed_offset) + int(i),
                options={
                    "random_start_goal": True,
                    "rand_min_dist_m": float(min_dist_m),
                    "rand_max_dist_m": float(max_dist_opt),
                    "rand_fixed_prob": float(forest_rand_fixed_prob),
                    "rand_tries": int(forest_rand_tries),
                    "rand_edge_margin_m": float(forest_rand_edge_margin_m),
                },
            )
            options_list.append(
                {
                    "start_xy": (int(env.start_xy[0]), int(env.start_xy[1])),
                    "goal_xy": (int(env.goal_xy[0]), int(env.goal_xy[1])),
                }
            )
        return options_list

    short_suite_reset_options: list[dict[str, object]] = []
    long_suite_reset_options: list[dict[str, object]] = []
    if bool(ckpt_joint_short_long):
        short_suite_reset_options = sample_joint_suite_reset_options(
            min_dist_m=float(save_ckpt_short_min_dist_m),
            max_dist_m=ckpt_short_max_dist_m,
            seed_offset=120_000,
        )
        long_suite_reset_options = sample_joint_suite_reset_options(
            min_dist_m=float(save_ckpt_long_min_dist_m),
            max_dist_m=ckpt_long_max_dist_m,
            seed_offset=140_000,
        )

    def eval_greedy(q_sd: dict[str, torch.Tensor], q_target_sd: dict[str, torch.Tensor]) -> tuple[int, int, int]:
        agent.q.load_state_dict(q_sd)
        agent.q_target.load_state_dict(q_target_sd)

        # Canonical (single) evaluation.
        if not (bool(forest_random_start_goal) and isinstance(env, AMRBicycleEnv)):
            obs, _ = env.reset(seed=seed + 9999)
            done = False
            truncated = False
            steps = 0
            ret = 0.0
            last_info: dict[str, object] = {}
            while not (done or truncated):
                steps += 1
                a = eval_action(obs)
                obs, r, done, truncated, info = env.step(a)
                last_info = dict(info)
                ret += float(r)

            reached = bool(last_info.get("reached", False))
            collision = bool(last_info.get("collision", False) or last_info.get("stuck", False))
            return episode_score(reached=reached, collision=collision, steps=steps, ret=ret)

        # Random-start/goal evaluation: use a fixed batch of sampled (start,goal) pairs.
        successes = 0
        total_ret = 0.0
        total_steps = 0
        for i, r_opts in enumerate(eval_reset_options_list):
            obs, _ = env.reset(seed=seed + 9999 + int(i), options=r_opts)
            done = False
            truncated = False
            steps = 0
            ret = 0.0
            last_info: dict[str, object] = {}
            while not (done or truncated):
                steps += 1
                a = eval_action(obs)
                obs, r, done, truncated, info = env.step(a)
                last_info = dict(info)
                ret += float(r)
            if bool(last_info.get("reached", False)):
                successes += 1
            total_ret += float(ret)
            total_steps += int(steps)

        n = max(1, int(len(eval_reset_options_list)))
        avg_ret = float(total_ret) / float(n)
        avg_steps = float(total_steps) / float(n)
        return (int(successes), int(1_000_000 * float(avg_ret)), -int(avg_steps))

    def eval_joint_suite_metrics(
        q_sd: dict[str, torch.Tensor],
        q_target_sd: dict[str, torch.Tensor],
        *,
        reset_options_list: list[dict[str, object]],
        seed_offset: int,
    ) -> dict[str, float | int]:
        agent.q.load_state_dict(q_sd)
        agent.q_target.load_state_dict(q_target_sd)

        successes = 0
        times_s: list[float] = []
        succ_path_lens_m: list[float] = []

        def extract_xy_m(info: dict[str, object]) -> tuple[float, float] | None:
            if "pose_m" in info:
                try:
                    x_m, y_m, _ = info["pose_m"]  # type: ignore[misc]
                    return (float(x_m), float(y_m))
                except Exception:
                    return None
            if "agent_xy" in info:
                try:
                    ax, ay = info["agent_xy"]  # type: ignore[misc]
                    if isinstance(env, AMRBicycleEnv):
                        return (float(ax) * float(env.cell_size_m), float(ay) * float(env.cell_size_m))
                    if isinstance(env, AMRGridEnv):
                        return (float(ax) * float(env.cell_size), float(ay) * float(env.cell_size))
                except Exception:
                    return None
            return None

        for i, r_opts in enumerate(reset_options_list):
            obs_eval, info0 = env.reset(seed=int(seed) + int(seed_offset) + int(i), options=r_opts)
            done_eval = False
            trunc_eval = False
            t_eval = 0.0
            path_len_m = 0.0
            last_xy_m = extract_xy_m(dict(info0))
            last_info_eval: dict[str, object] = {}

            while not (done_eval or trunc_eval):
                sync_cuda()
                t0 = time.perf_counter()
                a_eval = eval_action(obs_eval)
                sync_cuda()
                t_eval += float(time.perf_counter() - t0)

                obs_eval, _, done_eval, trunc_eval, info_eval = env.step(int(a_eval))
                last_info_eval = dict(info_eval)
                xy_m = extract_xy_m(last_info_eval)
                if last_xy_m is not None and xy_m is not None:
                    path_len_m += float(math.hypot(float(xy_m[0]) - float(last_xy_m[0]), float(xy_m[1]) - float(last_xy_m[1])))
                last_xy_m = xy_m

            if bool(last_info_eval.get("reached", False)):
                successes += 1
                succ_path_lens_m.append(float(path_len_m))
            times_s.append(float(t_eval))

        n = max(1, int(len(reset_options_list)))
        sr = float(successes) / float(n)
        avg_path_length = float(np.mean(succ_path_lens_m)) if succ_path_lens_m else float("nan")
        inference_time_s = float(np.mean(times_s)) if times_s else 0.0

        w_t = float(eval_score_time_weight)
        base = float(avg_path_length) + float(w_t) * float(inference_time_s)
        denom = max(float(sr), 1e-6)
        planning_cost = float(base) / float(denom)
        if not (sr > 0.0 and math.isfinite(base)):
            planning_cost = float("inf")

        return {
            "successes": int(successes),
            "runs": int(n),
            "success_rate": float(sr),
            "planning_cost": float(planning_cost),
        }

    def eval_greedy_joint_short_long(q_sd: dict[str, torch.Tensor], q_target_sd: dict[str, torch.Tensor]) -> dict[str, float | int]:
        short_metrics = eval_joint_suite_metrics(
            q_sd,
            q_target_sd,
            reset_options_list=short_suite_reset_options,
            seed_offset=220_000,
        )
        long_metrics = eval_joint_suite_metrics(
            q_sd,
            q_target_sd,
            reset_options_list=long_suite_reset_options,
            seed_offset=260_000,
        )

        short_succ = int(short_metrics["successes"])
        long_succ = int(long_metrics["successes"])
        short_cost = float(short_metrics["planning_cost"])
        long_cost = float(long_metrics["planning_cost"])
        joint_cost = float(short_cost + long_cost) if math.isfinite(short_cost) and math.isfinite(long_cost) else float("inf")

        return {
            "short_successes": int(short_succ),
            "long_successes": int(long_succ),
            "short_runs": int(short_metrics["runs"]),
            "long_runs": int(long_metrics["runs"]),
            "short_success_rate": float(short_metrics["success_rate"]),
            "long_success_rate": float(long_metrics["success_rate"]),
            "short_planning_cost": float(short_cost),
            "long_planning_cost": float(long_cost),
            "joint_successes": int(short_succ + long_succ),
            "joint_min_successes": int(min(short_succ, long_succ)),
            "joint_planning_cost": float(joint_cost),
        }

    def is_better_joint_score(candidate: dict[str, float | int], current: dict[str, float | int]) -> bool:
        cand_long_sr = float(candidate.get("long_success_rate", 0.0))
        curr_long_sr = float(current.get("long_success_rate", 0.0))
        cand_pass = bool(cand_long_sr >= float(ckpt_long_sr_floor))
        curr_pass = bool(curr_long_sr >= float(ckpt_long_sr_floor))
        if cand_pass != curr_pass:
            return bool(cand_pass)

        candidate_key = (
            int(candidate["joint_successes"]),
            int(candidate["joint_min_successes"]),
        )
        current_key = (
            int(current["joint_successes"]),
            int(current["joint_min_successes"]),
        )
        if candidate_key != current_key:
            return bool(candidate_key > current_key)

        cand_cost = float(candidate["joint_planning_cost"])
        curr_cost = float(current["joint_planning_cost"])
        if math.isfinite(cand_cost) and not math.isfinite(curr_cost):
            return True
        if not math.isfinite(cand_cost) and math.isfinite(curr_cost):
            return False
        if math.isfinite(cand_cost) and math.isfinite(curr_cost):
            return bool(cand_cost < curr_cost)
        return False

    # Choose between the final policy and the best (exploratory) episode checkpoint based on greedy performance.
    def select_checkpoint() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], int, str]:
        mode = str(save_ckpt).lower().strip()
        if mode == "final":
            return final_q, final_q_target, final_train_steps, "final"
        if mode == "best":
            if best_q is not None and best_q_target is not None:
                return best_q, best_q_target, best_train_steps, "best"
            return final_q, final_q_target, final_train_steps, "final"
        if mode == "pretrain":
            if pretrain_q is not None and pretrain_q_target is not None:
                return pretrain_q, pretrain_q_target, pretrain_train_steps, "pretrain"
            return final_q, final_q_target, final_train_steps, "final"
        if mode not in {"auto", ""}:
            raise ValueError("--save-ckpt must be one of: auto, final, best, pretrain")

        if bool(ckpt_joint_short_long):
            best_joint_score = eval_greedy_joint_short_long(final_q, final_q_target)
            chosen_q, chosen_q_target, chosen_train_steps, chosen_src = final_q, final_q_target, final_train_steps, "final"

            if best_q is not None and best_q_target is not None:
                candidate_score = eval_greedy_joint_short_long(best_q, best_q_target)
                if is_better_joint_score(candidate_score, best_joint_score):
                    best_joint_score = candidate_score
                    chosen_q, chosen_q_target, chosen_train_steps, chosen_src = best_q, best_q_target, best_train_steps, "best"

            if pretrain_q is not None and pretrain_q_target is not None:
                candidate_score = eval_greedy_joint_short_long(pretrain_q, pretrain_q_target)
                if is_better_joint_score(candidate_score, best_joint_score):
                    best_joint_score = candidate_score
                    chosen_q, chosen_q_target, chosen_train_steps, chosen_src = (
                        pretrain_q,
                        pretrain_q_target,
                        pretrain_train_steps,
                        "pretrain",
                    )

            log(
                "[train] "
                f"[{run_label}] save_ckpt joint short/long: chosen={chosen_src}, "
                f"short={float(best_joint_score['short_success_rate']):.3f} ({int(best_joint_score['short_successes'])}/{int(best_joint_score['short_runs'])}), "
                f"long={float(best_joint_score['long_success_rate']):.3f} ({int(best_joint_score['long_successes'])}/{int(best_joint_score['long_runs'])}), "
                f"long_sr_floor={float(ckpt_long_sr_floor):.3f}, "
                f"joint_cost={float(best_joint_score['joint_planning_cost']):.3f}"
            )
            return chosen_q, chosen_q_target, chosen_train_steps, chosen_src

        best_greedy_score = eval_greedy(final_q, final_q_target)
        chosen_q, chosen_q_target, chosen_train_steps, chosen_src = final_q, final_q_target, final_train_steps, "final"

        if best_q is not None and best_q_target is not None:
            candidate_score = eval_greedy(best_q, best_q_target)
            if candidate_score > best_greedy_score:
                best_greedy_score = candidate_score
                chosen_q, chosen_q_target, chosen_train_steps, chosen_src = best_q, best_q_target, best_train_steps, "best"

        if pretrain_q is not None and pretrain_q_target is not None:
            candidate_score = eval_greedy(pretrain_q, pretrain_q_target)
            if candidate_score > best_greedy_score:
                chosen_q, chosen_q_target, chosen_train_steps, chosen_src = (
                    pretrain_q,
                    pretrain_q_target,
                    pretrain_train_steps,
                    "pretrain",
                )

        return chosen_q, chosen_q_target, chosen_train_steps, chosen_src

    chosen_q, chosen_q_target, chosen_train_steps, chosen_src = select_checkpoint()

    agent.q.load_state_dict(chosen_q)
    agent.q_target.load_state_dict(chosen_q_target)
    agent._train_steps = int(chosen_train_steps)

    model_path = out_dir / "models" / env.map_spec.name / f"{agent.algo}.pt"
    agent.save(model_path)
    print(f"[train] Saved {agent.algo} ({chosen_src}, train_steps={int(chosen_train_steps)}) -> {model_path}")
    log(
        f"[train] [{run_label}] train_one done: ckpt={chosen_src}, "
        f"total_elapsed={format_elapsed_s(time.perf_counter() - t_train_one_start)}"
    )
    train_meta: dict[str, object] = {
        "episodes_target": int(episodes),
        "episodes_completed": int(episodes_completed),
        "stop_reason": str(train_stop_reason),
        "throughput_aborted": bool(throughput_aborted),
        "rl_early_stopped": bool(rl_early_stopped),
        "progress_enabled": bool(train_eval_short_long_enabled and train_eval_every_eff > 0),
        "train_progress_rows": int(len(train_progress_history)),
        "chosen_ckpt": str(chosen_src),
        "dynamic_curriculum_enabled": bool(train_dynamic_curriculum),
        "dynamic_short_prob_final": float(dynamic_short_prob),
        "dynamic_last_update_episode": int(dynamic_last_update_ep),
        "save_ckpt_long_sr_floor": float(ckpt_long_sr_floor),
    }
    return agent, returns, eval_history, {"meta": train_meta, "train_progress": list(train_progress_history)}


# ---------------------------------------------------------------------------
# SAC training loop (separate from DQN family)
# ---------------------------------------------------------------------------

def train_one_sac(
    env: AMRBicycleEnv,
    *,
    episodes: int,
    seed: int,
    out_dir: Path,
    sac_cfg_dict: dict,
    global_map_size: int = 48,
    learning_starts: int = 500,
    forest_random_start_goal: bool = True,
    forest_rand_min_dist_m: float = 6.0,
    forest_rand_max_dist_m: float | None = None,
    forest_rand_fixed_prob: float = 0.1,
    forest_rand_tries: int = 200,
    forest_rand_edge_margin_m: float = 3.0,
    forest_curriculum: bool = True,
    curriculum_band_m: float = 4.0,
    forest_train_two_suites: bool = True,
    forest_train_short_prob: float = 0.35,
    forest_train_short_min_dist_m: float = 6.0,
    forest_train_short_max_dist_m: float | None = 14.0,
    forest_train_long_min_dist_m: float = 14.0,
    forest_train_long_max_dist_m: float | None = None,
    bc_pretrain_steps: int = 0,
    cbf_alpha: float = 0.0,
    cbf_safety_margin_m: float = 0.15,
    use_tecrl: bool = False,
    entropy_budget_ratio: float = 0.6,
    alpha_min: float = 0.0,
    lr_entropy_critic: float = 3e-4,
    use_syllabus_plr: bool = False,
    plr_levels_str: str = "6,10,14,20,30,42",
    plr_staleness_coef: float = 0.3,
    progress: bool = True,
    device: torch.device = torch.device("cpu"),
    flow_log_fp: "TextIO | None" = None,
) -> tuple[object, np.ndarray, list, dict]:
    """Simplified SAC training loop for continuous-action forest navigation."""
    from forest_vehicle_dqn.sac_agent import SACAgent, SACConfig

    progress_write = make_progress_writer(progress, flow_log_fp=flow_log_fp)

    def log(msg: str) -> None:
        if progress_write is not None:
            progress_write(str(msg))

    sac_config = SACConfig(
        map_size=int(global_map_size),
        map_channels=int(sac_cfg_dict.get("global_map_channels", 3)),
        scalar_dim=12,
        action_dim=2,
        hidden_dim=int(sac_cfg_dict.get("sac_hidden_dim", 256)),
        gamma=float(sac_cfg_dict.get("gamma", 0.99)),
        tau=float(sac_cfg_dict.get("sac_tau", 0.005)),
        lr_actor=float(sac_cfg_dict.get("sac_lr_actor", 3e-4)),
        lr_critic=float(sac_cfg_dict.get("sac_lr_critic", 3e-4)),
        lr_alpha=float(sac_cfg_dict.get("sac_lr_alpha", 3e-4)),
        batch_size=int(sac_cfg_dict.get("sac_batch_size", 256)),
        buffer_size=int(sac_cfg_dict.get("sac_buffer_size", 1_000_000)),
        target_entropy=float(sac_cfg_dict.get("sac_target_entropy", -2.0)),
        reward_scale=float(sac_cfg_dict.get("sac_reward_scale", 0.01)),
        grad_clip_norm=float(sac_cfg_dict.get("sac_grad_clip_norm", 1.0)),
        reward_clip_min=float(sac_cfg_dict.get("sac_reward_clip_min", -500.0)),
        reward_clip_max=float(sac_cfg_dict.get("sac_reward_clip_max", 1100.0)),
        target_q_min=float(sac_cfg_dict.get("sac_target_q_min", -50.0)),
        target_q_max=float(sac_cfg_dict.get("sac_target_q_max", 50.0)),
        use_huber_loss=bool(sac_cfg_dict.get("sac_use_huber_loss", True)),
        critic_warmup_steps=int(sac_cfg_dict.get("sac_critic_warmup_steps", 2000)),
        use_tecrl=bool(use_tecrl or sac_cfg_dict.get("sac_use_tecrl", False)),
        lr_entropy_critic=float(sac_cfg_dict.get("sac_lr_entropy_critic", lr_entropy_critic)),
        entropy_budget_ratio=float(sac_cfg_dict.get("sac_entropy_budget_ratio", entropy_budget_ratio)),
        alpha_min=float(sac_cfg_dict.get("sac_alpha_min", alpha_min)),
    )
    agent = SACAgent(sac_config, device=str(device), seed=seed)
    log(f"[train-sac] SACAgent created: device={device}, config={sac_config}")

    # --- Syllabus PLR curriculum ---
    plr_curriculum = None
    plr_enabled = bool(use_syllabus_plr or sac_cfg_dict.get("use_syllabus_plr", False))
    if plr_enabled:
        from forest_vehicle_dqn.plr_curriculum import ForestPLRCurriculum
        levels_str = str(sac_cfg_dict.get("plr_levels", plr_levels_str))
        levels = [float(x) for x in levels_str.split(",")]
        staleness = float(sac_cfg_dict.get("plr_staleness_coef", plr_staleness_coef))
        plr_curriculum = ForestPLRCurriculum(levels, staleness_coef=staleness, seed=seed)
        log(f"[train-sac] PLR curriculum enabled: {len(levels)} levels={levels}")

    # --- BC pretraining from expert demonstrations ---
    if bc_pretrain_steps > 0:
        log(f"[train-sac] Collecting expert demos for BC pretraining...")
        demos = []
        demo_rng = np.random.default_rng(seed + 9999)
        from forest_vehicle_dqn.baselines.mpc_local_planner import MPCConfig
        mpc_cfg = MPCConfig()
        n_demo_eps = min(50, max(20, bc_pretrain_steps // 100))
        for demo_ep in range(n_demo_eps):
            reset_opts = {
                "random_start_goal": True,
                "rand_min_dist_m": float(forest_rand_min_dist_m),
                "rand_max_dist_m": 0.0 if forest_rand_max_dist_m is None else float(forest_rand_max_dist_m),
                "rand_fixed_prob": float(forest_rand_fixed_prob),
                "rand_tries": int(forest_rand_tries),
                "rand_edge_margin_m": float(forest_rand_edge_margin_m),
            }
            try:
                env.reset(seed=int(demo_rng.integers(0, 2**31)), options=reset_opts)
            except Exception:
                continue
            for _ in range(env.max_steps):
                obs_d = env.observe_sac(map_size=global_map_size)
                try:
                    action = env.expert_continuous_action_hybrid_astar_mpc(mpc_cfg=mpc_cfg)
                except Exception:
                    break
                demos.append({"maps": obs_d["maps"], "scalars": obs_d["scalars"], "action": action})
                delta_dot = float(action[0]) * float(env.model.delta_dot_max_rad_s)
                accel = float(action[1]) * float(env.model.a_max_m_s2)
                _, _, done, truncated, _ = env.step_continuous(
                    delta_dot_rad_s=delta_dot, a_m_s2=accel)
                if done or truncated:
                    break
        log(f"[train-sac] Collected {len(demos)} expert transitions from {n_demo_eps} episodes")
        if demos:
            agent.pretrain_bc(demos, steps=bc_pretrain_steps)
            log(f"[train-sac] BC pretraining done: {bc_pretrain_steps} steps")

    cbf_enabled = float(cbf_alpha) > 0.0
    cbf_interventions = 0

    returns = np.zeros((episodes,), dtype=np.float32)
    eval_returns = np.full((episodes,), float("nan"), dtype=np.float32)
    ep_stats_list: list[dict] = []  # per-episode stats for CSV
    global_step = 0
    best_return = float("-inf")
    best_state = None
    critic_warmed_up = False
    eval_interval = int(sac_cfg_dict.get("sac_eval_interval", 50))
    t_start = time.perf_counter()

    try:
        from tqdm import tqdm
        pbar = tqdm(range(episodes), desc="SAC", disable=not progress)
    except ImportError:
        pbar = range(episodes)

    for ep in pbar:
        # Reset with random start/goal
        reset_options: dict = {}
        if forest_random_start_goal:
            min_dist = float(forest_rand_min_dist_m)
            max_dist = 0.0 if forest_rand_max_dist_m is None else float(forest_rand_max_dist_m)
            if forest_train_two_suites:
                rng_suite = np.random.default_rng(seed + ep)
                if rng_suite.random() < float(forest_train_short_prob):
                    min_dist = float(forest_train_short_min_dist_m)
                    max_dist = 0.0 if forest_train_short_max_dist_m is None else float(forest_train_short_max_dist_m)
                else:
                    min_dist = float(forest_train_long_min_dist_m)
                    max_dist = 0.0 if forest_train_long_max_dist_m is None else float(forest_train_long_max_dist_m)
            reset_options = {
                "random_start_goal": True,
                "rand_min_dist_m": min_dist,
                "rand_max_dist_m": max_dist,
                "rand_fixed_prob": float(forest_rand_fixed_prob),
                "rand_tries": int(forest_rand_tries),
                "rand_edge_margin_m": float(forest_rand_edge_margin_m),
            }

        # PLR overrides distance range when enabled
        plr_level_idx = -1
        if plr_curriculum is not None:
            plr_level_idx = plr_curriculum.sample_level()
            lo, hi = plr_curriculum.level_to_dist_range(plr_level_idx)
            reset_options["random_start_goal"] = True
            reset_options["rand_min_dist_m"] = lo
            reset_options["rand_max_dist_m"] = hi
            reset_options.setdefault("rand_fixed_prob", float(forest_rand_fixed_prob))
            reset_options.setdefault("rand_tries", int(forest_rand_tries))
            reset_options.setdefault("rand_edge_margin_m", float(forest_rand_edge_margin_m))

        obs_raw, _ = env.reset(seed=seed + ep, options=reset_options)
        obs = env.observe_sac(map_size=global_map_size)
        done = False
        truncated = False
        ep_return = 0.0

        while not (done or truncated):
            global_step += 1
            action = agent.act(obs, explore=True)
            # Map [-1,1] to physical controls
            act_dd = float(action[0])
            act_a = float(action[1])
            # CBF safety filter
            if cbf_enabled:
                act_dd, act_a, cbf_info = env.cbf_safe_action(
                    act_dd, act_a,
                    alpha_cbf=float(cbf_alpha),
                    safety_margin=float(cbf_safety_margin_m),
                )
                if cbf_info["cbf_intervened"]:
                    cbf_interventions += 1
            delta_dot = act_dd * float(env.model.delta_dot_max_rad_s)
            accel = act_a * float(env.model.a_max_m_s2)
            _, reward, done, truncated, info = env.step_continuous(
                delta_dot_rad_s=delta_dot, a_m_s2=accel)
            next_obs = env.observe_sac(map_size=global_map_size)
            agent.observe(obs, action, float(reward), next_obs, bool(done))

            if global_step >= learning_starts:
                # Critic warmup: critic-only updates before first actor update
                if not critic_warmed_up and sac_config.critic_warmup_steps > 0:
                    log(f"[train-sac] Critic warmup: {sac_config.critic_warmup_steps} steps...")
                    agent.warmup_critic(sac_config.critic_warmup_steps)
                    log("[train-sac] Critic warmup done")
                    critic_warmed_up = True
                update_info = agent.update()
            else:
                update_info = {}

            obs = next_obs
            ep_return += float(reward)

        returns[ep] = float(ep_return)

        # PLR: report episode result
        if plr_curriculum is not None and plr_level_idx >= 0:
            ep_len = int(env.current_step) if hasattr(env, "current_step") else 1
            plr_curriculum.report(plr_level_idx, float(ep_return), length=ep_len)

        # Collect per-episode stats
        ep_stat = {
            "episode": ep + 1,
            "return": float(ep_return),
            "eval_return": float("nan"),
            "critic_loss": update_info.get("critic_loss", float("nan")),
            "actor_loss": update_info.get("actor_loss", float("nan")),
            "alpha": update_info.get("alpha", float("nan")),
            "q1_mean": update_info.get("q1_mean", float("nan")),
            "target_q_mean": update_info.get("target_q_mean", float("nan")),
            "target_q_max": update_info.get("target_q_max", float("nan")),
            "cbf_interventions": cbf_interventions,
        }

        # --- Periodic deterministic evaluation ---
        if eval_interval > 0 and (ep + 1) % eval_interval == 0:
            eval_reset_opts = {
                "random_start_goal": True,
                "rand_min_dist_m": float(forest_rand_min_dist_m),
                "rand_max_dist_m": 0.0 if forest_rand_max_dist_m is None else float(forest_rand_max_dist_m),
                "rand_fixed_prob": 0.0,
                "rand_tries": int(forest_rand_tries),
                "rand_edge_margin_m": float(forest_rand_edge_margin_m),
            }
            try:
                env.reset(seed=seed + episodes + ep, options=eval_reset_opts)
                eval_obs = env.observe_sac(map_size=global_map_size)
                eval_done, eval_trunc, eval_ret = False, False, 0.0
                while not (eval_done or eval_trunc):
                    eval_action = agent.act(eval_obs, explore=False)
                    eval_dd = float(eval_action[0])
                    eval_acc = float(eval_action[1])
                    if cbf_enabled:
                        eval_dd, eval_acc, _ = env.cbf_safe_action(
                            eval_dd, eval_acc,
                            alpha_cbf=float(cbf_alpha),
                            safety_margin=float(cbf_safety_margin_m),
                        )
                    d_dot = eval_dd * float(env.model.delta_dot_max_rad_s)
                    acc = eval_acc * float(env.model.a_max_m_s2)
                    _, eval_r, eval_done, eval_trunc, _ = env.step_continuous(
                        delta_dot_rad_s=d_dot, a_m_s2=acc)
                    eval_obs = env.observe_sac(map_size=global_map_size)
                    eval_ret += float(eval_r)
                eval_returns[ep] = float(eval_ret)
                ep_stat["eval_return"] = float(eval_ret)
                log(f"[train-sac] EVAL ep={ep+1}: eval_return={eval_ret:.1f}")
            except Exception:
                pass  # eval failure is non-fatal

        ep_stats_list.append(ep_stat)

        if ep_return > best_return:
            best_return = ep_return
            best_state = True
            model_dir = out_dir / "models" / str(env.map_spec.name)
            model_dir.mkdir(parents=True, exist_ok=True)
            agent.save(model_dir / "cnn-sac.pt")

        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(ret=f"{ep_return:.1f}", best=f"{best_return:.1f}",
                             steps=global_step, alpha=f"{agent.alpha:.3f}",
                             q1=f"{update_info.get('q1_mean', 0):.2f}")

        if (ep + 1) % max(1, episodes // 20) == 0 or ep == 0 or ep == episodes - 1:
            elapsed = time.perf_counter() - t_start
            log(f"[train-sac] ep={ep+1}/{episodes}, ret={ep_return:.1f}, "
                f"best={best_return:.1f}, alpha={agent.alpha:.3f}, "
                f"q1={update_info.get('q1_mean', 0):.2f}, "
                f"tq={update_info.get('target_q_mean', 0):.2f}, "
                f"cbf={cbf_interventions}, "
                f"elapsed={elapsed:.0f}s")

    # Final save (only if no best was saved, or save as separate file)
    model_dir = out_dir / "models" / str(env.map_spec.name)
    model_dir.mkdir(parents=True, exist_ok=True)
    agent.save(model_dir / "cnn-sac-final.pt")
    if best_state is None:
        # No best was saved during training, use final
        agent.save(model_dir / "cnn-sac.pt")
    log(f"[train-sac] Done: {episodes} episodes, best_return={best_return:.1f}")

    # Write enhanced training CSV
    import csv
    csv_path = out_dir / "training_stats.csv"
    csv_fields = ["episode", "return", "eval_return", "critic_loss",
                  "actor_loss", "alpha", "q1_mean", "target_q_mean",
                  "target_q_max", "cbf_interventions"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in ep_stats_list:
            writer.writerow(row)
    log(f"[train-sac] Saved training CSV: {csv_path}")

    return agent, returns, [], {"meta": {"algo": "cnn-sac", "episodes": episodes}}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train RL agents (default: DQN) and generate Fig. 13-style reward curves.")
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
        "--rl-algos",
        nargs="+",
        default=["mlp-dqn"],
        help=(
            "RL algorithms to train: mlp-dqn mlp-ddqn mlp-pddqn cnn-dqn cnn-ddqn cnn-pddqn (or 'all'). "
            "Legacy aliases: dqn ddqn iddqn cnn-iddqn. Default: mlp-dqn."
        ),
    )
    ap.add_argument("--episodes", type=int, default=1000)
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
        help="Forest lidar sectors (36=10°, 72=5°). Ignored for non-forest envs.",
    )
    ap.add_argument("--cell-size", type=float, default=1.0, help="Grid cell size in meters.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs"),
        help="Experiment name/dir (bare names are stored under --runs-root).",
    )
    ap.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="If --out is a bare name, store it under this directory.",
    )
    ap.add_argument(
        "--timestamp-runs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write into <experiment>/train_<timestamp>/ to avoid mixing outputs.",
    )
    ap.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Torch device selection (default: auto).",
    )
    ap.add_argument("--cuda-device", type=int, default=0, help="CUDA device index (when using --device=cuda).")
    ap.add_argument(
        "--self-check",
        action="store_true",
        help="Print CUDA/runtime info and exit (use to verify CUDA setup).",
    )
    ap.add_argument("--train-freq", type=int, default=4)
    ap.add_argument("--learning-starts", type=int, default=2000)
    ap.add_argument("--gamma", type=float, default=0.995, help="Discount factor for TD targets.")
    ap.add_argument("--learning-rate", type=float, default=5e-4, help="Adam learning rate.")
    ap.add_argument("--replay-capacity", type=int, default=100_000, help="Replay buffer capacity.")
    ap.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    ap.add_argument("--target-update-steps", type=int, default=1000, help="Hard target-network update interval.")
    ap.add_argument("--target-update-tau", type=float, default=0.0, help="Soft target update tau (0 disables Polyak updates).")
    ap.add_argument("--grad-clip-norm", type=float, default=10.0, help="Gradient clipping norm.")
    ap.add_argument("--eps-start", type=float, default=0.6, help="Epsilon-greedy start value.")
    ap.add_argument("--eps-final", type=float, default=0.01, help="Epsilon-greedy final value.")
    ap.add_argument("--eps-decay", type=int, default=2000, help="Epsilon linear decay episodes.")
    ap.add_argument("--hidden-layers", type=int, default=3, help="Q-network hidden layer count.")
    ap.add_argument("--hidden-dim", type=int, default=256, help="Q-network hidden width.")
    ap.add_argument("--dueling", action="store_true", default=False, help="Use Dueling DQN architecture (V + A streams).")
    ap.add_argument("--cbam", action="store_true", default=False, help="CBAM attention on CNN feature maps (Woo et al., ECCV 2018).")
    ap.add_argument("--noisy-net", action="store_true", default=False, help="NoisyNet exploration (Fortunato et al., ICLR 2018); disables epsilon-greedy.")
    ap.add_argument("--mha", action="store_true", default=False, help="Spatial multi-head self-attention on CNN feature maps.")
    ap.add_argument("--mha-heads", type=int, default=4, help="Number of attention heads for --mha.")
    ap.add_argument("--n-quantiles", type=int, default=1, help="QR-DQN quantiles (1=standard, >1=quantile regression).")
    ap.add_argument(
        "--demo-mode",
        type=str,
        choices=("dqfd", "legacy"),
        default="dqfd",
        help=(
            "How to use demonstrations. "
            "dqfd: strict DQfD-style (PER + 1-step TD + n-step TD + margin + L2; no CE). "
            "legacy: previous stabilizer (n-step TD + margin + optional CE) + optional stratified replay."
        ),
    )
    ap.add_argument("--dqfd-n-step", type=int, default=10, help="When --demo-mode=dqfd: n-step return horizon n.")
    ap.add_argument("--dqfd-lambda-n", type=float, default=1.0, help="When --demo-mode=dqfd: n-step TD loss weight.")
    ap.add_argument("--dqfd-l2", type=float, default=1e-5, help="When --demo-mode=dqfd: L2 regularization weight.")
    ap.add_argument("--demo-lambda", type=float, default=1.0, help="Demo large-margin loss weight (DQfD).")
    ap.add_argument("--demo-margin", type=float, default=0.8, help="Demo large-margin value (DQfD).")
    ap.add_argument(
        "--aux-admissibility-lambda",
        type=float,
        default=0.0,
        help=(
            "Training-only: BCE weight for admissibility auxiliary head (CNN only). "
            "Inference still uses pure argmax(Q) without masking/replacement."
        ),
    )
    ap.add_argument("--per-alpha", type=float, default=0.4, help="When --demo-mode=dqfd: PER exponent alpha.")
    ap.add_argument("--per-beta0", type=float, default=0.6, help="When --demo-mode=dqfd: PER IS exponent beta start.")
    ap.add_argument(
        "--per-beta-steps",
        type=int,
        default=0,
        help="When --demo-mode=dqfd: beta anneal steps (0 = auto, based on episodes/max_steps/train_freq).",
    )
    ap.add_argument("--per-eps-agent", type=float, default=1e-3, help="When --demo-mode=dqfd: PER eps for agent data.")
    ap.add_argument("--per-eps-demo", type=float, default=1.0, help="When --demo-mode=dqfd: PER eps for demo data.")
    ap.add_argument(
        "--per-boost-near-goal",
        type=float,
        default=0.0,
        help="When --demo-mode=dqfd: multiplicative PER priority boost for FLAG_NEAR_GOAL transitions.",
    )
    ap.add_argument(
        "--per-boost-stuck",
        type=float,
        default=0.0,
        help="When --demo-mode=dqfd: multiplicative PER priority boost for FLAG_STUCK transitions.",
    )
    ap.add_argument(
        "--per-boost-hazard",
        type=float,
        default=0.0,
        help="When --demo-mode=dqfd: multiplicative PER priority boost for FLAG_HAZARD transitions.",
    )
    ap.add_argument(
        "--replay-stratified",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use stratified replay sampling (oversample demo/near-goal/stuck/hazard transitions).",
    )
    ap.add_argument(
        "--replay-frac-demo",
        type=float,
        default=0.0,
        help="When --replay-stratified: fraction of each batch drawn from demo transitions.",
    )
    ap.add_argument(
        "--replay-frac-goal",
        type=float,
        default=0.0,
        help="When --replay-stratified: fraction of each batch drawn from near-goal transitions.",
    )
    ap.add_argument(
        "--replay-frac-stuck",
        type=float,
        default=0.0,
        help="When --replay-stratified: fraction of each batch drawn from stuck transitions.",
    )
    ap.add_argument(
        "--replay-frac-hazard",
        type=float,
        default=0.0,
        help="When --replay-stratified: fraction of each batch drawn from near-collision (low OD) transitions.",
    )
    ap.add_argument(
        "--replay-near-goal-factor",
        type=float,
        default=2.0,
        help="Near-goal threshold factor: d_goal_m <= factor * goal_tolerance_m.",
    )
    ap.add_argument(
        "--replay-hazard-od-m",
        type=float,
        default=0.4,
        help="Hazard threshold (meters): od_m <= this marks a transition as near-collision (<=0 uses an env-based default).",
    )
    ap.add_argument("--ma-window", type=int, default=20, help="Moving average window for plotting (1=raw).")
    ap.add_argument(
        "--save-ckpt",
        type=str,
        choices=("auto", "final", "best", "pretrain"),
        default="auto",
        help=(
            "Which policy snapshot to save as models/<env>/<algo>.pt. "
            "auto (default): choose best greedy among {final,best,pretrain}; "
            "final: save the last weights; best: save the best-scoring training episode snapshot; "
            "pretrain: save the post-imitation snapshot (train_steps usually 0)."
        ),
    )
    ap.add_argument(
        "--save-ckpt-joint-short-long",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When --save-ckpt=auto and forest random start/goal is enabled, "
            "select checkpoint using joint short+long suite score: "
            "success first, then lower planning_cost."
        ),
    )
    ap.add_argument(
        "--save-ckpt-suite-runs",
        type=int,
        default=10,
        help="Number of fixed short/long pairs per suite for joint checkpoint selection.",
    )
    ap.add_argument(
        "--save-ckpt-short-min-dist-m",
        type=float,
        default=6.0,
        help="Joint checkpoint short suite minimum start-goal distance (m).",
    )
    ap.add_argument(
        "--save-ckpt-short-max-dist-m",
        type=float,
        default=14.0,
        help="Joint checkpoint short suite maximum start-goal distance (m); <=0 means no upper bound.",
    )
    ap.add_argument(
        "--save-ckpt-long-min-dist-m",
        type=float,
        default=42.0,
        help="Joint checkpoint long suite minimum start-goal distance (m).",
    )
    ap.add_argument(
        "--save-ckpt-long-max-dist-m",
        type=float,
        default=0.0,
        help="Joint checkpoint long suite maximum start-goal distance (m); <=0 means no upper bound.",
    )
    ap.add_argument(
        "--save-ckpt-long-sr-floor",
        type=float,
        default=0.0,
        help="Joint checkpoint selection: require long success_rate >= this floor before comparing costs.",
    )
    ap.add_argument(
        "--periodic-ckpt-every",
        type=int,
        default=10,
        help="Save a periodic checkpoint every N episodes to models/<env>/periodic/ep{N:05d}.pt. 0 disables.",
    )
    ap.add_argument(
        "--periodic-ckpt-keep",
        type=int,
        default=0,
        help="Keep only the latest N periodic checkpoints (0 = keep all).",
    )
    ap.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Run a greedy evaluation rollout every N episodes (0 disables; recommended for smoother learning curves).",
    )
    ap.add_argument(
        "--eval-runs",
        type=int,
        default=5,
        help=(
            "Number of evaluation rollouts per eval point. For --forest-random-start-goal this is the fixed (start,goal) "
            "batch size used throughout training."
        ),
    )
    ap.add_argument(
        "--eval-score-time-weight",
        type=float,
        default=0.5,
        help=(
            "Time weight (m/s) for the planning_cost metric written to training_eval.csv/.xlsx: "
            "planning_cost = (avg_path_length + w * inference_time_s) / max(success_rate, eps)."
        ),
    )
    ap.add_argument(
        "--train-eval-every",
        type=int,
        default=10,
        help="Forest-only: evaluate fixed short/long suites every N episodes (0 disables train-progress curves).",
    )
    ap.add_argument(
        "--train-eval-short-min-dist-m",
        type=float,
        default=6.0,
        help="Forest-only train-progress short suite minimum start-goal distance (m).",
    )
    ap.add_argument(
        "--train-eval-short-max-dist-m",
        type=float,
        default=14.0,
        help="Forest-only train-progress short suite maximum start-goal distance (m); <=0 means no upper bound.",
    )
    ap.add_argument(
        "--train-eval-long-min-dist-m",
        type=float,
        default=42.0,
        help="Forest-only train-progress long suite minimum start-goal distance (m).",
    )
    ap.add_argument(
        "--train-eval-long-max-dist-m",
        type=float,
        default=0.0,
        help="Forest-only train-progress long suite maximum start-goal distance (m); <=0 means no upper bound.",
    )
    ap.add_argument(
        "--train-eval-seed-base",
        type=int,
        default=131071,
        help="Forest-only: seed base for fixed train-progress short/long pair generation.",
    )
    ap.add_argument(
        "--train-eval-astar-timeout",
        type=float,
        default=5.0,
        help="Forest-only: A* timeout (s) for distance_ratio denominator during train-progress eval.",
    )
    ap.add_argument(
        "--train-eval-astar-max-expanded",
        type=int,
        default=1_000_000,
        help="Forest-only: A* max expanded nodes for distance_ratio denominator during train-progress eval.",
    )
    ap.add_argument(
        "--rl-early-stop-warmup-episodes",
        type=int,
        default=80,
        help="RL early-stop warmup episodes before plateau checks are enabled.",
    )
    ap.add_argument(
        "--rl-early-stop-patience-points",
        type=int,
        default=6,
        help="RL early-stop patience in number of train-progress eval points.",
    )
    ap.add_argument(
        "--rl-early-stop-min-delta-sr",
        type=float,
        default=0.01,
        help="RL early-stop: minimum sr_all improvement to reset patience.",
    )
    ap.add_argument(
        "--rl-early-stop-min-delta-ratio",
        type=float,
        default=0.01,
        help="RL early-stop: minimum distance_ratio decrease to reset patience.",
    )
    ap.add_argument(
        "--throughput-abort-min-episodes",
        type=int,
        default=50,
        help="Abort abnormal runs only while completed episodes are below this threshold.",
    )
    ap.add_argument(
        "--throughput-abort-max-minutes",
        type=float,
        default=90.0,
        help="Abort run when elapsed minutes exceed this threshold before min-episode target is reached (<=0 disables).",
    )
    ap.add_argument(
        "--obs-map-size",
        type=int,
        default=12,
        help="Downsampled global-map observation size (applies to both grid and forest envs).",
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
    ap.add_argument(
        "--forest-curriculum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forest-only curriculum: start closer to the goal, then expand to the full start-goal distance.",
    )
    ap.add_argument(
        "--curriculum-band-m",
        type=float,
        default=2.0,
        help="Curriculum band width (meters) for sampling start states by goal distance.",
    )
    ap.add_argument(
        "--curriculum-ramp",
        type=float,
        default=0.35,
        help=(
            "Forest curriculum ramp fraction (0<r<=1). The fixed-start probability and curriculum distance "
            "reach 1.0 by r*episodes (smaller = harder sooner, but avoids train/test mismatch)."
        ),
    )
    ap.add_argument(
        "--forest-demo-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forest-only: prefill replay with expert rollouts (stabilizes training).",
    )
    ap.add_argument(
        "--forest-demo-target-mult",
        type=float,
        default=40.0,
        help="Forest-only: demo prefill target multiplier (target=max(learning_starts*mult, batch_size)).",
    )
    ap.add_argument(
        "--forest-demo-target-cap",
        type=int,
        default=20000,
        help="Forest-only: cap for demo prefill target transitions (<=0 disables cap).",
    )
    ap.add_argument(
        "--forest-demo-pretrain-steps",
        type=int,
        default=50_000,
        help="Forest-only: supervised warm-start steps on demo transitions (behavior cloning + margin).",
    )
    ap.add_argument(
        "--forest-demo-pretrain-eval-every",
        type=int,
        default=2000,
        help="Forest-only: pretrain validation interval (steps).",
    )
    ap.add_argument(
        "--forest-demo-pretrain-val-runs",
        type=int,
        default=8,
        help="Forest-only: validation rollouts per pretrain check.",
    )
    ap.add_argument(
        "--forest-demo-pretrain-early-stop-sr",
        type=float,
        default=0.75,
        help="Forest-only: pretrain early-stop success-rate threshold.",
    )
    ap.add_argument(
        "--forest-demo-pretrain-early-stop-patience",
        type=int,
        default=2,
        help="Forest-only: consecutive validation passes required for pretrain early stop.",
    )
    ap.add_argument(
        "--forest-demo-horizon",
        type=int,
        default=15,
        help="Forest-only: expert horizon steps for demo prefill (constant action).",
    )
    ap.add_argument(
        "--forest-demo-w-clearance",
        type=float,
        default=0.8,
        help="Forest-only: expert clearance weight for demo prefill.",
    )
    ap.add_argument(
        "--forest-demo-filter-min-progress-ratio",
        type=float,
        default=0.15,
        help=(
            "Forest-only: reject demo episodes with total goal-distance progress ratio below this value "
            "(0 disables ratio filtering)."
        ),
    )
    ap.add_argument(
        "--forest-demo-filter-min-progress-per-step-m",
        type=float,
        default=0.002,
        help=(
            "Forest-only: reject demo episodes whose average goal-distance progress per step is below this "
            "threshold in meters (0 disables per-step filtering)."
        ),
    )
    ap.add_argument(
        "--forest-demo-filter-max-steps",
        type=int,
        default=0,
        help="Forest-only: reject demo episodes longer than this many steps (<=0 disables).",
    )
    ap.add_argument(
        "--forest-expert-exploration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forest-only: mix expert actions into the behavior policy (stabilizes long-horizon learning).",
    )
    ap.add_argument(
        "--forest-action-shield",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Forest-only: apply an admissible-action mask (safety/progress shield) to the agent's actions during training. "
            "Recommended to keep enabled even when --no-forest-expert-exploration is used to avoid train/infer mismatch."
        ),
    )
    ap.add_argument(
        "--forest-no-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Forest-only strict mode: disable training/eval fallback heuristics (expert exploration, action-shield "
            "and admissibility-based action replacement) so the reported policy is pure DDQN/DQfD."
        ),
    )
    ap.add_argument(
        "--forest-topk",
        type=int,
        default=10,
        help="Forest-only: try the top-k greedy actions before applying full admissible-progress masking.",
    )
    ap.add_argument(
        "--forest-adm-horizon",
        type=int,
        default=10,
        help="Forest-only: admissible-action horizon steps used consistently in train-time gating and pretrain validation.",
    )
    ap.add_argument(
        "--forest-min-od-m",
        type=float,
        default=0.0,
        help="Forest-only: minimum clearance (OD) required by admissible-action gating.",
    )
    ap.add_argument(
        "--forest-min-progress-m",
        type=float,
        default=1e-4,
        help=(
            "Forest-only: admissibility progress threshold (meters). "
            "Use small negative values (e.g. -0.02) to allow slight short-horizon regression for detours."
        ),
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
        "--forest-reward-k-t",
        type=float,
        default=0.1,
        help="Forest-only: per-step time penalty coefficient (reward k_t).",
    )
    ap.add_argument(
        "--forest-reward-k-delta",
        type=float,
        default=1.5,
        help="Forest-only: steering-change smoothness penalty coefficient (reward k_delta).",
    )
    ap.add_argument(
        "--forest-reward-no-progress-penalty",
        type=float,
        default=0.0,
        help="Forest-only: extra per-step penalty when speed is low and goal progress is below threshold.",
    )
    ap.add_argument(
        "--forest-reward-no-progress-eps-m",
        type=float,
        default=0.02,
        help="Forest-only: progress threshold (meters) for no-progress penalty.",
    )
    ap.add_argument(
        "--forest-reward-idle-speed-m-s",
        type=float,
        default=0.08,
        help="Forest-only: speed threshold (m/s) used by no-progress penalty.",
    )
    # SAC-specific args (used by v8 config, passed through to train_one_sac)
    ap.add_argument("--forest-reward-k-p", type=float, default=12.0)
    ap.add_argument("--forest-reward-k-kappa", type=float, default=0.2)
    ap.add_argument("--forest-reward-k-len", type=float, default=0.0)
    ap.add_argument("--forest-action-mode", type=str, default="discrete")
    ap.add_argument("--sac-lr-actor", type=float, default=3e-4)
    ap.add_argument("--sac-lr-critic", type=float, default=3e-4)
    ap.add_argument("--sac-lr-alpha", type=float, default=3e-4)
    ap.add_argument("--sac-tau", type=float, default=0.005)
    ap.add_argument("--sac-batch-size", type=int, default=256)
    ap.add_argument("--sac-buffer-size", type=int, default=1_000_000)
    ap.add_argument("--sac-hidden-dim", type=int, default=256)
    ap.add_argument("--sac-target-entropy", type=float, default=-2.0)
    ap.add_argument("--sac-reward-scale", type=float, default=0.01)
    ap.add_argument("--sac-grad-clip-norm", type=float, default=1.0)
    ap.add_argument("--sac-bc-pretrain-steps", type=int, default=0)
    ap.add_argument("--sac-reward-clip-min", type=float, default=-500.0)
    ap.add_argument("--sac-reward-clip-max", type=float, default=1100.0)
    ap.add_argument("--sac-target-q-min", type=float, default=-50.0)
    ap.add_argument("--sac-target-q-max", type=float, default=50.0)
    ap.add_argument("--sac-use-huber-loss", type=bool, default=True)
    ap.add_argument("--sac-critic-warmup-steps", type=int, default=2000)
    ap.add_argument("--sac-eval-interval", type=int, default=50)
    # CBF safety filter + reward shaping (v8p2)
    ap.add_argument("--forest-cbf-alpha", type=float, default=0.0,
                     help="CBF decay rate (0=disabled). Set >0 to enable CBF filter.")
    ap.add_argument("--forest-cbf-safety-margin-m", type=float, default=0.15)
    ap.add_argument("--forest-reward-c-prog", type=float, default=0.0,
                     help="Potential-based progress reward coeff (0=use legacy k_p).")
    ap.add_argument("--forest-reward-c-cbf", type=float, default=0.0,
                     help="CBF log-barrier reward coeff (0=use legacy k_o).")
    ap.add_argument("--forest-cbf-h-max", type=float, default=2.0)
    ap.add_argument("--forest-reward-k-o", type=float, default=1.5)
    ap.add_argument("--global-map-size", type=int, default=48)
    ap.add_argument("--global-map-channels", type=int, default=3)
    # v8p3: TECRL + exponential potential + Syllabus PLR
    ap.add_argument("--sac-use-tecrl", action="store_true", default=False)
    ap.add_argument("--sac-entropy-budget-ratio", type=float, default=0.6)
    ap.add_argument("--sac-alpha-min", type=float, default=0.0)
    ap.add_argument("--sac-lr-entropy-critic", type=float, default=3e-4)
    ap.add_argument("--reward-potential-base", type=float, default=0.0)
    ap.add_argument("--reward-potential-bias", type=float, default=0.0)
    ap.add_argument("--use-syllabus-plr", action="store_true", default=False)
    ap.add_argument("--plr-levels", type=str, default="6,10,14,20,30,42")
    ap.add_argument("--plr-staleness-coef", type=float, default=0.3)

    ap.add_argument(
        "--forest-expert",
        choices=("auto", "hybrid_astar", "astar_mpc", "hybrid_astar_mpc"),
        default="auto",
        help=(
            "Forest-only: expert source used for demos / guided exploration. "
            "auto -> hybrid_astar (or set astar_mpc / hybrid_astar_mpc explicitly)."
        ),
    )
    ap.add_argument(
        "--forest-astar-opt-enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forest-only: enable A* curve optimization (shortcut + resample + curvature+collision checks).",
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
        help="Forest-only: number of shortcut simplification passes before curve smoothing.",
    )
    ap.add_argument(
        "--forest-astar-timeout",
        type=float,
        default=5.0,
        help="Forest-only: A* planning timeout per attempt (s).",
    )
    ap.add_argument(
        "--forest-astar-max-expanded",
        type=int,
        default=1_000_000,
        help="Forest-only: A* max expanded nodes per attempt.",
    )
    ap.add_argument(
        "--forest_mpc_horizon_steps",
        type=int,
        default=12,
        help="Forest-only: MPC prediction horizon length (steps) for A*+MPC expert.",
    )
    ap.add_argument(
        "--forest_mpc_candidates",
        type=int,
        default=256,
        help="Forest-only: MPC candidate controls evaluated per step for A*+MPC expert.",
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
        help="Forest-only: MPC weight for position error term.",
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
        "--forest-random-start-goal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forest-only: randomize start/goal at each reset (goal-conditioned training).",
    )
    ap.add_argument(
        "--forest-rand-min-dist-m",
        type=float,
        default=6.0,
        help="Forest-only: minimum start→goal Euclidean distance (meters) when sampling random pairs.",
    )
    ap.add_argument(
        "--forest-rand-max-dist-m",
        type=float,
        default=0.0,
        help="Forest-only: maximum start→goal Euclidean distance (meters) when sampling random pairs (<=0 disables).",
    )
    ap.add_argument(
        "--forest-rand-fixed-prob",
        type=float,
        default=0.2,
        help="Forest-only: probability of using the canonical fixed start/goal instead of a random pair.",
    )
    ap.add_argument(
        "--forest-rand-tries",
        type=int,
        default=200,
        help="Forest-only: rejection-sampling tries per episode when sampling random start/goal pairs.",
    )
    ap.add_argument(
        "--forest-rand-edge-margin-m",
        type=float,
        default=0.0,
        help="Forest-only: minimum distance to map boundary (meters) for random start/goal sampling (<=0 disables).",
    )
    ap.add_argument(
        "--forest-train-two-suites",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Forest-only: sample train episodes from mixed short/long distance suites to match infer two-suite setup."
        ),
    )
    ap.add_argument(
        "--forest-train-short-prob",
        type=float,
        default=0.5,
        help="Forest-only: probability of sampling short suite when --forest-train-two-suites is enabled.",
    )
    ap.add_argument(
        "--forest-train-short-prob-ramp",
        type=float,
        default=0.0,
        help=(
            "Forest-only: linear ramp fraction for short-suite sampling probability when --forest-train-two-suites is enabled. "
            "During ramp, p_short transitions from 1.0 to --forest-train-short-prob; 0 disables ramp."
        ),
    )
    ap.add_argument(
        "--forest-train-dynamic-curriculum",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Forest-only: dynamic short/long curriculum for two-suite training. "
            "Updates short sampling probability from train-progress SR gaps."
        ),
    )
    ap.add_argument(
        "--forest-train-dynamic-target-sr-short",
        type=float,
        default=0.6,
        help="Forest-only dynamic curriculum target success rate for short suite.",
    )
    ap.add_argument(
        "--forest-train-dynamic-target-sr-long",
        type=float,
        default=0.6,
        help="Forest-only dynamic curriculum target success rate for long suite.",
    )
    ap.add_argument(
        "--forest-train-dynamic-k",
        type=float,
        default=0.25,
        help="Forest-only dynamic curriculum proportional gain for short-prob updates.",
    )
    ap.add_argument(
        "--forest-train-dynamic-min-short-prob",
        type=float,
        default=0.2,
        help="Forest-only dynamic curriculum lower bound for short suite sampling probability.",
    )
    ap.add_argument(
        "--forest-train-dynamic-max-short-prob",
        type=float,
        default=0.9,
        help="Forest-only dynamic curriculum upper bound for short suite sampling probability.",
    )
    ap.add_argument(
        "--forest-train-short-min-dist-m",
        type=float,
        default=6.0,
        help="Forest-only: short suite minimum start-goal distance (meters) for training sampling.",
    )
    ap.add_argument(
        "--forest-train-short-max-dist-m",
        type=float,
        default=14.0,
        help="Forest-only: short suite maximum start-goal distance (meters) for training sampling (<=0 disables).",
    )
    ap.add_argument(
        "--forest-train-long-min-dist-m",
        type=float,
        default=42.0,
        help="Forest-only: long suite minimum start-goal distance (meters) for training sampling.",
    )
    ap.add_argument(
        "--forest-train-long-max-dist-m",
        type=float,
        default=0.0,
        help="Forest-only: long suite maximum start-goal distance (meters) for training sampling (<=0 disables).",
    )
    ap.add_argument(
        "--forest-expert-prob-start",
        type=float,
        default=0.7,
        help="Forest-only: probability of using expert instead of the agent's epsilon-greedy action selection (start).",
    )
    ap.add_argument(
        "--forest-expert-prob-final",
        type=float,
        default=0.0,
        help="Forest-only: probability of using expert instead of the agent's epsilon-greedy action selection (final).",
    )
    ap.add_argument(
        "--forest-expert-prob-decay",
        type=float,
        default=0.6,
        help="Forest-only: decay fraction (0<d<=1) for expert exploration probability.",
    )
    ap.add_argument(
        "--forest-expert-adapt-k",
        type=float,
        default=0.2,
        help="Forest-only: adaptive coefficient for expert-probability correction by recent fail rate.",
    )
    ap.add_argument(
        "--forest-expert-recent-window",
        type=int,
        default=30,
        help="Forest-only: recent episode window size for adaptive expert probability.",
    )
    ap.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show a training progress bar (default: on when running in a TTY).",
    )
    ap.add_argument(
        "--save-train-log",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write training process logs to <run_dir>/train_flow.log (default: True).",
    )
    ap.add_argument(
        "--live-view",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable pygame live window during RL training episodes only "
            "(demo collect/pretrain are not displayed)."
        ),
    )
    ap.add_argument(
        "--live-view-fps",
        type=int,
        default=0,
        help="Live view target FPS (0 = no active frame cap).",
    )
    ap.add_argument(
        "--live-view-window-size",
        type=int,
        default=900,
        help="Live view square window size in pixels.",
    )
    ap.add_argument(
        "--live-view-trail-len",
        type=int,
        default=300,
        help="Live view trajectory trail length (recent N points).",
    )
    ap.add_argument(
        "--live-view-skip-steps",
        type=int,
        default=1,
        help="Draw every N environment steps in live view.",
    )
    ap.add_argument(
        "--live-view-collision-box",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw live collision detection bounding box around the vehicle.",
    )
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
        print(str(exc), file=sys.stderr)
        return 2
    if config_path is not None:
        cfg_raw = load_json(Path(config_path))
        cfg = select_section(cfg_raw, section="train")
        apply_config_defaults(ap, cfg, strict=True)

    args = ap.parse_args(argv)
    demo_mode = str(getattr(args, "demo_mode", "legacy")).lower().strip()
    if demo_mode == "dqfd" and bool(getattr(args, "replay_stratified", False)):
        print("--demo-mode=dqfd is incompatible with --replay-stratified (DQfD uses PER).", file=sys.stderr)
        return 2
    forest_envs = set(FOREST_ENV_ORDER)
    if int(args.max_steps) == 300 and args.envs and all(str(e) in forest_envs for e in args.envs):
        args.max_steps = 600
    if bool(getattr(args, "forest_no_fallback", False)):
        args.forest_action_shield = False
        args.forest_expert_exploration = False
    canonical_all = ("mlp-dqn", "mlp-ddqn", "mlp-pddqn", "cnn-dqn", "cnn-ddqn", "cnn-pddqn", "cnn-sac")
    raw_algos = [str(a).lower().strip() for a in (args.rl_algos or [])]
    if any(a == "all" for a in raw_algos):
        raw_algos = list(canonical_all[:-1])  # exclude cnn-sac from "all"

    rl_algos: list[str] = []
    unknown = []
    for a in raw_algos:
        if a == "cnn-sac":
            if "cnn-sac" not in rl_algos:
                rl_algos.append("cnn-sac")
            continue
        try:
            canonical, _arch, _base, _legacy = parse_rl_algo(a)
        except ValueError:
            unknown.append(a)
            continue
        if canonical not in rl_algos:
            rl_algos.append(canonical)

    if unknown:
        print(
            f"Unknown --rl-algos value(s): {', '.join(unknown)}. Choose from: "
            f"{' '.join(canonical_all)} (or 'all'). Legacy aliases: dqn ddqn iddqn cnn-iddqn.",
            file=sys.stderr,
        )
        return 2
    if not rl_algos:
        print(f"No RL algorithms selected (choose from: {' '.join(canonical_all)}).", file=sys.stderr)
        return 2
    args.rl_algos = rl_algos

    try:
        device = select_device(device=args.device, cuda_device=args.cuda_device)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    progress = bool(sys.stderr.isatty()) if args.progress is None else bool(args.progress)
    save_train_log = bool(getattr(args, "save_train_log", True))
    experiment_dir = resolve_experiment_dir(args.out, runs_root=args.runs_root)
    run_paths = create_run_dir(experiment_dir, timestamp_runs=args.timestamp_runs, prefix="train")
    out_dir = run_paths.run_dir
    flow_log_path = out_dir / "train_flow.log"
    flow_log_fp: TextIO | None = None
    if save_train_log:
        flow_log_fp = flow_log_path.open("w", encoding="utf-8")
    progress_write = make_progress_writer(progress, flow_log_fp=flow_log_fp)

    def log(msg: str) -> None:
        progress_write(str(msg))

    t_main_start = time.perf_counter()
    log(
        f"[train] Run start: envs={len(args.envs)}, algos={len(args.rl_algos)}, "
        f"episodes={int(args.episodes)}, device={device}, demo_mode={demo_mode}, progress={bool(progress)}"
    )
    if flow_log_fp is not None:
        log(f"[train] Flow log path: {flow_log_path}")

    astar_curve_cfg = AStarCurveOptConfig(
        enable=bool(args.forest_astar_opt_enable),
        max_replans=max(1, int(args.forest_astar_opt_max_replans)),
        resample_ds_m=max(1e-4, float(args.forest_astar_opt_resample_ds_m)),
        collision_step_m=max(1e-4, float(args.forest_astar_opt_collision_step_m)),
        shortcut_passes=max(0, int(args.forest_astar_opt_shortcut_passes)),
    )
    mpc_cfg = MPCConfig(
        horizon_steps=max(1, int(args.forest_mpc_horizon_steps)),
        candidates=max(9, int(args.forest_mpc_candidates)),
        dt_s=float(args.forest_mpc_dt_s),
        w_u=float(args.forest_mpc_w_u),
        w_du=float(args.forest_mpc_w_du),
        w_pos=float(args.forest_mpc_w_pos),
        w_yaw=float(args.forest_mpc_w_yaw),
        align_dist_m=float(args.forest_mpc_align_dist_m),
        collision_padding_m=float(args.forest_mpc_collision_padding_m),
        goal_lookahead_steps=max(1, int(args.forest_mpc_goal_lookahead_steps)),
    )
    astar_timeout_s = max(1e-3, float(args.forest_astar_timeout))
    astar_max_expanded = max(1, int(args.forest_astar_max_expanded))

    agent_cfg = AgentConfig()
    per_beta_steps = int(getattr(args, "per_beta_steps", 0))
    if demo_mode == "dqfd" and per_beta_steps <= 0:
        per_beta_steps = max(1, int(args.episodes) * int(args.max_steps) // max(1, int(args.train_freq)))
    agent_cfg = replace(
        agent_cfg,
        gamma=float(getattr(args, "gamma", agent_cfg.gamma)),
        learning_rate=float(getattr(args, "learning_rate", agent_cfg.learning_rate)),
        replay_capacity=int(getattr(args, "replay_capacity", agent_cfg.replay_capacity)),
        batch_size=int(getattr(args, "batch_size", agent_cfg.batch_size)),
        target_update_steps=int(getattr(args, "target_update_steps", agent_cfg.target_update_steps)),
        target_update_tau=float(getattr(args, "target_update_tau", agent_cfg.target_update_tau)),
        grad_clip_norm=float(getattr(args, "grad_clip_norm", agent_cfg.grad_clip_norm)),
        eps_start=float(getattr(args, "eps_start", agent_cfg.eps_start)),
        eps_final=float(getattr(args, "eps_final", agent_cfg.eps_final)),
        eps_decay=int(getattr(args, "eps_decay", agent_cfg.eps_decay)),
        hidden_layers=int(getattr(args, "hidden_layers", agent_cfg.hidden_layers)),
        hidden_dim=int(getattr(args, "hidden_dim", agent_cfg.hidden_dim)),
        dueling=bool(getattr(args, "dueling", False)),
        cbam=bool(getattr(args, "cbam", False)),
        noisy_net=bool(getattr(args, "noisy_net", False)),
        mha=bool(getattr(args, "mha", False)),
        mha_heads=int(getattr(args, "mha_heads", 4)),
        n_quantiles=int(getattr(args, "n_quantiles", 1)),
        demo_mode=str(demo_mode),
        dqfd_lambda_n=float(getattr(args, "dqfd_lambda_n", 1.0)),
        l2_reg=float(getattr(args, "dqfd_l2", 0.0)),
        demo_lambda=float(getattr(args, "demo_lambda", agent_cfg.demo_lambda)),
        demo_margin=float(getattr(args, "demo_margin", agent_cfg.demo_margin)),
        aux_admissibility_lambda=float(getattr(args, "aux_admissibility_lambda", 0.0)),
        replay_stratified=bool(args.replay_stratified),
        replay_prioritized=bool(demo_mode == "dqfd"),
        per_alpha=float(getattr(args, "per_alpha", 0.4)),
        per_beta0=float(getattr(args, "per_beta0", 0.0)),
        per_beta_steps=int(per_beta_steps),
        per_eps_agent=float(getattr(args, "per_eps_agent", 1e-3)),
        per_eps_demo=float(getattr(args, "per_eps_demo", 1.0)),
        per_boost_near_goal=float(getattr(args, "per_boost_near_goal", 0.0)),
        per_boost_stuck=float(getattr(args, "per_boost_stuck", 0.0)),
        per_boost_hazard=float(getattr(args, "per_boost_hazard", 0.0)),
        demo_ce_lambda=(0.0 if demo_mode == "dqfd" else float(agent_cfg.demo_ce_lambda)),
        replay_frac_demo=float(args.replay_frac_demo),
        replay_frac_goal=float(args.replay_frac_goal),
        replay_frac_stuck=float(args.replay_frac_stuck),
        replay_frac_hazard=float(args.replay_frac_hazard),
    )
    legacy_n_step = 3
    dqfd_n_step = int(max(1, int(getattr(args, "dqfd_n_step", 10))))
    n_step = int(dqfd_n_step if demo_mode == "dqfd" else legacy_n_step)
    dqn_cfg = replace(agent_cfg, n_step=n_step)
    ddqn_cfg = replace(agent_cfg, n_step=n_step)
    pddqn_cfg = replace(agent_cfg, n_step=n_step, target_update_tau=0.01)
    (out_dir / "configs").mkdir(parents=True, exist_ok=True)
    args_payload: dict[str, object] = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            args_payload[k] = str(v)
        else:
            args_payload[k] = v

    (out_dir / "configs" / "run.json").write_text(
        json.dumps(
            {
                "kind": "train",
                "argv": list(sys.argv),
                "experiment_dir": str(run_paths.experiment_dir),
                "run_dir": str(run_paths.run_dir),
                "args": args_payload,
                "torch": asdict(torch_runtime_info()),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (out_dir / "configs" / "agent_config.json").write_text(json.dumps(asdict(agent_cfg), indent=2, sort_keys=True), encoding="utf-8")
    algo_cfgs = {
        "mlp-dqn": dqn_cfg,
        "mlp-ddqn": ddqn_cfg,
        "mlp-pddqn": pddqn_cfg,
        "cnn-dqn": dqn_cfg,
        "cnn-ddqn": ddqn_cfg,
        "cnn-pddqn": pddqn_cfg,
    }
    for algo in args.rl_algos:
        cfg = algo_cfgs.get(str(algo))
        if cfg is None:
            continue
        (out_dir / "configs" / f"agent_config_{algo}.json").write_text(
            json.dumps(asdict(cfg), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    all_rows: list[dict[str, float | int | str]] = []
    all_eval_rows: list[dict[str, float | int | str]] = []
    curves: dict[str, dict[str, np.ndarray]] = {}
    algo_labels = {
        "mlp-dqn": "MLP-DQN",
        "mlp-ddqn": "MLP-DDQN",
        "mlp-pddqn": "MLP-PDDQN",
        "cnn-dqn": "CNN-DQN",
        "cnn-ddqn": "CNN-DDQN",
        "cnn-pddqn": "CNN-PDDQN",
        "cnn-sac": "CNN-SAC",
    }

    for env_name in args.envs:
        t_env_start = time.perf_counter()
        log(f"[train] Env start: {env_name}")
        spec = get_map_spec(env_name)
        if env_name in FOREST_ENV_ORDER:
            env = AMRBicycleEnv(
                spec,
                max_steps=args.max_steps,
                cell_size_m=0.1,
                sensor_range_m=float(args.sensor_range),
                n_sectors=args.n_sectors,
                obs_map_size=int(args.obs_map_size),
                goal_tolerance_m=float(args.goal_tolerance_m),
                goal_angle_tolerance_deg=float(args.goal_angle_tolerance_deg),
                goal_stop_speed_m_s=float(args.goal_stop_speed_m_s),
                goal_stop_delta_deg=float(args.goal_stop_delta_deg),
                reward_k_t=float(args.forest_reward_k_t),
                reward_k_delta=float(args.forest_reward_k_delta),
                reward_k_p=float(getattr(args, "forest_reward_k_p", 12.0)),
                reward_k_kappa=float(getattr(args, "forest_reward_k_kappa", 0.2)),
                reward_k_len=float(getattr(args, "forest_reward_k_len", 0.0)),
                reward_no_progress_penalty=float(args.forest_reward_no_progress_penalty),
                reward_no_progress_eps_m=float(args.forest_reward_no_progress_eps_m),
                reward_idle_speed_m_s=float(args.forest_reward_idle_speed_m_s),
                goal_admissible_relax_factor=float(args.forest_goal_admissible_relax_factor),
                terminate_on_stuck=not bool(getattr(args, "no_terminate_on_stuck", False)),
                action_delta_dot_bins=int(args.forest_action_delta_dot_bins),
                action_accel_bins=int(args.forest_action_accel_bins),
                action_grid_power=float(args.forest_action_grid_power),
                reward_c_prog=float(getattr(args, "forest_reward_c_prog", 0.0)),
                reward_c_cbf=float(getattr(args, "forest_reward_c_cbf", 0.0)),
                cbf_h_max=float(getattr(args, "forest_cbf_h_max", 2.0)),
                gamma=float(getattr(args, "gamma", 0.99)),
                reward_k_o=float(getattr(args, "forest_reward_k_o", 1.5)),
                reward_potential_base=float(getattr(args, "reward_potential_base", 0.0)),
                reward_potential_bias=float(getattr(args, "reward_potential_bias", 0.0)),
            )
            forest_demo_data = None
            has_dqn_algos = any(str(a) != "cnn-sac" for a in args.rl_algos)
            if has_dqn_algos and bool(args.forest_demo_prefill) and int(args.learning_starts) > 0:
                demo_target = forest_demo_target(
                    learning_starts=int(args.learning_starts),
                    batch_size=int(agent_cfg.batch_size),
                    target_mult=float(args.forest_demo_target_mult),
                    target_cap=int(args.forest_demo_target_cap),
                )
                rand_max = None if float(args.forest_rand_max_dist_m) <= 0.0 else float(args.forest_rand_max_dist_m)
                log(
                    f"[train] [{env_name}] Prepare demo cache: target={int(demo_target)}, "
                    f"expert={str(args.forest_expert)}"
                )
                t_demo_collect_start = time.perf_counter()
                forest_demo_data = collect_forest_demos(
                    env,
                    target=int(demo_target),
                    seed=int(args.seed + 1000),
                    forest_curriculum=bool(args.forest_curriculum),
                    curriculum_band_m=float(args.curriculum_band_m),
                    forest_random_start_goal=bool(args.forest_random_start_goal),
                    forest_rand_min_dist_m=float(args.forest_rand_min_dist_m),
                    forest_rand_max_dist_m=rand_max,
                    forest_rand_fixed_prob=float(args.forest_rand_fixed_prob),
                    forest_rand_tries=int(args.forest_rand_tries),
                    forest_rand_edge_margin_m=float(args.forest_rand_edge_margin_m),
                    forest_expert=str(args.forest_expert),
                    forest_demo_horizon=int(args.forest_demo_horizon),
                    forest_adm_horizon=int(args.forest_adm_horizon),
                    forest_min_od_m=float(args.forest_min_od_m),
                    forest_min_progress_m=float(args.forest_min_progress_m),
                    forest_use_admissible_next_mask=(not bool(getattr(args, "forest_no_fallback", False))),
                    forest_demo_w_clearance=float(args.forest_demo_w_clearance),
                    mpc_cfg=mpc_cfg,
                    astar_curve_cfg=astar_curve_cfg,
                    astar_seed=int(args.seed + 1000),
                    astar_timeout_s=float(astar_timeout_s),
                    astar_max_expanded=int(astar_max_expanded),
                    forest_demo_filter_min_progress_ratio=float(args.forest_demo_filter_min_progress_ratio),
                    forest_demo_filter_min_progress_per_step_m=float(args.forest_demo_filter_min_progress_per_step_m),
                    forest_demo_filter_max_steps=int(args.forest_demo_filter_max_steps),
                    progress_write=progress_write,
                )
                log(
                    f"[train] [{env_name}] Demo cache ready: size={int(forest_demo_data[0].shape[0])}, "
                    f"elapsed={format_elapsed_s(time.perf_counter() - t_demo_collect_start)}"
                )
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
            forest_demo_data = None

        rand_max = None if float(args.forest_rand_max_dist_m) <= 0.0 else float(args.forest_rand_max_dist_m)

        env_curves: dict[str, np.ndarray] = {}
        env_eval_rows: dict[str, list[dict[str, float | int]]] = {}
        env_train_meta: dict[str, dict[str, object]] = {}
        for algo in args.rl_algos:
            t_algo_start = time.perf_counter()
            log(f"[train] Algo start: env={env_name}, algo={str(algo)}")

            # --- SAC branch ---
            if str(algo) == "cnn-sac":
                if not isinstance(env, AMRBicycleEnv):
                    print(f"cnn-sac requires AMRBicycleEnv (forest env), got {type(env).__name__}", file=sys.stderr)
                    return 2
                sac_cfg_raw = {}
                if config_path is not None:
                    sac_cfg_raw = select_section(load_json(Path(config_path)), section="train")
                try:
                    _, algo_returns, algo_eval, algo_extra = train_one_sac(
                        env,
                        episodes=args.episodes,
                        seed=args.seed + 1000,
                        out_dir=out_dir,
                        sac_cfg_dict=sac_cfg_raw,
                        global_map_size=int(getattr(args, "global_map_size", 48)),
                        learning_starts=int(args.learning_starts),
                        forest_random_start_goal=bool(args.forest_random_start_goal),
                        forest_rand_min_dist_m=float(args.forest_rand_min_dist_m),
                        forest_rand_max_dist_m=rand_max,
                        forest_rand_fixed_prob=float(args.forest_rand_fixed_prob),
                        forest_rand_tries=int(args.forest_rand_tries),
                        forest_rand_edge_margin_m=float(args.forest_rand_edge_margin_m),
                        forest_curriculum=bool(args.forest_curriculum),
                        forest_train_two_suites=bool(getattr(args, "forest_train_two_suites", False)),
                        forest_train_short_prob=float(getattr(args, "forest_train_short_prob", 0.35)),
                        forest_train_short_min_dist_m=float(getattr(args, "forest_train_short_min_dist_m", 6.0)),
                        forest_train_short_max_dist_m=(
                            None if float(getattr(args, "forest_train_short_max_dist_m", 14.0)) <= 0.0
                            else float(getattr(args, "forest_train_short_max_dist_m", 14.0))
                        ),
                        forest_train_long_min_dist_m=float(getattr(args, "forest_train_long_min_dist_m", 42.0)),
                        forest_train_long_max_dist_m=(
                            None if float(getattr(args, "forest_train_long_max_dist_m", 0.0)) <= 0.0
                            else float(getattr(args, "forest_train_long_max_dist_m", 0.0))
                        ),
                        bc_pretrain_steps=int(getattr(args, "sac_bc_pretrain_steps",
                                                      sac_cfg_raw.get("sac_bc_pretrain_steps", 0))),
                        cbf_alpha=float(getattr(args, "forest_cbf_alpha",
                                                sac_cfg_raw.get("forest_cbf_alpha", 0.0))),
                        cbf_safety_margin_m=float(getattr(args, "forest_cbf_safety_margin_m",
                                                          sac_cfg_raw.get("forest_cbf_safety_margin_m", 0.15))),
                        use_tecrl=bool(getattr(args, "sac_use_tecrl",
                                               sac_cfg_raw.get("sac_use_tecrl", False))),
                        entropy_budget_ratio=float(getattr(args, "sac_entropy_budget_ratio",
                                                           sac_cfg_raw.get("sac_entropy_budget_ratio", 0.6))),
                        alpha_min=float(getattr(args, "sac_alpha_min",
                                                sac_cfg_raw.get("sac_alpha_min", 0.0))),
                        lr_entropy_critic=float(getattr(args, "sac_lr_entropy_critic",
                                                        sac_cfg_raw.get("sac_lr_entropy_critic", 3e-4))),
                        use_syllabus_plr=bool(getattr(args, "use_syllabus_plr",
                                                      sac_cfg_raw.get("use_syllabus_plr", False))),
                        plr_levels_str=str(getattr(args, "plr_levels",
                                                   sac_cfg_raw.get("plr_levels", "6,10,14,20,30,42"))),
                        plr_staleness_coef=float(getattr(args, "plr_staleness_coef",
                                                         sac_cfg_raw.get("plr_staleness_coef", 0.3))),
                        progress=progress,
                        device=device,
                        flow_log_fp=flow_log_fp,
                    )
                except Exception:
                    import traceback
                    traceback.print_exc()
                    return 2
                env_curves[str(algo)] = algo_returns
                env_eval_rows[str(algo)] = list(algo_eval)
                env_train_meta[str(algo)] = algo_extra.get("meta", {})
                log(f"[train] Algo done: env={env_name}, algo={str(algo)}, "
                    f"elapsed={format_elapsed_s(time.perf_counter() - t_algo_start)}")
                continue

            # --- DQN family branch ---
            cfg = algo_cfgs[str(algo)]
            live_viewer = None
            if bool(getattr(args, "live_view", False)):
                live_viewer = TrainLiveViewer(
                    enabled=True,
                    fps=int(getattr(args, "live_view_fps", 0)),
                    window_size=max(64, int(getattr(args, "live_view_window_size", 900))),
                    trail_len=max(1, int(getattr(args, "live_view_trail_len", 300))),
                    skip_steps=max(1, int(getattr(args, "live_view_skip_steps", 1))),
                    show_collision_box=bool(getattr(args, "live_view_collision_box", True)),
                )
            try:
                _, algo_returns, algo_eval, algo_extra = train_one(
                    env,
                    str(algo),
                    episodes=args.episodes,
                    # Forest training (global-map + imitation warm-start) can be sensitive to random initialization.
                    # Keep a deterministic seed offset across algorithms for fair comparisons.
                    seed=args.seed + 1000,
                    out_dir=out_dir,
                    agent_cfg=cfg,
                    replay_near_goal_factor=float(args.replay_near_goal_factor),
                    replay_hazard_od_m=float(args.replay_hazard_od_m),
                    train_freq=args.train_freq,
                    learning_starts=args.learning_starts,
                    forest_demo_target_mult=float(args.forest_demo_target_mult),
                    forest_demo_target_cap=int(args.forest_demo_target_cap),
                    save_ckpt=str(args.save_ckpt),
                    forest_curriculum=bool(args.forest_curriculum),
                    curriculum_band_m=float(args.curriculum_band_m),
                    curriculum_ramp=float(args.curriculum_ramp),
                    forest_demo_prefill=bool(args.forest_demo_prefill),
                    forest_demo_pretrain_steps=int(args.forest_demo_pretrain_steps),
                    forest_demo_pretrain_eval_every=int(args.forest_demo_pretrain_eval_every),
                    forest_demo_pretrain_val_runs=int(args.forest_demo_pretrain_val_runs),
                    forest_demo_pretrain_early_stop_sr=float(args.forest_demo_pretrain_early_stop_sr),
                    forest_demo_pretrain_early_stop_patience=int(args.forest_demo_pretrain_early_stop_patience),
                    forest_demo_horizon=int(args.forest_demo_horizon),
                    forest_demo_w_clearance=float(args.forest_demo_w_clearance),
                    forest_demo_data=forest_demo_data,
                    forest_expert=str(args.forest_expert),
                    forest_expert_exploration=bool(args.forest_expert_exploration),
                    forest_action_shield=bool(args.forest_action_shield),
                    forest_adm_horizon=int(args.forest_adm_horizon),
                    forest_topk=int(args.forest_topk),
                    forest_min_od_m=float(args.forest_min_od_m),
                    forest_min_progress_m=float(args.forest_min_progress_m),
                    forest_no_fallback=bool(getattr(args, "forest_no_fallback", False)),
                    forest_expert_prob_start=float(args.forest_expert_prob_start),
                    forest_expert_prob_final=float(args.forest_expert_prob_final),
                    forest_expert_prob_decay=float(args.forest_expert_prob_decay),
                    forest_expert_adapt_k=float(args.forest_expert_adapt_k),
                    forest_expert_recent_window=int(args.forest_expert_recent_window),
                    forest_random_start_goal=bool(args.forest_random_start_goal),
                    forest_rand_min_dist_m=float(args.forest_rand_min_dist_m),
                    forest_rand_max_dist_m=rand_max,
                    forest_rand_fixed_prob=float(args.forest_rand_fixed_prob),
                    forest_rand_tries=int(args.forest_rand_tries),
                    forest_rand_edge_margin_m=float(args.forest_rand_edge_margin_m),
                    forest_train_two_suites=bool(getattr(args, "forest_train_two_suites", False)),
                    forest_train_short_prob=float(getattr(args, "forest_train_short_prob", 0.5)),
                    forest_train_short_prob_ramp=float(getattr(args, "forest_train_short_prob_ramp", 0.0)),
                    forest_train_dynamic_curriculum=bool(getattr(args, "forest_train_dynamic_curriculum", False)),
                    forest_train_dynamic_target_sr_short=float(
                        getattr(args, "forest_train_dynamic_target_sr_short", 0.6)
                    ),
                    forest_train_dynamic_target_sr_long=float(
                        getattr(args, "forest_train_dynamic_target_sr_long", 0.6)
                    ),
                    forest_train_dynamic_k=float(getattr(args, "forest_train_dynamic_k", 0.25)),
                    forest_train_dynamic_min_short_prob=float(
                        getattr(args, "forest_train_dynamic_min_short_prob", 0.2)
                    ),
                    forest_train_dynamic_max_short_prob=float(
                        getattr(args, "forest_train_dynamic_max_short_prob", 0.9)
                    ),
                    forest_train_short_min_dist_m=float(getattr(args, "forest_train_short_min_dist_m", 6.0)),
                    forest_train_short_max_dist_m=(
                        None
                        if float(getattr(args, "forest_train_short_max_dist_m", 14.0)) <= 0.0
                        else float(getattr(args, "forest_train_short_max_dist_m", 14.0))
                    ),
                    forest_train_long_min_dist_m=float(getattr(args, "forest_train_long_min_dist_m", 42.0)),
                    forest_train_long_max_dist_m=(
                        None
                        if float(getattr(args, "forest_train_long_max_dist_m", 0.0)) <= 0.0
                        else float(getattr(args, "forest_train_long_max_dist_m", 0.0))
                    ),
                    astar_curve_cfg=astar_curve_cfg,
                    astar_timeout_s=float(astar_timeout_s),
                    astar_max_expanded=int(astar_max_expanded),
                    mpc_cfg=mpc_cfg,
                    eval_every=int(args.eval_every),
                    eval_runs=int(args.eval_runs),
                    train_eval_every=int(getattr(args, "train_eval_every", 10)),
                    train_eval_short_min_dist_m=float(getattr(args, "train_eval_short_min_dist_m", 6.0)),
                    train_eval_short_max_dist_m=(
                        None
                        if float(getattr(args, "train_eval_short_max_dist_m", 14.0)) <= 0.0
                        else float(getattr(args, "train_eval_short_max_dist_m", 14.0))
                    ),
                    train_eval_long_min_dist_m=float(getattr(args, "train_eval_long_min_dist_m", 42.0)),
                    train_eval_long_max_dist_m=(
                        None
                        if float(getattr(args, "train_eval_long_max_dist_m", 0.0)) <= 0.0
                        else float(getattr(args, "train_eval_long_max_dist_m", 0.0))
                    ),
                    train_eval_seed_base=int(getattr(args, "train_eval_seed_base", 131071)) + int(args.seed + 1000),
                    train_eval_astar_timeout_s=float(getattr(args, "train_eval_astar_timeout", 5.0)),
                    train_eval_astar_max_expanded=int(getattr(args, "train_eval_astar_max_expanded", 1_000_000)),
                    rl_early_stop_warmup_episodes=int(getattr(args, "rl_early_stop_warmup_episodes", 80)),
                    rl_early_stop_patience_points=int(getattr(args, "rl_early_stop_patience_points", 6)),
                    rl_early_stop_min_delta_sr=float(getattr(args, "rl_early_stop_min_delta_sr", 0.01)),
                    rl_early_stop_min_delta_ratio=float(getattr(args, "rl_early_stop_min_delta_ratio", 0.01)),
                    throughput_abort_min_episodes=int(getattr(args, "throughput_abort_min_episodes", 50)),
                    throughput_abort_max_minutes=float(getattr(args, "throughput_abort_max_minutes", 90.0)),
                    eval_score_time_weight=float(args.eval_score_time_weight),
                    save_ckpt_joint_short_long=bool(getattr(args, "save_ckpt_joint_short_long", False)),
                    save_ckpt_suite_runs=int(getattr(args, "save_ckpt_suite_runs", 10)),
                    save_ckpt_short_min_dist_m=float(getattr(args, "save_ckpt_short_min_dist_m", 6.0)),
                    save_ckpt_short_max_dist_m=(
                        None
                        if float(getattr(args, "save_ckpt_short_max_dist_m", 14.0)) <= 0.0
                        else float(getattr(args, "save_ckpt_short_max_dist_m", 14.0))
                    ),
                    save_ckpt_long_min_dist_m=float(getattr(args, "save_ckpt_long_min_dist_m", 42.0)),
                    save_ckpt_long_max_dist_m=(
                        None
                        if float(getattr(args, "save_ckpt_long_max_dist_m", 0.0)) <= 0.0
                        else float(getattr(args, "save_ckpt_long_max_dist_m", 0.0))
                    ),
                    save_ckpt_long_sr_floor=float(getattr(args, "save_ckpt_long_sr_floor", 0.0)),
                    periodic_ckpt_every=int(getattr(args, "periodic_ckpt_every", 0)),
                    periodic_ckpt_keep=int(getattr(args, "periodic_ckpt_keep", 0)),
                    progress=progress,
                    device=device,
                    live_viewer=live_viewer,
                    progress_write=progress_write,
                )
            finally:
                if live_viewer is not None:
                    live_viewer.close()
            env_curves[str(algo)] = algo_returns
            env_eval_rows[str(algo)] = list(algo_eval)
            for row in algo_eval:
                all_eval_rows.append({"env": env_name, "algo": str(algo), **row})
            train_progress_rows = algo_extra.get("train_progress", [])
            if isinstance(train_progress_rows, list):
                for row in train_progress_rows:
                    if isinstance(row, dict):
                        all_eval_rows.append({"env": env_name, "algo": str(algo), **row})
            meta_obj = algo_extra.get("meta", {})
            env_train_meta[str(algo)] = dict(meta_obj) if isinstance(meta_obj, dict) else {}
            log(
                f"[train] Algo done: env={env_name}, algo={str(algo)}, "
                f"elapsed={format_elapsed_s(time.perf_counter() - t_algo_start)}"
            )

            meta_path = out_dir / "configs" / f"train_meta_{env_name}.json"
            meta_path.write_text(
                json.dumps({"env": str(env_name), "algos": env_train_meta}, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            log(f"[train] Wrote train meta: {meta_path}")

            if any(bool(meta.get("throughput_aborted", False)) for meta in env_train_meta.values()):
                print(
                    f"[train] throughput fuse triggered on env={env_name}; "
                    "failure_reason=training_throughput_abnormal",
                    file=sys.stderr,
                )

        curves[env_name] = dict(env_curves)
        for ep in range(args.episodes):
            row: dict[str, float | int | str] = {"env": env_name, "episode": ep + 1}
            for algo, returns_arr in env_curves.items():
                row[f"{algo}_return"] = float(returns_arr[ep])
            all_rows.append(row)
        log(f"[train] Env done: {env_name}, elapsed={format_elapsed_s(time.perf_counter() - t_env_start)}")

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "training_returns.csv", index=False)
    log(f"[train] Wrote returns csv: {out_dir / 'training_returns.csv'}")
    if all_eval_rows:
        df_eval = pd.DataFrame(all_eval_rows)
        df_eval.to_csv(out_dir / "training_eval.csv", index=False)
        log(f"[train] Wrote eval csv: {out_dir / 'training_eval.csv'}")
        try:
            df_eval.to_excel(out_dir / "training_eval.xlsx", index=False)
        except Exception as exc:
            print(f"Warning: failed to write training_eval.xlsx: {exc}", file=sys.stderr)
        try:
            plot_training_eval_metrics(df_eval, out_path=out_dir / "training_eval_metrics.png")
        except Exception as exc:
            print(f"Warning: failed to write training_eval_metrics.png: {exc}", file=sys.stderr)

    # Plot Fig. 13-style reward curves
    envs_to_plot = list(args.envs)[:4]
    n_env = int(len(envs_to_plot))
    cols = 1 if n_env <= 1 else 2
    rows_n = int(np.ceil(float(n_env) / float(cols))) if n_env else 1
    fig, axes = plt.subplots(rows_n, cols, figsize=(5.2 * cols, 3.8 * rows_n), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    used = 0
    for i, env_name in enumerate(envs_to_plot):
        ax = axes[i]
        for algo in args.rl_algos:
            series = curves.get(env_name, {}).get(str(algo))
            if series is None:
                continue
            series_plot = moving_average(series, args.ma_window)
            ax.plot(
                range(1, args.episodes + 1),
                series_plot,
                label=algo_labels.get(str(algo), str(algo).upper()),
                linewidth=1.0,
            )
        ax.set_title(f"Env. ({env_name})")
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Rewards")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        used += 1

    for ax in axes[n_env:]:
        ax.axis("off")

    algo_title = ", ".join(algo_labels.get(str(a), str(a).upper()) for a in args.rl_algos)
    fig.suptitle(f"Training reward curves ({algo_title})")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "fig13_rewards.png", dpi=200)
    plt.close(fig)
    log(f"[train] Wrote reward figure: {out_dir / 'fig13_rewards.png'}")

    print(f"Wrote: {out_dir / 'fig13_rewards.png'}")
    print(f"Wrote: {out_dir / 'training_returns.csv'}")
    if all_eval_rows:
        print(f"Wrote: {out_dir / 'training_eval.csv'}")
        if (out_dir / "training_eval.xlsx").exists():
            print(f"Wrote: {out_dir / 'training_eval.xlsx'}")
        if (out_dir / "training_eval_metrics.png").exists():
            print(f"Wrote: {out_dir / 'training_eval_metrics.png'}")
    print(f"Wrote models under: {out_dir / 'models'}")
    print(f"Run dir: {out_dir}")
    if flow_log_fp is not None:
        print(f"Wrote: {flow_log_path}")
    log(f"[train] Run done: total_elapsed={format_elapsed_s(time.perf_counter() - t_main_start)}")
    if flow_log_fp is not None:
        flow_log_fp.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
