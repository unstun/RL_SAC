from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import cv2
try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    import gym  # type: ignore
import numpy as np

from forest_vehicle_dqn.maps import MapSpec
from forest_vehicle_dqn.baselines.pathplan import (
    AStarCurveOptConfig,
    default_ackermann_params,
    forest_two_circle_footprint,
    grid_map_from_obstacles,
    plan_grid_astar,
)


_ACTIONS_8 = np.array(
    [
        (0, 1),  # up
        (0, -1),  # down
        (-1, 0),  # left
        (1, 0),  # right
        (1, 1),  # upper right
        (1, -1),  # lower right
        (-1, 1),  # upper left
        (-1, -1),  # lower left
    ],
    dtype=np.int32,
)


@dataclass(frozen=True)
class RewardWeights:
    lambda_target: float = 1.0
    lambda_distance: float = -0.35
    lambda_boundary: float = -1.0
    lambda_obstacle: float = -1.0


class AMRGridEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        map_spec: MapSpec,
        *,
        sensor_range: int = 6,
        max_steps: int = 500,
        reward: RewardWeights = RewardWeights(),
        cell_size: float = 1.0,
        safe_distance: float = 0.6,
        obs_map_size: int = 12,
        terminate_on_collision: bool = False,
    ) -> None:
        super().__init__()

        self.map_spec = map_spec
        self._grid = map_spec.obstacle_grid()  # (H, W), y=0 bottom
        self._height, self._width = self._grid.shape
        self.start_xy = map_spec.start_xy
        self.goal_xy = map_spec.goal_xy
        self.sensor_range = int(sensor_range)
        self.max_steps = int(max_steps)
        self.reward = reward
        self.cell_size = float(cell_size)
        if not (self.cell_size > 0):
            raise ValueError("cell_size must be > 0")
        self.safe_distance = float(safe_distance)
        self.terminate_on_collision = bool(terminate_on_collision)

        self.action_space = gym.spaces.Discrete(8)
        # Global-planning observation: full occupancy grid (downsampled) + agent/goal pose.
        self.obs_map_size = int(obs_map_size)
        if self.obs_map_size < 4:
            raise ValueError("obs_map_size must be >= 4")
        grid_ds = cv2.resize(
            self._grid.astype(np.float32, copy=False),
            dsize=(int(self.obs_map_size), int(self.obs_map_size)),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32, copy=False)
        self._obs_grid_flat = (2.0 * grid_ds.reshape(-1) - 1.0).astype(np.float32, copy=False)

        obs_dim = 5 + int(self.obs_map_size) * int(self.obs_map_size)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self._rng = np.random.default_rng()
        self._agent_xy = np.array(self.start_xy, dtype=np.int32)
        self._steps = 0
        self._dist_to_obstacle = self._compute_dist_to_obstacle()

    @property
    def grid(self) -> np.ndarray:
        return self._grid

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._agent_xy = np.array(self.start_xy, dtype=np.int32)
        self._steps = 0
        obs = self._observe()
        info = {"agent_xy": tuple(self._agent_xy.tolist())}
        return obs, info

    def step(self, action: int):
        self._steps += 1

        dx, dy = _ACTIONS_8[int(action)]
        old_xy = self._agent_xy.copy()
        new_xy = old_xy + np.array([dx, dy], dtype=np.int32)

        boundary_violation = not self._in_bounds(new_xy[0], new_xy[1])
        collision = False
        if boundary_violation:
            new_xy = old_xy  # stay in place
        else:
            collision = bool(self._grid[new_xy[1], new_xy[0]])
            if collision:
                new_xy = old_xy  # stay in place on collision

        self._agent_xy = new_xy

        reached = self._reached_goal()
        truncated = self._steps >= self.max_steps
        terminated = reached or (self.terminate_on_collision and collision)

        reward = self._reward(
            reached=reached,
            boundary_violation=boundary_violation,
            collision=collision,
        )

        obs = self._observe()
        info = {
            "agent_xy": tuple(self._agent_xy.tolist()),
            "boundary_violation": boundary_violation,
            "collision": collision,
            "reached": reached,
            "steps": self._steps,
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self._width and 0 <= y < self._height

    def _reached_goal(self) -> bool:
        return int(self._agent_xy[0]) == self.goal_xy[0] and int(self._agent_xy[1]) == self.goal_xy[1]

    def _distance_to_goal(self) -> float:
        dx = float(self._agent_xy[0] - self.goal_xy[0]) * float(self.cell_size)
        dy = float(self._agent_xy[1] - self.goal_xy[1]) * float(self.cell_size)
        return float(np.sqrt(dx * dx + dy * dy))

    def _min_obstacle_distance_cells(self) -> float:
        ax, ay = int(self._agent_xy[0]), int(self._agent_xy[1])
        return float(self._dist_to_obstacle[ay, ax])

    def _min_obstacle_distance(self) -> float:
        return float(self._min_obstacle_distance_cells()) * float(self.cell_size)

    def _compute_dist_to_obstacle(self) -> np.ndarray:
        # cv2.distanceTransform computes distance to nearest zero pixel, so we make
        # obstacles zeros and free space non-zero. Use top-left origin for OpenCV,
        # then flip back to y=0 bottom.
        grid_top = self._grid[::-1, :]
        free = (grid_top == 0).astype(np.uint8) * 255
        dist_top = cv2.distanceTransform(
            free, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE
        ).astype(np.float32)

        # Convert center-to-center distances to approximate clearance from the agent
        # (assumed at the cell center) to the obstacle cell boundary.
        dist_top = np.maximum(0.0, dist_top - 0.5).astype(np.float32, copy=False)
        return dist_top[::-1, :].astype(np.float32, copy=False)

    def _ray_distances(self) -> np.ndarray:
        ax, ay = int(self._agent_xy[0]), int(self._agent_xy[1])
        sr = self.sensor_range
        distances = np.zeros((8,), dtype=np.float32)
        for i, (dx, dy) in enumerate(_ACTIONS_8):
            d = 0
            x, y = ax, ay
            for step in range(1, sr + 1):
                x = ax + int(dx) * step
                y = ay + int(dy) * step
                if not self._in_bounds(x, y):
                    break
                if self._grid[y, x] == 1:
                    break
                d = step
            distances[i] = float(d) / float(sr)
        return distances

    def _observe(self) -> np.ndarray:
        ax, ay = int(self._agent_xy[0]), int(self._agent_xy[1])
        gx, gy = self.goal_xy
        # Normalize to [-1, 1] (stable for MLPs).
        ax_n = 2.0 * (ax / max(1, self._width - 1)) - 1.0
        ay_n = 2.0 * (ay / max(1, self._height - 1)) - 1.0
        gx_n = 2.0 * (gx / max(1, self._width - 1)) - 1.0
        gy_n = 2.0 * (gy / max(1, self._height - 1)) - 1.0

        td = self._distance_to_goal()
        diag = float(np.sqrt((self._width - 1) ** 2 + (self._height - 1) ** 2)) * float(self.cell_size)
        td01 = float(td / max(1e-6, diag))
        td_n = float(np.clip(2.0 * td01 - 1.0, -1.0, 1.0))

        obs = np.concatenate(
            [np.array([ax_n, ay_n, gx_n, gy_n, td_n], dtype=np.float32), self._obs_grid_flat],
        )
        return obs

    def _reward(self, *, reached: bool, boundary_violation: bool, collision: bool) -> float:
        r = 0.0
        if reached:
            r += float(self.reward.lambda_target)

        # Heuristic distance shaping (Eq. 11)
        r += float(self.reward.lambda_distance) * float(self._distance_to_goal())

        if boundary_violation:
            r += float(self.reward.lambda_boundary)

        # Obstacle proximity / collision penalty (Eq. 13 surrogate)
        if collision:
            r += float(self.reward.lambda_obstacle)
        else:
            od = self._min_obstacle_distance()
            if od < float(self.safe_distance):
                r += float(self.reward.lambda_obstacle)

        return float(r)


@dataclass(frozen=True)
class BicycleModelParams:
    dt: float = 0.05
    wheelbase_m: float = 0.6

    v_max_m_s: float = 2.0
    a_max_m_s2: float = 1.5

    delta_max_rad: float = math.radians(27.0)
    omega_max_rad_s: float = 1.223
    delta_dot_max_rad_s: float = math.radians(60.0)


def _symmetric_action_levels(max_abs: float, n: int, *, grid_power: float) -> np.ndarray:
    n_i = int(n)
    if n_i == 1:
        return np.array([0.0], dtype=np.float32)
    if n_i == 2:
        raise ValueError("n=2 cannot include 0 while preserving both +/-; use an odd number >= 3")
    p = float(grid_power)
    if not (p > 0.0):
        raise ValueError("grid_power must be > 0")
    u = np.linspace(-1.0, 1.0, num=n_i, dtype=np.float32)
    levels = np.sign(u) * np.power(np.abs(u), p, dtype=np.float32)
    vals = levels * float(max_abs)
    vals[int(np.argmin(np.abs(vals)))] = 0.0
    return vals.astype(np.float32, copy=False)


def build_ackermann_action_table_35(*, delta_dot_max_rad_s: float, a_max_m_s2: float) -> np.ndarray:
    """Returns (35, 2) array with columns [delta_dot(rad/s), a(m/s^2)]."""
    return build_ackermann_action_table(
        delta_dot_max_rad_s=float(delta_dot_max_rad_s),
        a_max_m_s2=float(a_max_m_s2),
        n_delta_dots=7,
        n_accels=5,
    )


def build_ackermann_action_table(
    *,
    delta_dot_max_rad_s: float,
    a_max_m_s2: float,
    n_delta_dots: int = 7,
    n_accels: int = 5,
    grid_power: float = 1.0,
) -> np.ndarray:
    """Return (n_delta_dots*n_accels, 2) array with columns [delta_dot(rad/s), a(m/s^2)]."""

    n_dd = int(n_delta_dots)
    n_a = int(n_accels)
    if n_dd < 1:
        raise ValueError("n_delta_dots must be >= 1")
    if n_a < 1:
        raise ValueError("n_accels must be >= 1")

    dd = float(delta_dot_max_rad_s)
    aa = float(a_max_m_s2)
    delta_dots = _symmetric_action_levels(float(dd), int(n_dd), grid_power=float(grid_power))
    accels = _symmetric_action_levels(float(aa), int(n_a), grid_power=float(grid_power))
    d_grid, a_grid = np.meshgrid(delta_dots, accels, indexing="ij")
    table = np.stack([d_grid, a_grid], axis=-1).reshape(-1, 2).astype(np.float32, copy=False)
    return table


def wrap_angle_rad(x: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return float((float(x) + math.pi) % (2.0 * math.pi) - math.pi)


def bicycle_integrate_one_step(
    *,
    x_m: float,
    y_m: float,
    psi_rad: float,
    v_m_s: float,
    delta_rad: float,
    delta_dot_rad_s: float,
    a_m_s2: float,
    params: BicycleModelParams,
) -> tuple[float, float, float, float, float]:
    """Rear-axle center bicycle model with one-step Euler integration."""
    dt = float(params.dt)
    v_next = float(np.clip(v_m_s + float(a_m_s2) * dt, -float(params.v_max_m_s), float(params.v_max_m_s)))

    delta_unclipped = float(delta_rad) + float(delta_dot_rad_s) * dt
    delta_lim = float(params.delta_max_rad)
    delta_next = float(np.clip(delta_unclipped, -float(delta_lim), +float(delta_lim)))

    x_next = float(x_m) + v_next * math.cos(float(psi_rad)) * dt
    y_next = float(y_m) + v_next * math.sin(float(psi_rad)) * dt
    psi_next = wrap_angle_rad(float(psi_rad) + (v_next / float(params.wheelbase_m)) * math.tan(delta_next) * dt)
    return x_next, y_next, psi_next, v_next, delta_next


def min_steps_to_cover_distance_m(
    distance_m: float,
    *,
    dt: float,
    v_max_m_s: float,
    a_max_m_s2: float,
    v0_m_s: float = 0.0,
) -> int:
    """Minimum steps needed to cover `distance_m` along a straight line.

    Uses the same discrete update as the environment speed integrator for forward motion:
        v_{k+1} = clip(v_k + a_max * dt, 0, v_max)
        x_{k+1} = x_k + v_{k+1} * dt
    """
    dist = max(0.0, float(distance_m))
    if dist <= 0.0:
        return 0

    dt_ = float(dt)
    if not (dt_ > 0.0):
        raise ValueError("dt must be > 0")
    v_max = float(v_max_m_s)
    if not (v_max > 0.0):
        raise ValueError("v_max_m_s must be > 0")
    a_max = float(a_max_m_s2)
    if not (a_max > 0.0):
        raise ValueError("a_max_m_s2 must be > 0")

    v = max(0.0, float(v0_m_s))
    covered = 0.0
    steps = 0
    while covered < dist:
        steps += 1
        v = min(v_max, v + a_max * dt_)
        covered += v * dt_
        if steps > 1_000_000:
            raise RuntimeError("min_steps_to_cover_distance_m exceeded step limit; check inputs.")
    return int(steps)


@dataclass(frozen=True)
class TwoCircleFootprint:
    radius_m: float = float(math.hypot(0.740 / 2.0, 0.924 / 4.0))
    x1_m: float = float((0.6 / 2.0) - (0.924 / 4.0))
    x2_m: float = float((0.6 / 2.0) + (0.924 / 4.0))


def compute_edt_distance_m(grid_y0_bottom: np.ndarray, *, cell_size_m: float) -> np.ndarray:
    """EDT distance (meters) from each cell center to the nearest obstacle cell center."""
    grid_top = grid_y0_bottom[::-1, :]
    free = (grid_top == 0).astype(np.uint8) * 255
    dist_top = cv2.distanceTransform(
        free, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE
    ).astype(np.float32)
    return (dist_top[::-1, :] * float(cell_size_m)).astype(np.float32, copy=False)


def bilinear_sample_2d(arr: np.ndarray, *, x: float, y: float, default: float = float("inf")) -> float:
    """Bilinear sample on a (H, W) array using x/y in index coordinates."""
    h, w = arr.shape
    if not (0.0 <= x <= (w - 1) and 0.0 <= y <= (h - 1)):
        return float(default)
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    fx = float(x - x0)
    fy = float(y - y0)

    v00 = float(arr[y0, x0])
    v10 = float(arr[y0, x1])
    v01 = float(arr[y1, x0])
    v11 = float(arr[y1, x1])
    v0 = v00 * (1.0 - fx) + v10 * fx
    v1 = v01 * (1.0 - fx) + v11 * fx
    return float(v0 * (1.0 - fy) + v1 * fy)


def bilinear_sample_2d_finite(
    arr: np.ndarray,
    *,
    x: float,
    y: float,
    fill_value: float,
    default: float | None = None,
) -> float:
    """Bilinear sample that replaces non-finite corner values with `fill_value`.

    Useful when arrays may contain non-finite values (for example masked regions):
    plain bilinear interpolation would otherwise propagate `inf`/`nan`.
    """

    h, w = arr.shape
    if not (0.0 <= x <= (w - 1) and 0.0 <= y <= (h - 1)):
        return float(fill_value if default is None else default)

    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    fx = float(x - x0)
    fy = float(y - y0)

    v00 = float(arr[y0, x0])
    v10 = float(arr[y0, x1])
    v01 = float(arr[y1, x0])
    v11 = float(arr[y1, x1])

    fv = float(fill_value)
    if not math.isfinite(v00):
        v00 = fv
    if not math.isfinite(v10):
        v10 = fv
    if not math.isfinite(v01):
        v01 = fv
    if not math.isfinite(v11):
        v11 = fv

    v0 = v00 * (1.0 - fx) + v10 * fx
    v1 = v01 * (1.0 - fx) + v11 * fx
    return float(v0 * (1.0 - fy) + v1 * fy)


def bilinear_sample_2d_vec(
    arr: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    default: float = float("inf"),
) -> np.ndarray:
    """Vectorized bilinear sampling on a (H, W) array using x/y in index coordinates."""
    h, w = arr.shape
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)

    if h == 0 or w == 0:
        return np.full_like(xv, float(default), dtype=np.float64)

    mask = (xv >= 0.0) & (xv <= float(w - 1)) & (yv >= 0.0) & (yv <= float(h - 1))
    x_c = np.clip(xv, 0.0, float(w - 1))
    y_c = np.clip(yv, 0.0, float(h - 1))

    x0 = np.floor(x_c).astype(np.int32, copy=False)
    y0 = np.floor(y_c).astype(np.int32, copy=False)
    x1 = np.minimum(x0 + 1, int(w - 1)).astype(np.int32, copy=False)
    y1 = np.minimum(y0 + 1, int(h - 1)).astype(np.int32, copy=False)

    fx = x_c - x0.astype(np.float64, copy=False)
    fy = y_c - y0.astype(np.float64, copy=False)

    v00 = arr[y0, x0].astype(np.float64, copy=False)
    v10 = arr[y0, x1].astype(np.float64, copy=False)
    v01 = arr[y1, x0].astype(np.float64, copy=False)
    v11 = arr[y1, x1].astype(np.float64, copy=False)

    v0 = v00 * (1.0 - fx) + v10 * fx
    v1 = v01 * (1.0 - fx) + v11 * fx
    out = v0 * (1.0 - fy) + v1 * fy
    return np.where(mask, out, float(default)).astype(np.float64, copy=False)


def bilinear_sample_2d_finite_vec(
    arr: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    fill_value: float,
    default: float | None = None,
) -> np.ndarray:
    """Vectorized bilinear sampling that replaces non-finite corner values with `fill_value`."""
    h, w = arr.shape
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)

    if h == 0 or w == 0:
        fv = float(fill_value if default is None else default)
        return np.full_like(xv, fv, dtype=np.float64)

    mask = (xv >= 0.0) & (xv <= float(w - 1)) & (yv >= 0.0) & (yv <= float(h - 1))
    x_c = np.clip(xv, 0.0, float(w - 1))
    y_c = np.clip(yv, 0.0, float(h - 1))

    x0 = np.floor(x_c).astype(np.int32, copy=False)
    y0 = np.floor(y_c).astype(np.int32, copy=False)
    x1 = np.minimum(x0 + 1, int(w - 1)).astype(np.int32, copy=False)
    y1 = np.minimum(y0 + 1, int(h - 1)).astype(np.int32, copy=False)

    fx = x_c - x0.astype(np.float64, copy=False)
    fy = y_c - y0.astype(np.float64, copy=False)

    v00 = arr[y0, x0].astype(np.float64, copy=False)
    v10 = arr[y0, x1].astype(np.float64, copy=False)
    v01 = arr[y1, x0].astype(np.float64, copy=False)
    v11 = arr[y1, x1].astype(np.float64, copy=False)

    fv = float(fill_value)
    v00 = np.where(np.isfinite(v00), v00, fv)
    v10 = np.where(np.isfinite(v10), v10, fv)
    v01 = np.where(np.isfinite(v01), v01, fv)
    v11 = np.where(np.isfinite(v11), v11, fv)

    v0 = v00 * (1.0 - fx) + v10 * fx
    v1 = v01 * (1.0 - fx) + v11 * fx
    out = v0 * (1.0 - fy) + v1 * fy

    outside = float(fill_value if default is None else default)
    return np.where(mask, out, outside).astype(np.float64, copy=False)


def euclidean_goal_dist_m(
    shape_hw: tuple[int, int],
    *,
    goal_xy: tuple[int, int],
    cell_size_m: float,
) -> np.ndarray:
    """Compute a dense Euclidean goal-distance field in meters."""
    h, w = int(shape_hw[0]), int(shape_hw[1])
    if h <= 0 or w <= 0:
        raise ValueError("shape_hw must be non-empty")
    cell = float(cell_size_m)
    if not (cell > 0.0):
        raise ValueError("cell_size_m must be > 0")

    gx, gy = int(goal_xy[0]), int(goal_xy[1])
    if not (0 <= gx < w and 0 <= gy < h):
        raise ValueError("goal_xy is out of bounds")

    xs = (np.arange(int(w), dtype=np.float32) - float(gx)) * float(cell)
    ys = (np.arange(int(h), dtype=np.float32) - float(gy)) * float(cell)
    x2 = np.square(xs, dtype=np.float32).reshape(1, -1)
    y2 = np.square(ys, dtype=np.float32).reshape(-1, 1)
    dist = np.sqrt(x2 + y2, dtype=np.float32)
    return dist.astype(np.float32, copy=False)


def dijkstra_goal_dist_m(
    shape_hw: tuple[int, int],
    *,
    goal_xy: tuple[int, int],
    obstacle_grid: np.ndarray,
    cell_size_m: float,
) -> np.ndarray:
    """Shortest-path distance field through free space (Dijkstra).

    Returns a (H, W) float32 array. Obstacle/unreachable cells get a large
    finite fallback (diagonal + cell_size) so bilinear interpolation stays safe.
    """
    import heapq
    h, w = int(shape_hw[0]), int(shape_hw[1])
    cell = float(cell_size_m)
    gx, gy = int(goal_xy[0]), int(goal_xy[1])
    INF = float(math.hypot(h, w) * cell + cell)
    dist = np.full((h, w), INF, dtype=np.float64)
    if obstacle_grid[gy, gx]:
        # Goal on obstacle — fall back to Euclidean
        return euclidean_goal_dist_m(shape_hw, goal_xy=goal_xy,
                                     cell_size_m=cell_size_m)
    dist[gy, gx] = 0.0
    pq: list[tuple[float, int, int]] = [(0.0, gx, gy)]
    DIAG = cell * math.sqrt(2.0)
    while pq:
        d, cx, cy = heapq.heappop(pq)
        if d > dist[cy, cx]:
            continue
        for dx, dy in ((-1,0),(1,0),(0,-1),(0,1),
                       (-1,-1),(-1,1),(1,-1),(1,1)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h and not obstacle_grid[ny, nx]:
                nd = d + (DIAG if (dx and dy) else cell)
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    heapq.heappush(pq, (nd, nx, ny))
    return dist.astype(np.float32, copy=False)


class AMRBicycleEnv(gym.Env):
    """Ackermann/bicycle dynamics on a grid occupancy map using EDT for collision + clearance (OD)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        map_spec: MapSpec,
        *,
        max_steps: int = 500,
        cell_size_m: float = 0.1,
        model: BicycleModelParams = BicycleModelParams(),
        footprint: TwoCircleFootprint = TwoCircleFootprint(),
        sensor_range_m: float = 6.0,
        n_sectors: int = 36,
        obs_map_size: int = 12,
        od_cap_m: float = 2.0,
        safe_distance_m: float = 0.20,
        safe_speed_distance_m: float = 0.20,
        # A slightly looser positional tolerance improves robustness with discrete controls
        # and short-horizon safety shields (the paper's 0.30m can be hard to hit exactly).
        goal_tolerance_m: float = 1,
        goal_angle_tolerance_deg: float = 180.0,
        # To count as "reached", require the vehicle to come to a stop and straighten the wheels.
        goal_stop_speed_m_s: float = 0.05,
        goal_stop_delta_deg: float = 1.0,
        reward_k_p: float = 12.0,
        reward_k_t: float = 0.1,
        reward_k_delta: float = 1.5,
        reward_k_a: float = 0.2,
        reward_k_kappa: float = 0.2,
        reward_k_len: float = 0.0,
        reward_k_o: float = 1.5,
        reward_k_v: float = 2.0,
        reward_k_c: float = 0.0,
        reward_no_progress_penalty: float = 0.0,
        reward_no_progress_eps_m: float = 0.02,
        reward_idle_speed_m_s: float = 0.08,
        reward_obs_max: float = 10.0,
        # Goal-alignment shaping: once inside the goal region, make "stop + straighten"
        # the dominant optimum (prevents "touch goal then dither until timeout").
        reward_goal_v: float = 0.8,
        reward_goal_delta: float = 0.8,
        reward_goal_time: float = 0.4,
        reward_goal_leave: float = 50.0,
        # Terminal rewards (additive).
        reward_reached: float = 1000.0,
        reward_collision: float = -200.0,
        terminate_on_stuck: bool = True,
        stuck_steps: int = 20,
        stuck_min_disp_m: float = 0.02,
        stuck_min_speed_m_s: float = 0.05,
        stuck_penalty: float = 300.0,
        reverse_v_min_m_s: float = 0.10,
        reverse_risk_od_m: float | None = None,
        reverse_no_progress_steps: int | None = None,
        reverse_progress_eps_m: float = 1e-3,
        goal_admissible_relax_factor: float = 1.5,
        action_delta_dot_bins: int = 7,
        action_accel_bins: int = 5,
        action_grid_power: float = 1.0,
        # CBF-aware reward shaping (v8p2) — set c_prog/c_cbf > 0 to activate.
        reward_c_prog: float = 0.0,
        reward_c_cbf: float = 0.0,
        cbf_h_max: float = 2.0,
        gamma: float = 0.99,
        # Exponential potential-based shaping (v8p3)
        reward_potential_base: float = 0.0,
        reward_potential_bias: float = 0.0,
        # A*-guided reward shaping (v9p1)
        reward_k_astar_dev: float = 0.0,
        reward_astar_corridor_m: float = 1.5,
        reward_astar_progress: bool = False,
        # Dijkstra obstacle-aware progress (v9p2)
        reward_dijkstra_progress: bool = False,
    ) -> None:
        super().__init__()

        self.map_spec = map_spec
        self._grid = map_spec.obstacle_grid().astype(np.uint8, copy=False)  # (H, W), y=0 bottom
        self._height, self._width = self._grid.shape
        xs_i = np.arange(self._width, dtype=np.int32).reshape(1, -1)
        ys_i = np.arange(self._height, dtype=np.int32).reshape(-1, 1)
        self._edge_dist_cells = np.minimum(
            np.minimum(xs_i, int(self._width - 1) - xs_i),
            np.minimum(ys_i, int(self._height - 1) - ys_i),
        ).astype(np.int32, copy=False)
        self._canonical_start_xy = (int(map_spec.start_xy[0]), int(map_spec.start_xy[1]))
        self._canonical_goal_xy = (int(map_spec.goal_xy[0]), int(map_spec.goal_xy[1]))
        self.start_xy = (int(self._canonical_start_xy[0]), int(self._canonical_start_xy[1]))
        self.goal_xy = (int(self._canonical_goal_xy[0]), int(self._canonical_goal_xy[1]))

        self.max_steps = int(max_steps)
        self.cell_size_m = float(cell_size_m)
        if not (self.cell_size_m > 0):
            raise ValueError("cell_size_m must be > 0")

        self.model = model
        self.footprint = footprint
        self.sensor_range_m = float(sensor_range_m)
        if not (self.sensor_range_m > 0):
            raise ValueError("sensor_range_m must be > 0")
        self.n_sectors = int(n_sectors)
        if self.n_sectors < 1:
            raise ValueError("n_sectors must be >= 1")
        self.obs_map_size = int(obs_map_size)
        if self.obs_map_size < 4:
            raise ValueError("obs_map_size must be >= 4")
        self.od_cap_m = float(od_cap_m)
        if not (self.od_cap_m > 0):
            raise ValueError("od_cap_m must be > 0")
        self.safe_distance_m = float(safe_distance_m)
        if not (self.safe_distance_m > 0):
            raise ValueError("safe_distance_m must be > 0")
        self.safe_speed_distance_m = float(safe_speed_distance_m)
        if not (self.safe_speed_distance_m > 0):
            raise ValueError("safe_speed_distance_m must be > 0")
        if float(self.safe_speed_distance_m) < float(self.safe_distance_m):
            raise ValueError("safe_speed_distance_m must be >= safe_distance_m")
        self.goal_tolerance_m = float(goal_tolerance_m)
        if not (self.goal_tolerance_m > 0):
            raise ValueError("goal_tolerance_m must be > 0")
        self.goal_angle_tolerance_rad = float(math.radians(float(goal_angle_tolerance_deg)))
        if not (0.0 < self.goal_angle_tolerance_rad <= math.pi):
            raise ValueError("goal_angle_tolerance_deg must be in (0, 180]")
        self.goal_stop_speed_m_s = float(goal_stop_speed_m_s)
        if self.goal_stop_speed_m_s < 0.0:
            raise ValueError("goal_stop_speed_m_s must be >= 0")
        self.goal_stop_delta_rad = float(math.radians(float(goal_stop_delta_deg)))
        if self.goal_stop_delta_rad < 0.0:
            raise ValueError("goal_stop_delta_deg must be >= 0")
        self.goal_stop_delta_rad = min(self.goal_stop_delta_rad, float(self.model.delta_max_rad))

        # Precompute EDT + distance fields once (forest maps are static).
        self._eps_cell_m = float(math.sqrt(2.0) * 0.5 * self.cell_size_m)
        self._dist_m = compute_edt_distance_m(self._grid, cell_size_m=self.cell_size_m)
        self._diag_m = float(
            math.hypot(float(self._width - 1) * self.cell_size_m, float(self._height - 1) * self.cell_size_m)
        )

        # Treat the world boundary as an obstacle for both collision checking and sensing.
        max_x = float(self._width - 1) * self.cell_size_m
        max_y = float(self._height - 1) * self.cell_size_m
        xs = (np.arange(self._width, dtype=np.float32) * float(self.cell_size_m)).reshape(1, -1)
        ys = (np.arange(self._height, dtype=np.float32) * float(self.cell_size_m)).reshape(-1, 1)
        boundary_dist = np.minimum(
            np.minimum(xs, float(max_x) - xs),
            np.minimum(ys, float(max_y) - ys),
        ).astype(np.float32, copy=False)
        self._dist_m = np.minimum(self._dist_m, boundary_dist).astype(np.float32, copy=False)

        # Traversability used for distance-based shaping and curriculum sampling.
        #
        # Use *collision-free* clearance (r + eps_cell). The reward's OD-based safe-distance terms
        # handle additional margin; making the traversable set too conservative can disconnect the
        # free space and remove useful progress gradients.
        self._clearance_thr_m = float(self.footprint.radius_m) + float(self._eps_cell_m)
        self._traversable_base = (self._dist_m > float(self._clearance_thr_m)).astype(bool, copy=False)
        # Ensure the canonical start/goal cells are always treated as traversable.
        self._traversable_base[self._canonical_start_xy[1], self._canonical_start_xy[0]] = True
        self._traversable_base[self._canonical_goal_xy[1], self._canonical_goal_xy[0]] = True

        # Candidate free cells for random start/goal sampling.
        free_y, free_x = np.where(self._traversable_base)
        self._rand_free_xy = np.stack([free_x, free_y], axis=1).astype(np.int32, copy=False)

        # Goal-dependent fields (distance field + curriculum candidates).
        # Must set dijkstra flag before _set_goal_xy uses it.
        self.reward_dijkstra_progress = bool(reward_dijkstra_progress)
        self._set_goal_xy(self.goal_xy)
        # Start-dependent normalization.
        self._update_start_dependent_fields(start_xy=self.start_xy)

        # Sanity-check horizon: use start distance to goal.
        min_steps = min_steps_to_cover_distance_m(
            max(0.0, float(self._goal_dist_norm_m) - float(self.goal_tolerance_m)),
            dt=float(self.model.dt),
            v_max_m_s=float(self.model.v_max_m_s),
            a_max_m_s2=float(self.model.a_max_m_s2),
            v0_m_s=0.0,
        )
        if int(self.max_steps) < int(min_steps):
            raise ValueError(
                f"max_steps={self.max_steps} is too small for forest env {self.map_spec.name!r} "
                f"(dist≈{self._goal_dist_norm_m:.2f}m with v_max={self.model.v_max_m_s:.2f}m/s, "
                f"a_max={self.model.a_max_m_s2:.2f}m/s², dt={self.model.dt:.3f}s). "
                f"Need at least {min_steps} steps (increase --max-steps)."
            )

        self.reward_k_p = float(reward_k_p)
        self.reward_k_t = float(reward_k_t)
        self.reward_k_delta = float(reward_k_delta)
        self.reward_k_a = float(reward_k_a)
        self.reward_k_kappa = float(reward_k_kappa)
        self.reward_k_len = float(reward_k_len)
        self.reward_k_o = float(reward_k_o)
        self.reward_k_v = float(reward_k_v)
        self.reward_k_c = float(reward_k_c)
        self.reward_no_progress_penalty = float(reward_no_progress_penalty)
        if not (self.reward_no_progress_penalty >= 0.0):
            raise ValueError("reward_no_progress_penalty must be >= 0")
        self.reward_no_progress_eps_m = float(reward_no_progress_eps_m)
        if not (self.reward_no_progress_eps_m >= 0.0):
            raise ValueError("reward_no_progress_eps_m must be >= 0")
        self.reward_idle_speed_m_s = float(reward_idle_speed_m_s)
        if not (self.reward_idle_speed_m_s >= 0.0):
            raise ValueError("reward_idle_speed_m_s must be >= 0")
        self.reward_obs_max = float(reward_obs_max)
        self.reward_goal_v = float(reward_goal_v)
        self.reward_goal_delta = float(reward_goal_delta)
        self.reward_goal_time = float(reward_goal_time)
        self.reward_goal_leave = float(reward_goal_leave)
        self.reward_reached = float(reward_reached)
        self.reward_collision = float(reward_collision)
        self.reward_eps = 1e-3
        # CBF-aware reward shaping (v8p2)
        self.reward_c_prog = float(reward_c_prog)
        self.reward_c_cbf = float(reward_c_cbf)
        self.cbf_h_max = float(cbf_h_max)
        self._gamma = float(gamma)
        self.reward_potential_base = float(reward_potential_base)
        self.reward_potential_bias = float(reward_potential_bias)
        # A*-guided reward shaping (v9p1)
        self.reward_k_astar_dev = float(reward_k_astar_dev)
        self.reward_astar_corridor_m = float(reward_astar_corridor_m)
        self.reward_astar_progress = bool(reward_astar_progress)
        self.reward_dijkstra_progress = bool(reward_dijkstra_progress)
        self._initial_dist = None  # set in reset()
        self._cbf_collision_thr = float(self.footprint.radius_m) + float(self._eps_cell_m)
        self.terminate_on_stuck = bool(terminate_on_stuck)
        self.stuck_steps = int(stuck_steps)
        if self.stuck_steps < 1:
            raise ValueError("stuck_steps must be >= 1")
        self.stuck_min_disp_m = float(stuck_min_disp_m)
        if not (self.stuck_min_disp_m >= 0.0):
            raise ValueError("stuck_min_disp_m must be >= 0")
        self.stuck_min_speed_m_s = float(stuck_min_speed_m_s)
        if not (self.stuck_min_speed_m_s >= 0.0):
            raise ValueError("stuck_min_speed_m_s must be >= 0")
        self.stuck_penalty = float(stuck_penalty)
        if not (self.stuck_penalty >= 0.0):
            raise ValueError("stuck_penalty must be >= 0")

        self.reverse_v_min_m_s = float(reverse_v_min_m_s)
        if not (self.reverse_v_min_m_s >= 0.0):
            raise ValueError("reverse_v_min_m_s must be >= 0")
        if reverse_risk_od_m is None:
            reverse_risk_od_m = float(self.safe_distance_m)
        self.reverse_risk_od_m = float(reverse_risk_od_m)
        if not (self.reverse_risk_od_m >= 0.0):
            raise ValueError("reverse_risk_od_m must be >= 0")
        if reverse_no_progress_steps is None:
            reverse_no_progress_steps = max(3, int(self.stuck_steps) // 2)
        self.reverse_no_progress_steps = int(reverse_no_progress_steps)
        if self.reverse_no_progress_steps < 0:
            raise ValueError("reverse_no_progress_steps must be >= 0")
        self.reverse_progress_eps_m = float(reverse_progress_eps_m)
        if not (self.reverse_progress_eps_m >= 0.0):
            raise ValueError("reverse_progress_eps_m must be >= 0")
        self.goal_admissible_relax_factor = float(goal_admissible_relax_factor)
        if not (self.goal_admissible_relax_factor >= 1.0):
            raise ValueError("goal_admissible_relax_factor must be >= 1.0")

        self.action_delta_dot_bins = int(action_delta_dot_bins)
        if self.action_delta_dot_bins < 1:
            raise ValueError("action_delta_dot_bins must be >= 1")
        self.action_accel_bins = int(action_accel_bins)
        if self.action_accel_bins < 1:
            raise ValueError("action_accel_bins must be >= 1")
        self.action_grid_power = float(action_grid_power)
        if not (self.action_grid_power > 0.0):
            raise ValueError("action_grid_power must be > 0")

        self.action_table = build_ackermann_action_table(
            delta_dot_max_rad_s=float(model.delta_dot_max_rad_s),
            a_max_m_s2=float(model.a_max_m_s2),
            n_delta_dots=int(self.action_delta_dot_bins),
            n_accels=int(self.action_accel_bins),
            grid_power=float(self.action_grid_power),
        )
        self.action_space = gym.spaces.Discrete(int(self.action_table.shape[0]))

        # Global-planning observation: agent/goal pose + downsampled (static) maps.
        occ_ds = cv2.resize(
            self._grid.astype(np.float32, copy=False),
            dsize=(int(self.obs_map_size), int(self.obs_map_size)),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32, copy=False)
        self._obs_occ_flat = (2.0 * occ_ds.reshape(-1) - 1.0).astype(np.float32, copy=False)

        obs_dim = 10 + int(self.obs_map_size) * int(self.obs_map_size)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self._rng = np.random.default_rng()
        self._steps = 0
        self._no_progress_steps = 0
        self._in_goal_region_prev = False

        self._x_m = float(self.start_xy[0]) * self.cell_size_m
        self._y_m = float(self.start_xy[1]) * self.cell_size_m
        self._psi_rad = 0.0
        self._v_m_s = 0.0
        self._delta_rad = 0.0
        self._prev_delta_dot = 0.0
        self._prev_a = 0.0
        self._last_od_m = 0.0
        self._last_collision = False
        self._stuck_pos_history: list[tuple[float, float]] = []
        self._in_goal_region_prev: bool = False
        self._ha_path_cache: dict[tuple[int, int, int, int], list[tuple[float, float]]] = {}
        self._astar_path_cache: dict[tuple[int, int, int, int, int, int], tuple[list[tuple[float, float]], list[float]]] = {}
        self._ha_progress_idx: int = 0
        self._ha_start_xy: tuple[int, int] = self.start_xy
        self._astar_progress_idx: int = 0
        self._astar_start_xy: tuple[int, int] = self.start_xy
        # A*-guided shaping runtime state (v9p1)
        self._astar_ref_xy: np.ndarray | None = None  # (N, 2) continuous coords in meters
        self._astar_ref_cumlen: np.ndarray | None = None  # (N,) cumulative path length
        self._astar_ref_total_len: float = 0.0

    def _goal_pose_reached(self, *, d_goal_m: float, alpha_rad: float) -> bool:
        return (float(d_goal_m) <= float(self.goal_tolerance_m)) and (
            abs(float(alpha_rad)) <= float(self.goal_angle_tolerance_rad)
        )

    def _goal_stop_reached(self, *, v_m_s: float, delta_rad: float) -> bool:
        return (abs(float(v_m_s)) <= float(self.goal_stop_speed_m_s)) and (
            abs(float(delta_rad)) <= float(self.goal_stop_delta_rad)
        )

    def _reverse_unlocked(self, *, min_od_m: float) -> bool:
        # Avoid opening reverse during the near-goal "stop/straighten" phase.
        reached_pose = bool(
            self._goal_pose_reached(
                d_goal_m=float(self._distance_to_goal_m()),
                alpha_rad=float(self._goal_relative_angle_rad()),
            )
        )
        if reached_pose:
            return False

        od_m = float(self._last_od_m)
        od_thr = max(float(min_od_m), float(self.reverse_risk_od_m))
        near_obstacle = math.isfinite(od_m) and (od_m >= 0.0) and (od_m < float(od_thr))
        stagnating = int(self._no_progress_steps) >= int(self.reverse_no_progress_steps)
        return bool(near_obstacle or stagnating)

    @property
    def grid(self) -> np.ndarray:
        return self._grid

    def _in_bounds_xy(self, xy: tuple[int, int]) -> bool:
        x, y = int(xy[0]), int(xy[1])
        return 0 <= x < int(self._width) and 0 <= y < int(self._height)

    def _set_goal_xy(self, goal_xy: tuple[int, int]) -> None:
        gx, gy = int(goal_xy[0]), int(goal_xy[1])
        if not self._in_bounds_xy((gx, gy)):
            raise ValueError("goal_xy is out of bounds")

        self.goal_xy = (int(gx), int(gy))

        if self.reward_dijkstra_progress:
            self._goal_dist_m = dijkstra_goal_dist_m(
                (int(self._height), int(self._width)),
                goal_xy=self.goal_xy,
                obstacle_grid=self._grid,
                cell_size_m=self.cell_size_m,
            )
        else:
            self._goal_dist_m = euclidean_goal_dist_m(
                (int(self._height), int(self._width)),
                goal_xy=self.goal_xy,
                cell_size_m=self.cell_size_m,
            )
        self._goal_dist_fill_m = float(self._diag_m) + float(self.cell_size_m)

        # Curriculum: candidate start cells (reachable under clearance + minimum goal distance).
        self._curriculum_min_dist_m = float(max(self.goal_tolerance_m + self.cell_size_m, 1.0))
        cand_mask = self._traversable_base.astype(bool, copy=True)
        # Exclude the goal cell (too trivial) and any cells inside the goal tolerance.
        cand_mask[int(gy), int(gx)] = False
        cand_mask &= self._goal_dist_m >= float(self._curriculum_min_dist_m)

        cand_y, cand_x = np.where(cand_mask)
        self._curriculum_start_xy = np.stack([cand_x, cand_y], axis=1).astype(np.int32, copy=False)
        self._curriculum_start_dists_m = self._goal_dist_m[cand_y, cand_x].astype(np.float32, copy=False)

    def _update_start_dependent_fields(self, *, start_xy: tuple[int, int]) -> None:
        sx, sy = int(start_xy[0]), int(start_xy[1])
        if not self._in_bounds_xy((sx, sy)):
            raise ValueError("start_xy is out of bounds")

        # Normalize by the anchor start pose (two-circle footprint),
        # not just the rear-axle cell, so shaping remains meaningful near obstacles.
        start_dist_cell = float(self._goal_dist_m[int(sy), int(sx)])
        start_x_m = float(sx) * self.cell_size_m
        start_y_m = float(sy) * self.cell_size_m
        dx0 = float(self.goal_xy[0] - int(sx)) * self.cell_size_m
        dy0 = float(self.goal_xy[1] - int(sy)) * self.cell_size_m
        psi0 = wrap_angle_rad(math.atan2(dy0, dx0))
        start_dist_pose = self._goal_dist_pose_m(start_x_m, start_y_m, psi0)
        self._goal_dist_norm_m = float(max(self._diag_m, start_dist_cell, start_dist_pose))
        if not math.isfinite(self._goal_dist_norm_m):
            raise ValueError(
                f"Forest map {self.map_spec.name!r} is unreachable under the clearance constraint; "
                "regenerate the map or reduce obstacle density."
            )

        # Curriculum distance anchor (used only for curriculum sampling by distance bands).
        start_dist = float(start_dist_pose if math.isfinite(start_dist_pose) else start_dist_cell)
        self._curriculum_start_dist_m = float(start_dist)

    def _sample_random_start_goal(
        self,
        *,
        min_dist_m: float,
        max_dist_m: float | None,
        fixed_prob: float,
        tries: int,
        edge_margin_m: float,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        if self._rand_free_xy.size == 0:
            return self._canonical_start_xy, self._canonical_goal_xy

        p_fixed = float(np.clip(float(fixed_prob), 0.0, 1.0))
        if float(self._rng.random()) < p_fixed:
            self._set_goal_xy(self._canonical_goal_xy)
            self._update_start_dependent_fields(start_xy=self._canonical_start_xy)
            return self._canonical_start_xy, self._canonical_goal_xy

        min_dist = max(0.0, float(min_dist_m))
        max_dist = None if max_dist_m is None else max(0.0, float(max_dist_m))
        n_tries = max(1, int(tries))
        edge_margin_cells = 0
        if float(edge_margin_m) > 0.0:
            edge_margin_cells = int(math.ceil(float(edge_margin_m) / float(self.cell_size_m)))
        max_edge_margin = max(0, (min(int(self._width), int(self._height)) // 2) - 1)
        edge_margin_cells = int(np.clip(int(edge_margin_cells), 0, int(max_edge_margin)))

        # Performance note:
        # Sampling "very long" start/goal pairs can be expensive if we pick goals uniformly
        # over the map. This happens in large forests when `min_dist_m` is close to map diameter.
        # edge-margin constraint (e.g., long-suite sampling).
        #
        # To keep sampling tractable, bias goal sampling toward corner regions (within the allowed
        # edge margin) when the constraint is tight. This dramatically increases the probability
        # that a sampled goal has any valid starts with dist>=min_dist.
        goal_xy_choices = self._rand_free_xy
        if edge_margin_cells > 0:
            # `edge_dist_cells` is min distance to any edge; requiring >= edge_margin_cells implies
            # distance to both x-edges and y-edges is >= edge_margin_cells.
            edge_ok = self._edge_dist_cells[goal_xy_choices[:, 1], goal_xy_choices[:, 0]] >= int(edge_margin_cells)
            goal_xy_choices = goal_xy_choices[edge_ok]

        if goal_xy_choices.size > 0 and min_dist > 0.0 and edge_margin_cells > 0:
            w_eff_cells = max(0, (int(self._width) - 1) - 2 * int(edge_margin_cells))
            h_eff_cells = max(0, (int(self._height) - 1) - 2 * int(edge_margin_cells))
            diag_eff_m = float(math.hypot(float(w_eff_cells) * float(self.cell_size_m), float(h_eff_cells) * float(self.cell_size_m)))
            tight = bool(diag_eff_m > 1e-9 and float(min_dist) >= 0.85 * float(diag_eff_m))
            if tight:
                slack_m = max(0.0, float(diag_eff_m) - float(min_dist))
                # Keep a small minimum band so detour-heavy "long" pairs remain sampleable.
                band_m = float(min(5.0, float(slack_m) + 2.0))
                band_cells = int(max(1, int(math.ceil(float(band_m) / float(self.cell_size_m)))))
                x = goal_xy_choices[:, 0].astype(np.int32, copy=False)
                y = goal_xy_choices[:, 1].astype(np.int32, copy=False)
                vx = np.minimum(x, int(self._width - 1) - x)
                vy = np.minimum(y, int(self._height - 1) - y)
                corner = (vx <= int(edge_margin_cells) + int(band_cells)) & (vy <= int(edge_margin_cells) + int(band_cells))
                corner_xy = goal_xy_choices[corner]
                if corner_xy.size > 0:
                    goal_xy_choices = corner_xy

        for _ in range(n_tries):
            xy_src = goal_xy_choices if goal_xy_choices.size > 0 else self._rand_free_xy
            gi = int(self._rng.integers(0, int(xy_src.shape[0])))
            gx, gy = (int(xy_src[gi, 0]), int(xy_src[gi, 1]))

            try:
                self._set_goal_xy((gx, gy))
            except Exception:
                continue

            goal_dists = self._goal_dist_m
            cand_mask = self._traversable_base.astype(bool, copy=True)
            cand_mask[int(gy), int(gx)] = False
            if min_dist > 0.0:
                cand_mask &= goal_dists >= float(min_dist)
            if max_dist is not None and max_dist > 0.0:
                cand_mask &= goal_dists <= float(max_dist)
            if edge_margin_cells > 0:
                cand_mask &= self._edge_dist_cells >= int(edge_margin_cells)

            sy, sx = np.where(cand_mask)
            if sx.size == 0:
                continue

            si = int(self._rng.integers(0, int(sx.size)))
            start_xy = (int(sx[si]), int(sy[si]))

            # Verify the initial pose (heading to goal) is collision-free under the two-circle footprint.
            dx0 = float(gx - int(start_xy[0])) * float(self.cell_size_m)
            dy0 = float(gy - int(start_xy[1])) * float(self.cell_size_m)
            psi0 = wrap_angle_rad(math.atan2(dy0, dx0))
            _od0, coll0 = self._od_and_collision_at_pose_m(
                float(start_xy[0]) * float(self.cell_size_m),
                float(start_xy[1]) * float(self.cell_size_m),
                float(psi0),
            )
            if bool(coll0):
                continue

            # Update normalization anchor to the sampled start.
            try:
                self._update_start_dependent_fields(start_xy=start_xy)
            except Exception:
                continue

            # Reject pairs that cannot be executed within the episode horizon.
            try:
                min_steps = min_steps_to_cover_distance_m(
                    max(0.0, float(self._goal_dist_norm_m) - float(self.goal_tolerance_m)),
                    dt=float(self.model.dt),
                    v_max_m_s=float(self.model.v_max_m_s),
                    a_max_m_s2=float(self.model.a_max_m_s2),
                    v0_m_s=0.0,
                )
            except Exception:
                continue
            if int(self.max_steps) < int(min_steps):
                continue

            return start_xy, (gx, gy)

        # Fallback: canonical pair.
        self._set_goal_xy(self._canonical_goal_xy)
        self._update_start_dependent_fields(start_xy=self._canonical_start_xy)
        return self._canonical_start_xy, self._canonical_goal_xy

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._steps = 0
        self._no_progress_steps = 0

        # Default: canonical fixed start/goal (backwards compatible).
        start_xy = (int(self._canonical_start_xy[0]), int(self._canonical_start_xy[1]))
        goal_xy = (int(self._canonical_goal_xy[0]), int(self._canonical_goal_xy[1]))

        start_override: tuple[int, int] | None = None
        goal_override: tuple[int, int] | None = None
        random_start_goal = False
        rand_min_dist_m = 0.0
        rand_max_dist_m: float | None = None
        rand_fixed_prob = 0.0
        rand_tries = 200
        rand_edge_margin_m = 0.0

        if options:
            if options.get("start_xy") is not None:
                sx, sy = options["start_xy"]
                start_override = (int(sx), int(sy))
            if options.get("goal_xy") is not None:
                gx, gy = options["goal_xy"]
                goal_override = (int(gx), int(gy))
            random_start_goal = bool(options.get("random_start_goal", False))
            rand_min_dist_m = float(options.get("rand_min_dist_m", 0.0))
            max_raw = options.get("rand_max_dist_m", None)
            rand_max_dist_m = None if max_raw is None else float(max_raw)
            rand_fixed_prob = float(options.get("rand_fixed_prob", 0.0))
            rand_tries = int(options.get("rand_tries", 200))
            rand_edge_margin_m = float(options.get("rand_edge_margin_m", 0.0))

        if goal_override is not None:
            goal_xy = (int(goal_override[0]), int(goal_override[1]))

        if random_start_goal and start_override is None and goal_override is None:
            start_xy, goal_xy = self._sample_random_start_goal(
                min_dist_m=float(rand_min_dist_m),
                max_dist_m=rand_max_dist_m,
                fixed_prob=float(rand_fixed_prob),
                tries=int(rand_tries),
                edge_margin_m=float(rand_edge_margin_m),
            )
        else:
            # Ensure env goal-dependent fields match the requested goal.
            if (int(self.goal_xy[0]), int(self.goal_xy[1])) != (int(goal_xy[0]), int(goal_xy[1])):
                self._set_goal_xy(goal_xy)

            if start_override is not None:
                start_xy = (int(start_override[0]), int(start_override[1]))
                # For explicit (start,goal) overrides, normalize by the episode start.
                self._update_start_dependent_fields(start_xy=start_xy)
            else:
                # Keep normalization anchored at the canonical start (same behavior as fixed-start training),
                # even when curriculum samples a different episode start.
                self._update_start_dependent_fields(start_xy=self._canonical_start_xy)

        self.start_xy = (int(start_xy[0]), int(start_xy[1]))
        self.goal_xy = (int(goal_xy[0]), int(goal_xy[1]))

        ha_start_xy = (int(self.start_xy[0]), int(self.start_xy[1]))
        ha_progress_idx = 0
        psi_override: float | None = None
        if (
            options
            and (not bool(random_start_goal))
            and start_override is None
            and goal_override is None
            and options.get("curriculum_progress") is not None
            and len(self._curriculum_start_xy) > 0
        ):
            # Forest curriculum: early episodes start closer to the goal; later episodes gradually
            # shift probability mass back to the canonical start. This prevents a train/test mismatch
            # where the agent never practices the true start state but inference always begins there.
            p = float(options["curriculum_progress"])
            p = float(np.clip(p, 0.0, 1.0))

            # With probability p, use the fixed start (p=1 => always start from SP).
            if float(self._rng.random()) >= float(p):
                band_m = float(options.get("curriculum_band_m", 2.0))
                band_m = max(float(self.cell_size_m), float(band_m))

                hi = float(self._curriculum_min_dist_m) + p * float(
                    max(0.0, float(self._curriculum_start_dist_m) - float(self._curriculum_min_dist_m))
                )
                lo = max(float(self._curriculum_min_dist_m), float(hi) - float(band_m))

                # Prefer sampling starts along the (precomputed) Hybrid A* reference path when available.
                # This keeps curriculum starts on a known feasible corridor and avoids repeatedly
                # re-planning from many random start states (which can time out on large forests).
                chosen_ref_idx: int | None = None
                ref_path = self._hybrid_astar_path(start_xy=self.start_xy)
                if len(ref_path) >= 2:
                    # Sample by reference-path progress (more robust than matching exact distance bands
                    # after rounding continuous Hybrid A* coordinates back to grid cells).
                    max_idx = max(0, int(len(ref_path) - 2))  # exclude last point (goal vicinity)
                    band_steps = max(1, int(round(float(band_m) / float(self.cell_size_m))))
                    target_idx = int(round((1.0 - float(p)) * float(max_idx)))
                    lo_i = max(0, int(target_idx) - int(band_steps))
                    hi_i = min(int(max_idx), int(target_idx) + int(band_steps))

                    ref_idxs: list[int] = []
                    for i in range(int(lo_i), int(hi_i) + 1):
                        px, py = ref_path[int(i)]
                        ix = int(round(float(px)))
                        iy = int(round(float(py)))
                        if not (0 <= ix < self._width and 0 <= iy < self._height):
                            continue
                        c = float(self._goal_dist_m[iy, ix])
                        if not math.isfinite(c) or float(c) < float(self._curriculum_min_dist_m):
                            continue
                        ref_idxs.append(int(i))

                    if ref_idxs:
                        chosen_ref_idx = int(self._rng.choice(ref_idxs))
                        px, py = ref_path[int(chosen_ref_idx)]
                        start_xy = (int(round(float(px))), int(round(float(py))))
                        ha_start_xy = (int(self.start_xy[0]), int(self.start_xy[1]))
                        ha_progress_idx = int(chosen_ref_idx)
                        j = min(int(chosen_ref_idx) + 1, len(ref_path) - 1)
                        px2, py2 = ref_path[int(j)]
                        dx = (float(px2) - float(px)) * float(self.cell_size_m)
                        dy = (float(py2) - float(py)) * float(self.cell_size_m)
                        if abs(float(dx)) + abs(float(dy)) > 1e-9:
                            psi_override = wrap_angle_rad(math.atan2(float(dy), float(dx)))

                if chosen_ref_idx is None:
                    dists = self._curriculum_start_dists_m
                    idxs = np.nonzero((dists >= float(lo)) & (dists <= float(hi)))[0]
                    if idxs.size == 0:
                        idxs = np.nonzero(dists <= float(hi))[0]
                    if idxs.size > 0:
                        j = int(self._rng.choice(idxs))
                        start_xy = (int(self._curriculum_start_xy[j, 0]), int(self._curriculum_start_xy[j, 1]))
                        ha_start_xy = (int(start_xy[0]), int(start_xy[1]))
                        ha_progress_idx = 0

        # Finalize episode start (may differ from the canonical start when curriculum/randomization is used).
        self.start_xy = (int(start_xy[0]), int(start_xy[1]))

        self._x_m = float(start_xy[0]) * self.cell_size_m
        self._y_m = float(start_xy[1]) * self.cell_size_m
        dx = float(self.goal_xy[0] - start_xy[0]) * self.cell_size_m
        dy = float(self.goal_xy[1] - start_xy[1]) * self.cell_size_m
        psi = wrap_angle_rad(math.atan2(dy, dx))
        if psi_override is not None:
            psi = float(psi_override)
        self._psi_rad = float(psi)
        self._v_m_s = 0.0
        self._delta_rad = 0.0
        self._prev_delta_dot = 0.0
        self._prev_a = 0.0
        self._last_od_m, self._last_collision = self._od_and_collision_m()
        self._stuck_pos_history = [(float(self._x_m), float(self._y_m))]
        self._ha_start_xy = (int(ha_start_xy[0]), int(ha_start_xy[1]))
        self._ha_progress_idx = int(ha_progress_idx)
        self._astar_start_xy = (int(ha_start_xy[0]), int(ha_start_xy[1]))
        self._astar_progress_idx = 0

        # Record initial distance for exponential potential shaping (v8p3)
        self._initial_dist = self._distance_to_goal_m()

        # A*-guided shaping: compute reference path once per episode (v9p1)
        self._astar_ref_xy = None
        self._astar_ref_cumlen = None
        self._astar_ref_total_len = 0.0
        if self.reward_k_astar_dev > 0.0 or self.reward_astar_progress:
            self._compute_astar_ref_path()

        obs = self._observe()
        info = {"agent_xy": self._agent_xy_for_plot(), "pose_m": (self._x_m, self._y_m, self._psi_rad)}
        return obs, info

    def _step_with_controls(self, *, delta_dot: float, a: float):
        self._steps += 1

        delta_dot = float(delta_dot)
        a = float(a)
        prev_delta_dot = float(self._prev_delta_dot)
        prev_a = float(self._prev_a)

        x_before = float(self._x_m)
        y_before = float(self._y_m)
        dist_before = self._goal_dist_pose_m(x_before, y_before, float(self._psi_rad))
        d_goal_before = self._distance_to_goal_m()
        delta_before = float(self._delta_rad)

        x_next, y_next, psi_next, v_next, delta_next = bicycle_integrate_one_step(
            x_m=self._x_m,
            y_m=self._y_m,
            psi_rad=self._psi_rad,
            v_m_s=self._v_m_s,
            delta_rad=self._delta_rad,
            delta_dot_rad_s=delta_dot,
            a_m_s2=a,
            params=self.model,
        )

        self._x_m, self._y_m, self._psi_rad, self._v_m_s, self._delta_rad = (
            x_next,
            y_next,
            psi_next,
            v_next,
            delta_next,
        )
        self._last_od_m, self._last_collision = self._od_and_collision_m()

        dist_after = self._goal_dist_pose_m(self._x_m, self._y_m, float(self._psi_rad))
        d_goal_after = self._distance_to_goal_m()
        alpha = self._goal_relative_angle_rad()
        reached_pose = self._goal_pose_reached(d_goal_m=d_goal_after, alpha_rad=alpha)
        reached_stop = self._goal_stop_reached(v_m_s=v_next, delta_rad=delta_next)
        reached = bool(reached_pose and reached_stop)

        collision = bool(self._last_collision)
        od_m = float(self._last_od_m)
        truncated = self._steps >= self.max_steps
        terminated = bool(collision or reached)

        # Track goal-region enter/leave transitions for shaping.
        left_goal_region = bool(self._in_goal_region_prev) and (not bool(reached_pose))
        self._in_goal_region_prev = bool(reached_pose)

        # Track consecutive no-progress steps (used to unlock reverse actions for recovery).
        if math.isfinite(dist_before) and math.isfinite(dist_after):
            progress_m = float(dist_before - dist_after)
        else:
            progress_m = float(d_goal_before - d_goal_after)
        if terminated or truncated or reached_pose:
            self._no_progress_steps = 0
        else:
            if float(progress_m) >= float(self.reverse_progress_eps_m):
                self._no_progress_steps = 0
            else:
                self._no_progress_steps += 1

        # Stuck detection (helps prevent in-place steering jitter / stopping forever).
        #
        # Use *windowed* displacement, not per-step displacement: with dt=0.05s the vehicle can
        # legitimately move <2cm per step at low speeds, so per-step thresholds cause false stuck.
        stuck = False
        # When the pose goal is reached but the stop/straighten condition is not yet satisfied,
        # allow the vehicle to remain in place without triggering the stuck terminator.
        if not (terminated or truncated or reached_pose):
            self._stuck_pos_history.append((float(self._x_m), float(self._y_m)))
            max_hist = int(self.stuck_steps) + 1
            if len(self._stuck_pos_history) > max_hist:
                self._stuck_pos_history.pop(0)

            if len(self._stuck_pos_history) >= max_hist and abs(float(self._v_m_s)) < float(self.stuck_min_speed_m_s):
                x0, y0 = self._stuck_pos_history[0]
                x1, y1 = self._stuck_pos_history[-1]
                disp = float(math.hypot(float(x1) - float(x0), float(y1) - float(y0)))
                if disp < float(self.stuck_min_disp_m):
                    stuck = True
                    if bool(self.terminate_on_stuck):
                        terminated = True

        reward = 0.0
        # Progress reward
        if self.reward_c_prog > 0.0:
            if self.reward_potential_base > 0.0 and self._initial_dist and self._initial_dist > 0:
                # Exponential potential-based shaping (v8p3)
                base = self.reward_potential_base
                bias = self.reward_potential_bias
                d_now = float(dist_after) if math.isfinite(dist_after) else float(d_goal_after)
                d_prev = float(dist_before) if math.isfinite(dist_before) else float(d_goal_before)
                prog_now = max(0.0, min(1.0, 1.0 - d_now / self._initial_dist))
                prog_prev = max(0.0, min(1.0, 1.0 - d_prev / self._initial_dist))
                phi_now = math.exp(base * prog_now)
                phi_prev = math.exp(base * prog_prev)
                reward += self.reward_c_prog * (
                    self._gamma * (phi_now + bias) - (phi_prev + bias))
            else:
                # Linear potential-based shaping (v8p2)
                if math.isfinite(dist_before) and math.isfinite(dist_after):
                    reward += self.reward_c_prog * (float(dist_before) - self._gamma * float(dist_after))
                else:
                    reward += self.reward_c_prog * (float(d_goal_before) - self._gamma * float(d_goal_after))
        else:
            # Legacy k_p progress reward — optionally A*-guided (v9p1)
            if self.reward_astar_progress and self._astar_ref_xy is not None:
                _, rem_before = self._astar_ref_query(x_before, y_before)
                _, rem_after = self._astar_ref_query(float(self._x_m), float(self._y_m))
                if math.isfinite(rem_before) and math.isfinite(rem_after):
                    reward += self.reward_k_p * float(rem_before - rem_after)
                elif math.isfinite(dist_before) and math.isfinite(dist_after):
                    reward += self.reward_k_p * float(dist_before - dist_after)
                else:
                    reward += self.reward_k_p * float(d_goal_before - d_goal_after)
            elif math.isfinite(dist_before) and math.isfinite(dist_after):
                reward += self.reward_k_p * float(dist_before - dist_after)
            else:
                reward += self.reward_k_p * float(d_goal_before - d_goal_after)
        # Time (fast): per-step penalty.
        reward -= self.reward_k_t
        if (not collision) and (not reached):
            if (float(progress_m) < float(self.reward_no_progress_eps_m)) and (
                abs(float(v_next)) < float(self.reward_idle_speed_m_s)
            ):
                reward -= float(self.reward_no_progress_penalty)
        # Smoothness
        reward -= self.reward_k_delta * float(delta_next - delta_before) ** 2
        # Acceleration smoothness should not prevent "getting going" from rest; scale by speed.
        v_scale = (float(v_next) / float(self.model.v_max_m_s)) ** 2
        reward -= self.reward_k_a * float(a - prev_a) ** 2 * float(v_scale)
        # Curvature / large steering penalty
        reward -= self.reward_k_kappa * float(math.tan(delta_next) ** 2)
        # Path length penalty (per-step distance traveled)
        if self.reward_k_len > 0.0:
            step_dist = float(math.hypot(
                float(self._x_m) - float(x_before),
                float(self._y_m) - float(y_before)))
            reward -= self.reward_k_len * step_dist
        # A* corridor deviation penalty (v9p1)
        if self.reward_k_astar_dev > 0.0 and self._astar_ref_xy is not None:
            d_astar, _ = self._astar_ref_query(float(self._x_m), float(self._y_m))
            excess = max(0.0, d_astar - self.reward_astar_corridor_m)
            reward -= self.reward_k_astar_dev * excess
        # Clearance-based safety shaping. Skip when already in collision to avoid compounding huge penalties.
        if not collision:
            od_pos = max(0.0, float(od_m))

            if self.reward_c_cbf > 0.0:
                # CBF log-barrier reward: c_cbf * log(max(h / h_max, eps))
                h_val = float(od_m)  # od_m already = min_od - collision_thr
                h_ratio = max(float(h_val) / float(self.cbf_h_max), 1e-6)
                reward += self.reward_c_cbf * float(math.log(h_ratio))
            else:
                # Legacy k_o near-obstacle penalty (using OD).
                if od_pos < self.safe_distance_m:
                    obs_term = (1.0 / (od_pos + self.reward_eps)) - (1.0 / (self.safe_distance_m + self.reward_eps))
                    obs_pen = float(self.reward_k_o) * float(obs_term)
                    reward -= min(float(self.reward_obs_max), float(obs_pen))

            # Forest near-obstacle speed coupling + optional soft speed cap.
            if od_pos < self.safe_speed_distance_m:
                # Speed coupling term (penalize speed when clearance is small).
                reward -= self.reward_k_v * ((self.safe_speed_distance_m - od_pos) / self.safe_speed_distance_m) * (
                    float(v_next) / float(self.model.v_max_m_s)
                ) ** 2

                # Soft speed cap (optional, but stabilizes forest driving in thin corridors).
                v_cap = float(self.model.v_max_m_s) * float(
                    np.clip(float(od_pos) / float(self.safe_speed_distance_m), 0.0, 1.0)
                )
                dv = max(0.0, float(v_next) - float(v_cap))
                reward -= self.reward_k_c * float(dv) ** 2

        # Goal alignment: once inside the goal region, make "stop + straighten" the dominant optimum.
        if bool(reached_pose) and (not bool(reached_stop)) and (not bool(collision)):
            v_thr = max(1e-6, float(self.goal_stop_speed_m_s))
            d_thr = max(1e-6, float(self.goal_stop_delta_rad))
            v_ratio2 = (abs(float(v_next)) / float(v_thr)) ** 2
            d_ratio2 = (abs(float(delta_next)) / float(d_thr)) ** 2
            reward -= float(self.reward_goal_v) * float(min(100.0, v_ratio2))
            reward -= float(self.reward_goal_delta) * float(min(100.0, d_ratio2))
            reward -= float(self.reward_goal_time)

        # Penalize leaving the goal region after having been inside it ("touch then leave").
        if bool(left_goal_region) and (not bool(collision)):
            reward -= float(self.reward_goal_leave)

        # Terminal
        if collision:
            reward += float(self.reward_collision)
        elif reached:
            reward += float(self.reward_reached)
        if stuck:
            reward -= float(self.stuck_penalty)

        # Markov: the *next* state carries previous action = action taken now.
        self._prev_delta_dot = float(delta_dot)
        self._prev_a = float(a)

        obs = self._observe()
        info = {
            "agent_xy": self._agent_xy_for_plot(),
            "pose_m": (self._x_m, self._y_m, self._psi_rad),
            "collision": bool(collision),
            "reached": bool(reached),
            "reached_pose": bool(reached_pose),
            "reached_stop": bool(reached_stop),
            "stuck": bool(stuck),
            "od_m": float(od_m),
            "d_goal_m": float(d_goal_after),
            "alpha_rad": float(alpha),
            "v_m_s": float(self._v_m_s),
            "delta_rad": float(self._delta_rad),
            "progress_m": float(progress_m),
            "no_progress_steps": int(self._no_progress_steps),
            "steps": int(self._steps),
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    def step(self, action: int):
        a_id = int(action)
        delta_dot = float(self.action_table[a_id, 0])
        a = float(self.action_table[a_id, 1])
        return self._step_with_controls(delta_dot=delta_dot, a=a)

    def step_continuous(self, *, delta_dot_rad_s: float, a_m_s2: float):
        """Continuous-control variant of `step()` (uses the same dynamics/collision/termination).

        This is intended for evaluating continuous controllers (e.g., MPC) on the forest env without
        forcing them through the discrete `action_table` interface used by DQN.
        """
        dd_max = float(self.model.delta_dot_max_rad_s)
        a_max = float(self.model.a_max_m_s2)
        delta_dot = float(np.clip(float(delta_dot_rad_s), -dd_max, +dd_max))
        a = float(np.clip(float(a_m_s2), -a_max, +a_max))
        return self._step_with_controls(delta_dot=delta_dot, a=a)

    # ------------------------------------------------------------------
    # CBF Safety Filter (v8p2)
    # ------------------------------------------------------------------
    def cbf_safe_action(
        self,
        delta_dot_norm: float,
        a_norm: float,
        *,
        alpha_cbf: float = 0.3,
        safety_margin: float = 0.15,
    ) -> tuple[float, float, dict]:
        """Discrete-time CBF safety filter for continuous actions.

        Args:
            delta_dot_norm: normalised steering rate in [-1, 1]
            a_norm: normalised acceleration in [-1, 1]
            alpha_cbf: CBF decay rate (0 < α < 1)
            safety_margin: additional safety margin in metres

        Returns:
            (dd_safe, a_safe, info) where info contains ``cbf_intervened``.
        """
        dd_raw = float(np.clip(float(delta_dot_norm), -1.0, 1.0))
        a_raw = float(np.clip(float(a_norm), -1.0, 1.0))

        # Current barrier value: h = min_od - safety_margin
        od_curr, _ = self._od_and_collision_m()
        h_curr = float(od_curr) - float(safety_margin)

        # Emergency brake when already inside unsafe set
        if h_curr <= 0.0:
            return 0.0, -1.0, {"cbf_intervened": True, "cbf_mode": "emergency"}

        # Convert normalised action to physical controls
        dd_max = float(self.model.delta_dot_max_rad_s)
        a_max = float(self.model.a_max_m_s2)
        dd_phys = dd_raw * dd_max
        a_phys = a_raw * a_max

        # Simulate one step with raw action
        x_n, y_n, psi_n, _, _ = bicycle_integrate_one_step(
            x_m=float(self._x_m), y_m=float(self._y_m),
            psi_rad=float(self._psi_rad), v_m_s=float(self._v_m_s),
            delta_rad=float(self._delta_rad),
            delta_dot_rad_s=dd_phys, a_m_s2=a_phys, params=self.model,
        )
        od_next, _ = self._od_and_collision_at_pose_m(x_n, y_n, psi_n)
        h_next = float(od_next) - float(safety_margin)

        # CBF constraint: h_next >= (1 - α) * h_curr
        threshold = (1.0 - float(alpha_cbf)) * h_curr
        violation = threshold - h_next

        if violation <= 0.0:
            # Safe — pass through
            return dd_raw, a_raw, {"cbf_intervened": False, "cbf_mode": "safe"}

        # --- Projection via numerical gradient ---
        eps = 1e-3
        grad = np.zeros(2, dtype=np.float64)
        for dim in range(2):
            u_plus = [dd_phys, a_phys]
            u_minus = [dd_phys, a_phys]
            scale = dd_max if dim == 0 else a_max
            u_plus[dim] += eps * scale
            u_minus[dim] -= eps * scale
            # Forward
            xp, yp, pp, _, _ = bicycle_integrate_one_step(
                x_m=float(self._x_m), y_m=float(self._y_m),
                psi_rad=float(self._psi_rad), v_m_s=float(self._v_m_s),
                delta_rad=float(self._delta_rad),
                delta_dot_rad_s=u_plus[0], a_m_s2=u_plus[1], params=self.model,
            )
            od_p, _ = self._od_and_collision_at_pose_m(xp, yp, pp)
            h_p = float(od_p) - float(safety_margin)
            # Backward
            xm, ym, pm, _, _ = bicycle_integrate_one_step(
                x_m=float(self._x_m), y_m=float(self._y_m),
                psi_rad=float(self._psi_rad), v_m_s=float(self._v_m_s),
                delta_rad=float(self._delta_rad),
                delta_dot_rad_s=u_minus[0], a_m_s2=u_minus[1], params=self.model,
            )
            od_m, _ = self._od_and_collision_at_pose_m(xm, ym, pm)
            h_m = float(od_m) - float(safety_margin)
            grad[dim] = (h_p - h_m) / (2.0 * eps * scale)

        grad_norm_sq = float(np.dot(grad, grad))
        if grad_norm_sq < 1e-12:
            # Gradient too small — fallback brake
            return 0.0, -1.0, {"cbf_intervened": True, "cbf_mode": "fallback"}

        # Linear projection: u_safe = u_raw + (violation / ||∇h||²) * ∇h
        correction = (violation / grad_norm_sq) * grad
        dd_safe_phys = dd_phys + correction[0] * dd_max
        a_safe_phys = a_phys + correction[1] * a_max
        dd_safe = float(np.clip(dd_safe_phys / dd_max, -1.0, 1.0))
        a_safe = float(np.clip(a_safe_phys / a_max, -1.0, 1.0))

        # Verify projection
        xv, yv, pv, _, _ = bicycle_integrate_one_step(
            x_m=float(self._x_m), y_m=float(self._y_m),
            psi_rad=float(self._psi_rad), v_m_s=float(self._v_m_s),
            delta_rad=float(self._delta_rad),
            delta_dot_rad_s=dd_safe * dd_max, a_m_s2=a_safe * a_max,
            params=self.model,
        )
        od_v, _ = self._od_and_collision_at_pose_m(xv, yv, pv)
        h_v = float(od_v) - float(safety_margin)
        if h_v < threshold:
            # Projection failed — fallback brake
            return 0.0, -1.0, {"cbf_intervened": True, "cbf_mode": "fallback"}

        return dd_safe, a_safe, {"cbf_intervened": True, "cbf_mode": "projected"}

    def _agent_xy_for_plot(self) -> tuple[float, float]:
        return (float(self._x_m) / self.cell_size_m, float(self._y_m) / self.cell_size_m)

    def _circle_centers_m(self) -> tuple[tuple[float, float], tuple[float, float]]:
        c = math.cos(float(self._psi_rad))
        s = math.sin(float(self._psi_rad))
        c1 = (float(self._x_m) + c * float(self.footprint.x1_m), float(self._y_m) + s * float(self.footprint.x1_m))
        c2 = (float(self._x_m) + c * float(self.footprint.x2_m), float(self._y_m) + s * float(self.footprint.x2_m))
        return c1, c2

    def _dist_at_m(self, x_m: float, y_m: float) -> float:
        xi = float(x_m) / self.cell_size_m
        yi = float(y_m) / self.cell_size_m
        return bilinear_sample_2d(self._dist_m, x=xi, y=yi, default=0.0)

    @staticmethod
    def _wrap_angle_rad_np(x: np.ndarray) -> np.ndarray:
        return (np.remainder(np.asarray(x, dtype=np.float64) + math.pi, 2.0 * math.pi) - math.pi).astype(
            np.float64, copy=False
        )

    def _dist_at_m_vec(self, x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
        xi = np.asarray(x_m, dtype=np.float64) / float(self.cell_size_m)
        yi = np.asarray(y_m, dtype=np.float64) / float(self.cell_size_m)
        return bilinear_sample_2d_vec(self._dist_m, x=xi, y=yi, default=0.0)

    def _od_and_collision_m(self) -> tuple[float, bool]:
        if not self._in_world_bounds(self._x_m, self._y_m):
            return -float("inf"), True
        c1, c2 = self._circle_centers_m()
        d1 = self._dist_at_m(c1[0], c1[1])
        d2 = self._dist_at_m(c2[0], c2[1])
        r = float(self.footprint.radius_m)
        thr = r + float(self._eps_cell_m)
        od_m = min(d1 - thr, d2 - thr)
        # Collision when the footprint intersects an occupied grid cell square.
        # The EDT is computed on the occupancy grid, so we pad by half the cell diagonal to account
        # for map resolution (circle-vs-cell-square intersection).
        collision = (d1 <= thr) or (d2 <= thr)
        return float(od_m), bool(collision)

    def _od_and_collision_at_pose_m(self, x_m: float, y_m: float, psi_rad: float) -> tuple[float, bool]:
        if not self._in_world_bounds(x_m, y_m):
            return -float("inf"), True

        c = math.cos(float(psi_rad))
        s = math.sin(float(psi_rad))
        c1x = float(x_m) + c * float(self.footprint.x1_m)
        c1y = float(y_m) + s * float(self.footprint.x1_m)
        c2x = float(x_m) + c * float(self.footprint.x2_m)
        c2y = float(y_m) + s * float(self.footprint.x2_m)

        d1 = self._dist_at_m(c1x, c1y)
        d2 = self._dist_at_m(c2x, c2y)
        r = float(self.footprint.radius_m)
        thr = r + float(self._eps_cell_m)
        od_m = min(float(d1) - thr, float(d2) - thr)
        collision = (float(d1) <= thr) or (float(d2) <= thr)
        return float(od_m), bool(collision)

    def _od_and_collision_at_pose_m_vec(
        self,
        x_m: np.ndarray,
        y_m: np.ndarray,
        psi_rad: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        xv = np.asarray(x_m, dtype=np.float64)
        yv = np.asarray(y_m, dtype=np.float64)
        psiv = np.asarray(psi_rad, dtype=np.float64)

        max_x = float(self._width - 1) * float(self.cell_size_m)
        max_y = float(self._height - 1) * float(self.cell_size_m)
        in_bounds = (xv >= 0.0) & (xv <= max_x) & (yv >= 0.0) & (yv <= max_y)

        c = np.cos(psiv)
        s = np.sin(psiv)
        c1x = xv + c * float(self.footprint.x1_m)
        c1y = yv + s * float(self.footprint.x1_m)
        c2x = xv + c * float(self.footprint.x2_m)
        c2y = yv + s * float(self.footprint.x2_m)

        d1 = self._dist_at_m_vec(c1x, c1y)
        d2 = self._dist_at_m_vec(c2x, c2y)
        r = float(self.footprint.radius_m)
        thr = r + float(self._eps_cell_m)
        od = np.minimum(d1 - thr, d2 - thr)
        coll = (d1 <= thr) | (d2 <= thr)

        od = np.where(in_bounds, od, -float("inf")).astype(np.float64, copy=False)
        coll = np.where(in_bounds, coll, True).astype(np.bool_, copy=False)
        return od, coll

    def _goal_dist_at_m_vec(self, x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
        xi = np.asarray(x_m, dtype=np.float64) / float(self.cell_size_m)
        yi = np.asarray(y_m, dtype=np.float64) / float(self.cell_size_m)
        return bilinear_sample_2d_vec(
            self._goal_dist_m,
            x=xi,
            y=yi,
            default=float(self._goal_dist_fill_m),
        )

    def _goal_dist_pose_m_vec(self, x_m: np.ndarray, y_m: np.ndarray, psi_rad: np.ndarray) -> np.ndarray:
        xv = np.asarray(x_m, dtype=np.float64)
        yv = np.asarray(y_m, dtype=np.float64)
        psiv = np.asarray(psi_rad, dtype=np.float64)
        c = np.cos(psiv)
        s = np.sin(psiv)
        c1x = xv + c * float(self.footprint.x1_m)
        c1y = yv + s * float(self.footprint.x1_m)
        c2x = xv + c * float(self.footprint.x2_m)
        c2y = yv + s * float(self.footprint.x2_m)
        c1 = self._goal_dist_at_m_vec(c1x, c1y)
        c2 = self._goal_dist_at_m_vec(c2x, c2y)
        return np.maximum(c1, c2).astype(np.float64, copy=False)

    def _bicycle_integrate_one_step_vec(
        self,
        *,
        x_m: np.ndarray,
        y_m: np.ndarray,
        psi_rad: np.ndarray,
        v_m_s: np.ndarray,
        delta_rad: np.ndarray,
        delta_dot_rad_s: np.ndarray,
        a_m_s2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        dt = float(self.model.dt)
        wheelbase = float(self.model.wheelbase_m)
        v_max = float(self.model.v_max_m_s)
        delta_max = float(self.model.delta_max_rad)

        v_next = np.clip(
            np.asarray(v_m_s, dtype=np.float64) + np.asarray(a_m_s2, dtype=np.float64) * dt,
            -float(v_max),
            float(v_max),
        )

        delta_unclipped = np.asarray(delta_rad, dtype=np.float64) + np.asarray(delta_dot_rad_s, dtype=np.float64) * dt
        delta_next = np.clip(delta_unclipped, -float(delta_max), +float(delta_max))

        psi = np.asarray(psi_rad, dtype=np.float64)
        x_next = np.asarray(x_m, dtype=np.float64) + v_next * np.cos(psi) * dt
        y_next = np.asarray(y_m, dtype=np.float64) + v_next * np.sin(psi) * dt
        psi_next = self._wrap_angle_rad_np(psi + (v_next / wheelbase) * np.tan(delta_next) * dt)
        return x_next, y_next, psi_next, v_next, delta_next

    def _rollout_constant_actions_end_state(
        self,
        *,
        delta_dot_rad_s: np.ndarray,
        a_m_s2: np.ndarray,
        horizon_steps: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        h = max(1, int(horizon_steps))
        delta_dot = np.asarray(delta_dot_rad_s, dtype=np.float64).reshape(-1)
        accel = np.asarray(a_m_s2, dtype=np.float64).reshape(-1)
        if delta_dot.shape != accel.shape:
            raise ValueError("delta_dot_rad_s and a_m_s2 must have the same shape")

        n = int(delta_dot.size)
        x = np.full((n,), float(self._x_m), dtype=np.float64)
        y = np.full((n,), float(self._y_m), dtype=np.float64)
        psi = np.full((n,), float(self._psi_rad), dtype=np.float64)
        v = np.full((n,), float(self._v_m_s), dtype=np.float64)
        delta = np.full((n,), float(self._delta_rad), dtype=np.float64)

        min_od = np.full((n,), float("inf"), dtype=np.float64)
        coll = np.zeros((n,), dtype=np.bool_)
        reached = np.zeros((n,), dtype=np.bool_)
        active = np.ones((n,), dtype=np.bool_)

        gx_m = float(self.goal_xy[0]) * float(self.cell_size_m)
        gy_m = float(self.goal_xy[1]) * float(self.cell_size_m)
        tol_m = float(self.goal_tolerance_m)

        for _ in range(h):
            if not bool(active.any()):
                break

            x1, y1, psi1, v1, delta1 = self._bicycle_integrate_one_step_vec(
                x_m=x,
                y_m=y,
                psi_rad=psi,
                v_m_s=v,
                delta_rad=delta,
                delta_dot_rad_s=delta_dot,
                a_m_s2=accel,
            )
            # Freeze terminated rollouts (reached/collided) so later steps do not affect masks.
            x = np.where(active, x1, x)
            y = np.where(active, y1, y)
            psi = np.where(active, psi1, psi)
            v = np.where(active, v1, v)
            delta = np.where(active, delta1, delta)

            od, coll_step = self._od_and_collision_at_pose_m_vec(x, y, psi)
            min_od = np.where(active, np.minimum(min_od, od), min_od)
            coll_now = coll_step & active
            coll |= coll_now

            d_goal_m = np.hypot(float(gx_m) - x, float(gy_m) - y)
            reached_now = (d_goal_m <= float(tol_m)) & active & (~coll_now)
            reached |= reached_now

            active &= ~(coll_now | reached_now)

        return x, y, psi, v, min_od, coll, reached

    def _fallback_action_short_rollout(
        self,
        *,
        horizon_steps: int,
        min_od_m: float = 0.0,
    ) -> int:
        """Fallback action chooser when Hybrid A* guidance is unavailable.

        Chooses the collision-free action that yields the lowest goal distance after a short
        constant-action rollout. This is used only as a last resort to keep rollouts moving.
        """

        h = max(1, int(horizon_steps))
        min_od_thr = float(min_od_m)
        delta_dot = self.action_table[:, 0]
        accel = self.action_table[:, 1]
        x, y, psi, _v, min_od, coll, reached = self._rollout_constant_actions_end_state(
            delta_dot_rad_s=delta_dot,
            a_m_s2=accel,
            horizon_steps=h,
        )
        dist1 = self._goal_dist_pose_m_vec(x, y, psi)

        ok = (~coll) & (min_od >= float(min_od_thr)) & np.isfinite(dist1)
        if bool(ok.any()):
            ok_reached = ok & reached
            idx = np.nonzero(ok_reached if bool(ok_reached.any()) else ok)[0]
            dists = dist1[idx]
            best_dist = float(np.min(dists))
            cand = idx[dists <= float(best_dist) + 1e-9]
            if cand.size == 0:
                cand = idx[int(np.argmin(dists))]
                return int(cand)
            best = int(cand[int(np.argmax(min_od[cand]))])
            return int(best)

        # Last resort: pick the one-step action with maximum clearance (even if it still collides).
        x0 = float(self._x_m)
        y0 = float(self._y_m)
        psi0 = float(self._psi_rad)
        v0 = float(self._v_m_s)
        delta0 = float(self._delta_rad)

        best_action = 0
        best_od = -float("inf")
        for a_id in range(int(self.action_table.shape[0])):
            delta_dot = float(self.action_table[a_id, 0])
            a = float(self.action_table[a_id, 1])
            x, y, psi, _v, _delta = bicycle_integrate_one_step(
                x_m=x0,
                y_m=y0,
                psi_rad=psi0,
                v_m_s=v0,
                delta_rad=delta0,
                delta_dot_rad_s=delta_dot,
                a_m_s2=a,
                params=self.model,
            )
            od, _coll = self._od_and_collision_at_pose_m(x, y, psi)
            if float(od) > float(best_od):
                best_od = float(od)
                best_action = int(a_id)
        return int(best_action)

    def _hybrid_astar_path(self, *, start_xy: tuple[int, int], timeout_s: float = 5.0, max_nodes: int = 200_000) -> list[tuple[float, float]]:
        key = (int(start_xy[0]), int(start_xy[1]), int(self.goal_xy[0]), int(self.goal_xy[1]))
        cached = self._ha_path_cache.get(key)
        if cached is not None:
            return cached

        # Fast path: load precomputed Hybrid A* reference paths for the canonical forest starts.
        # These are deterministic given the fixed forest seeds and avoid paying planning cost during training/inference.
        if (
            self.map_spec.name.startswith("forest_")
            and (int(start_xy[0]), int(start_xy[1])) == (int(self._canonical_start_xy[0]), int(self._canonical_start_xy[1]))
            and (int(self.goal_xy[0]), int(self.goal_xy[1])) == (int(self._canonical_goal_xy[0]), int(self._canonical_goal_xy[1]))
        ):
            pre = Path(__file__).resolve().parent / "maps" / "precomputed" / f"{self.map_spec.name}_hybrid_astar_path.json"
            if pre.exists():
                try:
                    payload = json.loads(pre.read_text(encoding="utf-8"))
                    pts = payload.get("path_xy_cells")
                    if bool(payload.get("success")) and isinstance(pts, list) and len(pts) >= 2:
                        path = [(float(p[0]), float(p[1])) for p in pts]
                        h, w = self._grid.shape
                        sx, sy = int(self._canonical_start_xy[0]), int(self._canonical_start_xy[1])
                        gx, gy = int(self._canonical_goal_xy[0]), int(self._canonical_goal_xy[1])
                        tol_cells = float(self.goal_tolerance_m) / float(self.cell_size_m)

                        def cell_free(xc: float, yc: float) -> bool:
                            xi = int(round(float(xc)))
                            yi = int(round(float(yc)))
                            if not (0 <= xi < w and 0 <= yi < h):
                                return False
                            return int(self._grid[yi, xi]) == 0

                        start_ok = (float(path[0][0]) - float(sx)) ** 2 + (float(path[0][1]) - float(sy)) ** 2 <= 4.0
                        goal_ok = (float(path[-1][0]) - float(gx)) ** 2 + (float(path[-1][1]) - float(gy)) ** 2 <= (tol_cells + 2.0) ** 2
                        path_ok = bool(start_ok) and bool(goal_ok) and all(cell_free(x, y) for x, y in path)
                        if path_ok:
                            self._ha_path_cache[key] = path
                            return path
                except Exception:
                    pass

        try:
            from forest_vehicle_dqn.baselines.pathplan import (
                default_ackermann_params,
                forest_two_circle_footprint,
                grid_map_from_obstacles,
                plan_hybrid_astar,
            )
        except Exception:
            self._ha_path_cache[key] = []
            return []

        grid_map = grid_map_from_obstacles(grid_y0_bottom=self._grid, cell_size_m=float(self.cell_size_m))
        params = default_ackermann_params(
            wheelbase_m=float(self.model.wheelbase_m),
            delta_max_rad=float(self.model.delta_max_rad),
            v_max_m_s=float(self.model.v_max_m_s),
        )
        footprint = forest_two_circle_footprint()

        res = plan_hybrid_astar(
            grid_map=grid_map,
            footprint=footprint,
            params=params,
            start_xy=(int(start_xy[0]), int(start_xy[1])),
            goal_xy=(int(self.goal_xy[0]), int(self.goal_xy[1])),
            goal_theta_rad=None,
            start_theta_rad=None,
            goal_xy_tol_m=float(self.goal_tolerance_m),
            goal_theta_tol_rad=float(math.pi),
            timeout_s=float(timeout_s),
            max_nodes=int(max_nodes),
        )
        path = list(res.path_xy_cells) if res.success else []
        self._ha_path_cache[key] = path
        return path

    # ------------------------------------------------------------------
    # A*-guided reward shaping helpers (v9p1)
    # ------------------------------------------------------------------

    def _compute_astar_ref_path(self) -> None:
        """Compute Grid A* reference path and store as continuous coords."""
        grid_map = grid_map_from_obstacles(
            grid_y0_bottom=self._grid, cell_size_m=float(self.cell_size_m))
        try:
            res = plan_grid_astar(
                grid_map=grid_map,
                start_xy=(int(self.start_xy[0]), int(self.start_xy[1])),
                goal_xy=(int(self.goal_xy[0]), int(self.goal_xy[1])),
                timeout_s=2.0,
                max_expanded=500_000,
            )
        except Exception:
            return
        if not bool(res.success) or len(res.path_xy_cells) < 2:
            return
        cs = float(self.cell_size_m)
        pts = np.array([(float(x) * cs, float(y) * cs)
                        for x, y in res.path_xy_cells], dtype=np.float64)
        diffs = np.diff(pts, axis=0)
        seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
        cumlen = np.zeros(len(pts), dtype=np.float64)
        cumlen[1:] = np.cumsum(seg_lens)
        self._astar_ref_xy = pts
        self._astar_ref_cumlen = cumlen
        self._astar_ref_total_len = float(cumlen[-1])

    def _astar_ref_query(self, x_m: float, y_m: float) -> tuple[float, float]:
        """Return (dist_to_path, remaining_path_len) for position (x_m, y_m).

        If no A* ref path is available, returns (0.0, nan).
        """
        if self._astar_ref_xy is None:
            return 0.0, float('nan')
        pt = np.array([float(x_m), float(y_m)], dtype=np.float64)
        dists_sq = ((self._astar_ref_xy - pt) ** 2).sum(axis=1)
        nearest_idx = int(np.argmin(dists_sq))
        dist_to_path = float(math.sqrt(float(dists_sq[nearest_idx])))
        remaining = self._astar_ref_total_len - float(
            self._astar_ref_cumlen[nearest_idx])
        return dist_to_path, float(remaining)

    def _astar_path_with_curve_opt(
        self,
        *,
        start_xy: tuple[int, int],
        curve_cfg: AStarCurveOptConfig,
        seed: int,
        replan_attempt: int,
        timeout_s: float = 5.0,
        max_expanded: int = 1_000_000,
    ) -> tuple[list[tuple[float, float]], list[float], dict[str, object]]:
        key = (
            int(start_xy[0]),
            int(start_xy[1]),
            int(self.goal_xy[0]),
            int(self.goal_xy[1]),
            int(seed),
            int(replan_attempt),
        )
        cached = self._astar_path_cache.get(key)
        if cached is not None:
            return list(cached[0]), list(cached[1]), {"cached": True}

        grid_map = grid_map_from_obstacles(grid_y0_bottom=self._grid, cell_size_m=float(self.cell_size_m))
        params = default_ackermann_params(
            wheelbase_m=float(self.model.wheelbase_m),
            delta_max_rad=float(self.model.delta_max_rad),
            v_max_m_s=float(self.model.v_max_m_s),
        )
        footprint = forest_two_circle_footprint(wheelbase_m=float(self.model.wheelbase_m))

        res = plan_grid_astar(
            grid_map=grid_map,
            start_xy=(int(start_xy[0]), int(start_xy[1])),
            goal_xy=(int(self.goal_xy[0]), int(self.goal_xy[1])),
            timeout_s=float(timeout_s),
            max_expanded=int(max_expanded),
            seed=int(seed),
            replan_attempt=int(replan_attempt),
            curve_opt_cfg=curve_cfg,
            footprint=footprint,
            ackermann_params=params,
        )
        if not bool(res.success):
            reason = str(res.stats.get("failure_reason", "planner_fail"))
            raise RuntimeError(f"A*+curve-opt failed: {reason}; stats={res.stats}")

        path = list(res.path_xy_cells)
        theta = (
            [float(t) for t in res.path_theta_rad]
            if isinstance(res.path_theta_rad, list)
            else [0.0 for _ in range(max(1, len(path)))]
        )
        self._astar_path_cache[key] = (list(path), list(theta))
        return path, theta, dict(res.stats)


    def expert_action_astar_mpc(
        self,
        *,
        mpc_cfg,
        curve_cfg: AStarCurveOptConfig,
        seed: int,
        replan_attempt: int,
        timeout_s: float = 5.0,
        max_expanded: int = 1_000_000,
    ) -> int:
        from forest_vehicle_dqn.baselines.mpc_local_planner import solve_mpc_one_step

        path, theta, _stats = self._astar_path_with_curve_opt(
            start_xy=self._astar_start_xy,
            curve_cfg=curve_cfg,
            seed=int(seed),
            replan_attempt=int(replan_attempt),
            timeout_s=float(timeout_s),
            max_expanded=int(max_expanded),
        )
        if len(path) < 2:
            return self._fallback_action_short_rollout(horizon_steps=max(1, int(mpc_cfg.horizon_steps)), min_od_m=0.0)

        cell_size_m = float(self.cell_size_m)
        path_xy_m = np.asarray([(float(x) * cell_size_m, float(y) * cell_size_m) for x, y in path], dtype=np.float64)
        path_theta = np.asarray([float(t) for t in theta], dtype=np.float64)
        if int(path_theta.size) != int(path_xy_m.shape[0]):
            path_theta = np.zeros((int(path_xy_m.shape[0]),), dtype=np.float64)

        step = solve_mpc_one_step(
            env=self,
            path_xy_m=path_xy_m,
            path_theta=path_theta,
            prev_path_idx=int(self._astar_progress_idx),
            config=mpc_cfg,
        )
        self._astar_progress_idx = int(step.nearest_path_idx)

        if not bool(step.feasible):
            raise RuntimeError(f"A*+MPC infeasible at step: {step.fail_reason}")

        dd = float(step.delta_dot_rad_s)
        aa = float(step.a_m_s2)
        controls = self.action_table
        d2 = np.square(controls[:, 0] - dd) + np.square(controls[:, 1] - aa)
        best = int(np.argmin(d2))
        return int(best)


    def expert_action_hybrid_astar_mpc(
        self,
        *,
        mpc_cfg,
    ) -> int:
        from forest_vehicle_dqn.baselines.mpc_local_planner import solve_mpc_one_step

        path = self._hybrid_astar_path(start_xy=self._ha_start_xy)
        if len(path) < 2:
            return self._fallback_action_short_rollout(horizon_steps=max(1, int(mpc_cfg.horizon_steps)), min_od_m=0.0)

        cell_size_m = float(self.cell_size_m)
        path_xy_m = np.asarray([(float(x) * cell_size_m, float(y) * cell_size_m) for x, y in path], dtype=np.float64)
        path_theta = np.zeros((int(path_xy_m.shape[0]),), dtype=np.float64)
        if int(path_xy_m.shape[0]) >= 2:
            dxy = np.diff(path_xy_m, axis=0)
            seg_h = np.arctan2(dxy[:, 1], dxy[:, 0]).astype(np.float64, copy=False)
            path_theta[:-1] = seg_h
            path_theta[-1] = seg_h[-1]

        step = solve_mpc_one_step(
            env=self,
            path_xy_m=path_xy_m,
            path_theta=path_theta,
            prev_path_idx=int(self._ha_progress_idx),
            config=mpc_cfg,
        )
        self._ha_progress_idx = int(step.nearest_path_idx)

        if not bool(step.feasible):
            raise RuntimeError(f"Hybrid A*+MPC infeasible at step: {step.fail_reason}")

        dd = float(step.delta_dot_rad_s)
        aa = float(step.a_m_s2)
        controls = self.action_table
        d2 = np.square(controls[:, 0] - dd) + np.square(controls[:, 1] - aa)
        best = int(np.argmin(d2))
        return int(best)

    def expert_continuous_action_hybrid_astar_mpc(
        self,
        *,
        mpc_cfg,
    ) -> np.ndarray:
        """Like expert_action_hybrid_astar_mpc but returns continuous action in [-1, 1]."""
        from forest_vehicle_dqn.baselines.mpc_local_planner import solve_mpc_one_step

        path = self._hybrid_astar_path(start_xy=self._ha_start_xy)
        if len(path) < 2:
            return np.zeros(2, dtype=np.float32)

        cell_size_m = float(self.cell_size_m)
        path_xy_m = np.asarray([(float(x) * cell_size_m, float(y) * cell_size_m) for x, y in path], dtype=np.float64)
        path_theta = np.zeros((int(path_xy_m.shape[0]),), dtype=np.float64)
        if int(path_xy_m.shape[0]) >= 2:
            dxy = np.diff(path_xy_m, axis=0)
            seg_h = np.arctan2(dxy[:, 1], dxy[:, 0]).astype(np.float64, copy=False)
            path_theta[:-1] = seg_h
            path_theta[-1] = seg_h[-1]

        step = solve_mpc_one_step(
            env=self,
            path_xy_m=path_xy_m,
            path_theta=path_theta,
            prev_path_idx=int(self._ha_progress_idx),
            config=mpc_cfg,
        )
        self._ha_progress_idx = int(step.nearest_path_idx)

        if not bool(step.feasible):
            return np.zeros(2, dtype=np.float32)

        dd_max = float(self.model.delta_dot_max_rad_s)
        a_max = float(self.model.a_max_m_s2)
        return np.array([
            float(np.clip(float(step.delta_dot_rad_s) / dd_max, -1.0, 1.0)),
            float(np.clip(float(step.a_m_s2) / a_max, -1.0, 1.0)),
        ], dtype=np.float32)


    def expert_action_hybrid_astar(
        self,
        *,
        lookahead_points: int = 3,
        horizon_steps: int = 4,
        w_target: float = 0.5,
        w_heading: float = 0.4,
        w_clearance: float = 0.2,
        w_speed: float = 0.0,
    ) -> int:
        """Hybrid-A* guided expert (DQfD demos / guided exploration).

        Computes a Hybrid A* reference path once per episode-start (cached by start cell) and then
        tracks it with a pure-pursuit style steering target + discrete control selection under a
        short-horizon safety mask.
        """

        path = self._hybrid_astar_path(start_xy=self._ha_start_xy)
        if len(path) < 2:
            return self._fallback_action_short_rollout(horizon_steps=int(horizon_steps), min_od_m=0.0)

        x_cells = float(self._x_m) / float(self.cell_size_m)
        y_cells = float(self._y_m) / float(self.cell_size_m)

        # Find nearest path index (limited window around previous index).
        start_i = max(0, int(self._ha_progress_idx) - 25)
        end_i = min(len(path), int(self._ha_progress_idx) + 250)
        if end_i <= start_i:
            start_i, end_i = 0, len(path)
        best_i = start_i
        best_d2 = float("inf")
        for i in range(start_i, end_i):
            px, py = path[i]
            d2 = (float(px) - x_cells) ** 2 + (float(py) - y_cells) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        self._ha_progress_idx = int(best_i)

        la = max(1, int(lookahead_points))
        tgt_i = min(int(best_i) + la, len(path) - 1)
        tx_cells, ty_cells = path[tgt_i]
        tx_m = float(tx_cells) * float(self.cell_size_m)
        ty_m = float(ty_cells) * float(self.cell_size_m)

        h = max(1, int(horizon_steps))
        delta_dot = self.action_table[:, 0]
        accel = self.action_table[:, 1]
        x, y, psi, v, min_od, coll, _reached = self._rollout_constant_actions_end_state(
            delta_dot_rad_s=delta_dot,
            a_m_s2=accel,
            horizon_steps=h,
        )

        goal_dist = self._goal_dist_pose_m_vec(x, y, psi)
        dist_tgt = np.hypot(float(tx_m) - x, float(ty_m) - y)
        tgt_heading = np.arctan2(float(ty_m) - y, float(tx_m) - x)
        heading_err = self._wrap_angle_rad_np(tgt_heading - psi)

        score = -goal_dist
        score += -float(w_target) * dist_tgt - float(w_heading) * np.abs(heading_err)
        score += float(w_clearance) * min_od

        if float(w_speed) != 0.0:
            v_max = float(self.model.v_max_m_s)
            score += float(w_speed) * (v / max(1e-9, float(v_max)))

        invalid = coll | (~np.isfinite(score)) | (~np.isfinite(goal_dist))
        score = np.where(invalid, -float("inf"), score)
        best_action = int(np.argmax(score))
        if not math.isfinite(float(score[best_action])):
            return self._fallback_action_short_rollout(horizon_steps=int(horizon_steps), min_od_m=0.0)

        return int(best_action)

    def _rollout_constant_action_metrics(
        self,
        a_id: int,
        *,
        horizon_steps: int,
    ) -> tuple[float, float, float, bool, bool]:
        """Simulate a constant discrete action for a short horizon.

        Returns: (goal_dist_end, v_end, min_od_over_horizon, collision_over_horizon, reached_within_horizon).
        """

        h = max(1, int(horizon_steps))
        a_id = int(a_id)

        delta_dot = float(self.action_table[a_id, 0])
        a = float(self.action_table[a_id, 1])

        x = float(self._x_m)
        y = float(self._y_m)
        psi = float(self._psi_rad)
        v = float(self._v_m_s)
        delta = float(self._delta_rad)

        gx_m = float(self.goal_xy[0]) * float(self.cell_size_m)
        gy_m = float(self.goal_xy[1]) * float(self.cell_size_m)
        tol_m = float(self.goal_tolerance_m)

        min_od = float("inf")
        for _ in range(h):
            x, y, psi, v, delta = bicycle_integrate_one_step(
                x_m=x,
                y_m=y,
                psi_rad=psi,
                v_m_s=v,
                delta_rad=delta,
                delta_dot_rad_s=delta_dot,
                a_m_s2=a,
                params=self.model,
            )
            od, coll = self._od_and_collision_at_pose_m(x, y, psi)
            min_od = min(float(min_od), float(od))
            if coll:
                return float("inf"), float(v), float(min_od), True, False

            if float(math.hypot(float(gx_m) - float(x), float(gy_m) - float(y))) <= float(tol_m):
                goal_dist = float(self._goal_dist_pose_m(x, y, psi))
                return float(goal_dist), float(v), float(min_od), False, True

        goal_dist = float(self._goal_dist_pose_m(x, y, psi))
        return float(goal_dist), float(v), float(min_od), False, False

    def is_action_safe(
        self,
        a_id: int,
        *,
        horizon_steps: int = 10,
        min_od_m: float = 0.0,
    ) -> bool:
        _goal_dist, _v, min_od, coll, _reached = self._rollout_constant_action_metrics(
            int(a_id), horizon_steps=int(horizon_steps)
        )
        if bool(coll):
            return False
        return float(min_od) >= float(min_od_m)

    def is_action_admissible(
        self,
        a_id: int,
        *,
        horizon_steps: int = 10,
        min_od_m: float = 0.0,
        min_progress_m: float = 1e-4,
        allow_reverse: bool = True,
    ) -> bool:
        dist0 = float(self._goal_dist_pose_m(float(self._x_m), float(self._y_m), float(self._psi_rad)))
        if not math.isfinite(dist0):
            return True
        reached_pose_now = bool(
            self._goal_pose_reached(
                d_goal_m=float(self._distance_to_goal_m()),
                alpha_rad=float(self._goal_relative_angle_rad()),
            )
        )
        goal_relax_dist = float(self.goal_tolerance_m) * float(self.goal_admissible_relax_factor)

        # Progress is judged at the end of the short-horizon constant-action rollout, while safety
        # (collision / clearance) is judged over the same horizon.
        h = max(1, int(horizon_steps))
        dist1, v_end, min_od, coll, reached = self._rollout_constant_action_metrics(int(a_id), horizon_steps=h)
        if bool(coll):
            return False
        if float(min_od) < float(min_od_m):
            return False
        if bool(reached):
            return True
        if bool(reached_pose_now) and math.isfinite(dist1) and (float(dist1) <= float(goal_relax_dist)):
            return True
        if not math.isfinite(dist1):
            return False
        if float(dist0 - dist1) >= float(min_progress_m):
            return True

        # Reverse is primarily an "escape" maneuver. Besides the original fallback (no forward-progress
        # actions), also allow safe reverse when clearance is small or the agent has stagnated for
        # several steps so the policy can learn to back out.
        if bool(allow_reverse):
            reverse_v_min = float(self.reverse_v_min_m_s)
            if float(v_end) < -float(reverse_v_min):
                if bool(self._reverse_unlocked(min_od_m=float(min_od_m))):
                    return True
                prog_mask = self.admissible_action_mask(
                    horizon_steps=h,
                    min_od_m=float(min_od_m),
                    min_progress_m=float(min_progress_m),
                    fallback_to_safe=False,
                    allow_reverse=False,
                )
                if not bool(prog_mask.any()):
                    return True
        return False

    def safe_action_mask(
        self,
        *,
        horizon_steps: int = 10,
        min_od_m: float = 0.0,
    ) -> np.ndarray:
        """Return a boolean mask of actions that remain collision-free over a short horizon."""
        h = max(1, int(horizon_steps))
        min_od_thr = float(min_od_m)
        delta_dot = self.action_table[:, 0]
        accel = self.action_table[:, 1]
        _x, _y, _psi, _v, min_od, coll, _reached = self._rollout_constant_actions_end_state(
            delta_dot_rad_s=delta_dot,
            a_m_s2=accel,
            horizon_steps=h,
        )
        out = (~coll) & (min_od >= float(min_od_thr))
        return out.astype(np.bool_, copy=False)

    def admissible_action_mask(
        self,
        *,
        horizon_steps: int = 10,
        min_od_m: float = 0.0,
        min_progress_m: float = 1e-4,
        fallback_to_safe: bool = True,
        allow_reverse: bool = True,
    ) -> np.ndarray:
        """Mask actions that are safe and make goal-distance progress (optionally allow reverse)."""

        dist0 = float(self._goal_dist_pose_m(float(self._x_m), float(self._y_m), float(self._psi_rad)))
        out = np.zeros((int(self.action_table.shape[0]),), dtype=np.bool_)
        if not math.isfinite(dist0):
            out[:] = True
            return out
        reached_pose_now = bool(
            self._goal_pose_reached(
                d_goal_m=float(self._distance_to_goal_m()),
                alpha_rad=float(self._goal_relative_angle_rad()),
            )
        )
        goal_relax_dist = float(self.goal_tolerance_m) * float(self.goal_admissible_relax_factor)

        h = max(1, int(horizon_steps))
        min_od_thr = float(min_od_m)
        min_prog = float(min_progress_m)

        delta_dot = self.action_table[:, 0]
        accel = self.action_table[:, 1]
        x, y, psi, v_end, min_od, coll, reached = self._rollout_constant_actions_end_state(
            delta_dot_rad_s=delta_dot,
            a_m_s2=accel,
            horizon_steps=h,
        )
        dist1 = self._goal_dist_pose_m_vec(x, y, psi)

        safe = (~coll) & (min_od >= float(min_od_thr)) & np.isfinite(dist1)
        prog = ((float(dist0) - dist1) >= float(min_prog)) | reached
        out = safe & prog
        if bool(reached_pose_now):
            out = out | (safe & (dist1 <= float(goal_relax_dist)))
        if bool(allow_reverse):
            reverse_v_min = float(self.reverse_v_min_m_s)
            reverse = safe & (v_end < -float(reverse_v_min))
            if bool(self._reverse_unlocked(min_od_m=float(min_od_m))) or not bool(out.any()):
                out = out | reverse

        # Fallback: if everything is filtered out, keep the collision-safe actions.
        if bool(fallback_to_safe) and not bool(out.any()):
            out = (~coll) & (min_od >= float(min_od_thr))
        return out.astype(np.bool_, copy=False)

    def _observe(self) -> np.ndarray:
        # Normalized (x,y) + goal (x,y) in [-1,1].
        max_x = max(1e-6, float(self._width - 1) * self.cell_size_m)
        max_y = max(1e-6, float(self._height - 1) * self.cell_size_m)
        ax_n = 2.0 * (float(self._x_m) / float(max_x)) - 1.0
        ay_n = 2.0 * (float(self._y_m) / float(max_y)) - 1.0
        gx_n = 2.0 * ((float(self.goal_xy[0]) * self.cell_size_m) / float(max_x)) - 1.0
        gy_n = 2.0 * ((float(self.goal_xy[1]) * self.cell_size_m) / float(max_y)) - 1.0

        # Scalars
        sin_psi = float(math.sin(float(self._psi_rad)))
        cos_psi = float(math.cos(float(self._psi_rad)))
        v_n = float(self._v_m_s) / float(self.model.v_max_m_s)
        delta_lim = float(self.model.delta_max_rad)
        delta_n = 0.0 if abs(delta_lim) < 1e-9 else float(self._delta_rad) / float(delta_lim)

        alpha_n = float(self._goal_relative_angle_rad()) / math.pi
        od01 = min(self.od_cap_m, max(0.0, float(self._last_od_m))) / float(self.od_cap_m)
        od_n = 2.0 * float(np.clip(od01, 0.0, 1.0)) - 1.0

        # Clamp to stable ranges.
        ax_n = float(np.clip(ax_n, -1.0, 1.0))
        ay_n = float(np.clip(ay_n, -1.0, 1.0))
        gx_n = float(np.clip(gx_n, -1.0, 1.0))
        gy_n = float(np.clip(gy_n, -1.0, 1.0))
        sin_psi = float(np.clip(sin_psi, -1.0, 1.0))
        cos_psi = float(np.clip(cos_psi, -1.0, 1.0))
        v_n = float(np.clip(v_n, -1.0, 1.0))
        delta_n = float(np.clip(delta_n, -1.0, 1.0))
        alpha_n = float(np.clip(alpha_n, -1.0, 1.0))
        obs = np.concatenate(
            [
                np.array(
                    [ax_n, ay_n, gx_n, gy_n, sin_psi, cos_psi, v_n, delta_n, alpha_n, od_n],
                    dtype=np.float32,
                ),
                self._obs_occ_flat,
            ]
        )
        return obs.astype(np.float32, copy=False)

    def _distance_to_goal_m(self) -> float:
        gx = float(self.goal_xy[0]) * self.cell_size_m
        gy = float(self.goal_xy[1]) * self.cell_size_m
        return float(math.hypot(gx - float(self._x_m), gy - float(self._y_m)))

    def _goal_relative_angle_rad(self) -> float:
        gx = float(self.goal_xy[0]) * self.cell_size_m
        gy = float(self.goal_xy[1]) * self.cell_size_m
        goal_heading = math.atan2(gy - float(self._y_m), gx - float(self._x_m))
        return wrap_angle_rad(float(goal_heading) - float(self._psi_rad))

    def _goal_dist_at_m(self, x_m: float, y_m: float) -> float:
        xi = float(x_m) / self.cell_size_m
        yi = float(y_m) / self.cell_size_m
        return bilinear_sample_2d(self._goal_dist_m, x=xi, y=yi, default=float(self._goal_dist_fill_m))

    def _goal_dist_pose_m(self, x_m: float, y_m: float, psi_rad: float) -> float:
        """Goal distance for the whole vehicle footprint (two circles).

        Uses the max over circle-center distances so progress shaping does not
        encourage motions where one circle becomes trapped/unsafe.
        """
        c = math.cos(float(psi_rad))
        s = math.sin(float(psi_rad))
        c1x = float(x_m) + c * float(self.footprint.x1_m)
        c1y = float(y_m) + s * float(self.footprint.x1_m)
        c2x = float(x_m) + c * float(self.footprint.x2_m)
        c2y = float(y_m) + s * float(self.footprint.x2_m)
        c1 = self._goal_dist_at_m(c1x, c1y)
        c2 = self._goal_dist_at_m(c2x, c2y)
        return float(max(float(c1), float(c2)))

    def _in_world_bounds(self, x_m: float, y_m: float) -> bool:
        max_x = float(self._width - 1) * self.cell_size_m
        max_y = float(self._height - 1) * self.cell_size_m
        return (0.0 <= float(x_m) <= max_x) and (0.0 <= float(y_m) <= max_y)

    def _sector_ray_distances_n(self) -> np.ndarray:
        """LIDAR-like ray distances, normalized to [0,1]. Angles are in the vehicle frame."""
        x0 = float(self._x_m) / self.cell_size_m
        y0 = float(self._y_m) / self.cell_size_m
        max_range_cells = float(self.sensor_range_m) / self.cell_size_m
        step_cells = 0.5
        max_steps = max(1, int(math.ceil(max_range_cells / step_cells)))

        out = np.ones((self.n_sectors,), dtype=np.float32)
        for i in range(self.n_sectors):
            ang = float(self._psi_rad) + (2.0 * math.pi) * (float(i) / float(self.n_sectors))
            c = math.cos(ang)
            s = math.sin(ang)
            hit_cells = max_range_cells
            for j in range(1, max_steps + 1):
                d_cells = float(j) * step_cells
                xi = x0 + c * d_cells
                yi = y0 + s * d_cells
                ix = int(math.floor(xi + 0.5))
                iy = int(math.floor(yi + 0.5))
                if not (0 <= ix < self._width and 0 <= iy < self._height):
                    hit_cells = d_cells
                    break
                if self._grid[iy, ix] == 1:
                    hit_cells = d_cells
                    break
            out[i] = float(np.clip((hit_cells * self.cell_size_m) / float(self.sensor_range_m), 0.0, 1.0))
        return out

    # ------------------------------------------------------------------
    # SAC global-map observation
    # ------------------------------------------------------------------

    def get_global_map(self, *, channels: int = 3,
                       map_size: int = 48) -> np.ndarray:
        """Return (C, map_size, map_size) float32 global map for SAC.

        Ch0: occupancy grid (0=free, 1=obstacle)
        Ch1: agent position Gaussian blob
        Ch2: goal position Gaussian blob
        """
        sz = int(map_size)
        # Ch0 – occupancy (nearest-neighbour downsample)
        occ = cv2.resize(
            self._grid.astype(np.float32, copy=False),
            dsize=(sz, sz), interpolation=cv2.INTER_NEAREST,
        )

        # Helper: Gaussian blob at (x_m, y_m) on the downsampled grid
        def _blob(x_m: float, y_m: float) -> np.ndarray:
            max_x = max(1e-6, float(self._width - 1) * self.cell_size_m)
            max_y = max(1e-6, float(self._height - 1) * self.cell_size_m)
            cx = float(x_m) / max_x * (sz - 1)
            cy = float(y_m) / max_y * (sz - 1)
            xs = np.arange(sz, dtype=np.float32)
            ys = np.arange(sz, dtype=np.float32)
            gx = np.exp(-0.5 * ((xs - cx) / max(1.0, sz / 8.0)) ** 2)
            gy = np.exp(-0.5 * ((ys - cy) / max(1.0, sz / 8.0)) ** 2)
            return np.outer(gy, gx).astype(np.float32)

        # Ch1 – agent blob
        agent_blob = _blob(float(self._x_m), float(self._y_m))
        # Ch2 – goal blob
        gx_m = float(self.goal_xy[0]) * self.cell_size_m
        gy_m = float(self.goal_xy[1]) * self.cell_size_m
        goal_blob = _blob(gx_m, gy_m)

        return np.stack([occ, agent_blob, goal_blob], axis=0).astype(
            np.float32, copy=False)

    def observe_sac(self, *, map_size: int = 48) -> dict:
        """Return SAC observation dict: {maps: (3,H,W), scalars: (D,)}."""
        maps = self.get_global_map(channels=3, map_size=map_size)

        # Reuse scalar computation from _observe()
        max_x = max(1e-6, float(self._width - 1) * self.cell_size_m)
        max_y = max(1e-6, float(self._height - 1) * self.cell_size_m)
        ax_n = float(np.clip(2.0 * float(self._x_m) / max_x - 1.0, -1, 1))
        ay_n = float(np.clip(2.0 * float(self._y_m) / max_y - 1.0, -1, 1))
        gx_n = float(np.clip(
            2.0 * (float(self.goal_xy[0]) * self.cell_size_m) / max_x - 1.0,
            -1, 1))
        gy_n = float(np.clip(
            2.0 * (float(self.goal_xy[1]) * self.cell_size_m) / max_y - 1.0,
            -1, 1))
        sin_psi = float(np.clip(math.sin(float(self._psi_rad)), -1, 1))
        cos_psi = float(np.clip(math.cos(float(self._psi_rad)), -1, 1))
        v_n = float(np.clip(
            float(self._v_m_s) / float(self.model.v_max_m_s), -1, 1))
        delta_lim = float(self.model.delta_max_rad)
        delta_n = float(np.clip(
            0.0 if abs(delta_lim) < 1e-9
            else float(self._delta_rad) / delta_lim, -1, 1))
        alpha_n = float(np.clip(
            float(self._goal_relative_angle_rad()) / math.pi, -1, 1))
        od01 = min(self.od_cap_m, max(0.0, float(self._last_od_m))) / float(
            self.od_cap_m)
        od_n = float(np.clip(2.0 * od01 - 1.0, -1, 1))

        # Extra scalars for SAC: prev controls + path length so far
        dd_max = float(self.model.delta_dot_max_rad_s)
        a_max = float(self.model.a_max_m_s2)
        prev_dd_n = float(np.clip(
            0.0 if abs(dd_max) < 1e-9
            else float(self._prev_delta_dot) / dd_max, -1, 1))
        prev_a_n = float(np.clip(
            0.0 if abs(a_max) < 1e-9
            else float(self._prev_a) / a_max, -1, 1))

        scalars = np.array([
            ax_n, ay_n, gx_n, gy_n, sin_psi, cos_psi,
            v_n, delta_n, alpha_n, od_n, prev_dd_n, prev_a_n,
        ], dtype=np.float32)

        return {"maps": maps, "scalars": scalars}
