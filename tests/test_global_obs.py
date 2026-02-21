"""Tests for global map observation in AMRBicycleEnv."""
import numpy as np
from forest_vehicle_dqn.maps import ArrayGridMapSpec
from forest_vehicle_dqn.env import AMRBicycleEnv


def _make_test_env(grid_size: int = 60) -> AMRBicycleEnv:
    """Create a minimal env with a small empty grid for testing."""
    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    # Add a few obstacles
    grid[25:35, 25:35] = 1
    spec = ArrayGridMapSpec(
        name="test_global",
        grid_y0_bottom=grid,
        start_xy=(5, 5),
        goal_xy=(55, 55),
    )
    return AMRBicycleEnv(spec, max_steps=200, obs_map_size=12)


def test_global_map_shape():
    env = _make_test_env()
    env.reset(seed=42)
    gmap = env.get_global_map(channels=3, map_size=48)
    assert gmap.shape == (3, 48, 48)
    assert gmap.dtype == np.float32


def test_global_map_channels():
    env = _make_test_env()
    env.reset(seed=42)
    gmap = env.get_global_map(channels=3, map_size=48)
    # Ch0: occupancy should have some 1s (obstacles) and 0s (free)
    assert gmap[0].max() >= 0.5  # obstacles present
    assert gmap[0].min() < 0.5   # free space present
    # Ch1/Ch2: blobs should be in [0, 1]
    assert 0.0 <= gmap[1].min() and gmap[1].max() <= 1.0
    assert 0.0 <= gmap[2].min() and gmap[2].max() <= 1.0


def test_observe_sac_dict():
    env = _make_test_env()
    env.reset(seed=42)
    obs = env.observe_sac(map_size=48)
    assert "maps" in obs
    assert "scalars" in obs
    assert obs["maps"].shape == (3, 48, 48)
    assert obs["scalars"].shape == (12,)
    assert obs["scalars"].dtype == np.float32
