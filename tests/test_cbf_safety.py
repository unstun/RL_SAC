"""CBF safety filter 单元测试。"""
import math
import numpy as np
import pytest


def _make_env():
    """创建一个 ForestEnv (AMRBicycleEnv) 用于测试。"""
    from forest_vehicle_dqn.maps import get_map_spec
    from forest_vehicle_dqn.env import AMRBicycleEnv
    spec = get_map_spec("forest_a")
    env = AMRBicycleEnv(spec, max_steps=1200)
    return env


def test_cbf_safe_action_passthrough():
    """安全动作应直通不修改。"""
    env = _make_env()
    env.reset(seed=42)
    dd_out, a_out, info = env.cbf_safe_action(0.0, 0.1)
    assert not info["cbf_intervened"], "安全动作不应被 CBF 修改"
    assert dd_out == pytest.approx(0.0, abs=1e-6)
    assert a_out == pytest.approx(0.1, abs=1e-6)


def test_cbf_safe_action_intervenes_near_obstacle():
    """接近障碍物时 CBF 应介入修正动作。"""
    env = _make_env()
    env.reset(seed=42)
    # safety_margin=100.0 使得几乎任何位置都"不安全"，强制 CBF 介入
    _, _, info = env.cbf_safe_action(0.0, 1.0, safety_margin=100.0)
    assert info["cbf_intervened"], "接近障碍物时 CBF 应介入"


def test_cbf_safe_action_output_in_bounds():
    """CBF 输出动作应在 [-1, 1] 范围内。"""
    env = _make_env()
    env.reset(seed=42)
    for dd_raw, a_raw in [(-1, -1), (1, 1), (0, 0), (-0.5, 0.8)]:
        dd_out, a_out, _ = env.cbf_safe_action(dd_raw, a_raw)
        assert -1.0 <= dd_out <= 1.0, f"dd_out={dd_out} 超出范围"
        assert -1.0 <= a_out <= 1.0, f"a_out={a_out} 超出范围"
