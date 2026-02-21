"""Tests for ForestPLRCurriculum."""
import pytest
from forest_vehicle_dqn.plr_curriculum import (
    ForestPLRCurriculum, HAS_SYLLABUS,
)


@pytest.mark.skipif(not HAS_SYLLABUS, reason="syllabus-rl not installed")
def test_plr_sample_and_report():
    """PLR should sample valid levels and accept reports."""
    plr = ForestPLRCurriculum([6, 10, 14, 20, 30, 42])
    idx = plr.sample_level()
    assert 0 <= idx < 6
    lo, hi = plr.level_to_dist_range(idx)
    assert lo >= 6.0
    plr.report(idx, -100.0)  # should not raise


@pytest.mark.skipif(not HAS_SYLLABUS, reason="syllabus-rl not installed")
def test_plr_last_level_open_ended():
    """Last level should have max_dist=0.0 (no upper bound)."""
    plr = ForestPLRCurriculum([6, 10, 14])
    lo, hi = plr.level_to_dist_range(2)
    assert lo == 14.0
    assert hi == 0.0


@pytest.mark.skipif(not HAS_SYLLABUS, reason="syllabus-rl not installed")
def test_plr_multiple_reports():
    """Multiple reports should not raise."""
    plr = ForestPLRCurriculum([6, 10, 14, 20])
    for _ in range(5):
        idx = plr.sample_level()
        plr.report(idx, float(-50 + idx * 10), length=100)
