"""Tests for TECRL (reward-entropy separated SAC)."""
import numpy as np
import torch
from forest_vehicle_dqn.sac_agent import SACAgent, SACConfig


def test_tecrl_agent_creates():
    """TECRL agent should have entropy_critic and entropy_budget."""
    cfg = SACConfig(use_tecrl=True, buffer_size=100)
    agent = SACAgent(cfg, device="cpu", seed=0)
    assert hasattr(agent, "entropy_critic")
    assert hasattr(agent, "entropy_critic_target")
    assert hasattr(agent, "entropy_budget")
    assert agent.entropy_budget < 0  # negative (ρ * -dim / (1-γ))


def test_tecrl_update_returns_stats():
    """TECRL update should return expected stat keys."""
    cfg = SACConfig(use_tecrl=True, buffer_size=100, batch_size=4)
    agent = SACAgent(cfg, device="cpu", seed=0)
    rng = np.random.default_rng(42)
    for _ in range(10):
        obs = {
            "maps": rng.standard_normal((3, 48, 48)).astype(np.float32),
            "scalars": rng.standard_normal(12).astype(np.float32),
        }
        act = rng.standard_normal(2).astype(np.float32)
        agent.observe(obs, act, -1.0, obs, False)
    stats = agent.update()
    assert "critic_loss" in stats
    assert "ent_critic_loss" in stats
    assert "alpha" in stats
    assert "q_e_mean" in stats


def test_tecrl_save_load(tmp_path):
    """TECRL save/load should round-trip entropy critic state."""
    cfg = SACConfig(use_tecrl=True, buffer_size=100)
    agent = SACAgent(cfg, device="cpu", seed=0)
    path = tmp_path / "tecrl.pt"
    agent.save(path)
    agent2 = SACAgent(cfg, device="cpu", seed=1)
    agent2.load(path)
    # Check entropy critic weights match
    for p1, p2 in zip(agent.entropy_critic.parameters(),
                       agent2.entropy_critic.parameters()):
        assert torch.allclose(p1, p2)
