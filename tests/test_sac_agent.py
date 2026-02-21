"""Tests for SACAgent with replay buffer and BC pretraining."""
import numpy as np
from forest_vehicle_dqn.sac_agent import SACAgent, SACConfig


def test_agent_act_returns_continuous():
    cfg = SACConfig(map_size=48, map_channels=3, scalar_dim=12)
    agent = SACAgent(cfg, device="cpu")
    obs = {"maps": np.random.randn(3, 48, 48).astype(np.float32),
           "scalars": np.random.randn(12).astype(np.float32)}
    action = agent.act(obs, explore=True)
    assert action.shape == (2,)
    assert np.all(action >= -1) and np.all(action <= 1)


def test_agent_update_returns_losses():
    cfg = SACConfig(map_size=48, map_channels=3, scalar_dim=12, batch_size=4)
    agent = SACAgent(cfg, device="cpu")
    obs = {"maps": np.random.randn(3, 48, 48).astype(np.float32),
           "scalars": np.random.randn(12).astype(np.float32)}
    for _ in range(8):
        action = agent.act(obs, explore=True)
        agent.observe(obs, action, -1.0, obs, False)
    losses = agent.update()
    assert "critic_loss" in losses
    assert "actor_loss" in losses
    assert "alpha" in losses


def test_agent_save_load(tmp_path):
    cfg = SACConfig(map_size=48, map_channels=3, scalar_dim=12)
    agent = SACAgent(cfg, device="cpu")
    path = tmp_path / "sac_test.pt"
    agent.save(path)
    agent2 = SACAgent(cfg, device="cpu")
    agent2.load(path)
