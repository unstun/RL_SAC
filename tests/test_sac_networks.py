"""Tests for SAC Actor-Critic networks with Global CNN encoder."""
import torch
from forest_vehicle_dqn.sac_networks import GlobalCNNEncoder, SACActor, SACCritic


def test_encoder_output_shape():
    enc = GlobalCNNEncoder(map_size=48, map_channels=3, scalar_dim=12)
    maps = torch.randn(4, 3, 48, 48)
    scalars = torch.randn(4, 12)
    out = enc(maps, scalars)
    assert out.shape == (4, enc.feature_dim)


def test_actor_output_shape():
    enc = GlobalCNNEncoder(map_size=48, map_channels=3, scalar_dim=12)
    actor = SACActor(enc, action_dim=2, hidden_dim=256)
    maps = torch.randn(4, 3, 48, 48)
    scalars = torch.randn(4, 12)
    action, log_prob = actor.sample(maps, scalars)
    assert action.shape == (4, 2)
    assert log_prob.shape == (4,)
    assert (action >= -1).all() and (action <= 1).all()  # tanh squash


def test_critic_output_shape():
    enc = GlobalCNNEncoder(map_size=48, map_channels=3, scalar_dim=12)
    critic = SACCritic(enc, action_dim=2, hidden_dim=256)
    maps = torch.randn(4, 3, 48, 48)
    scalars = torch.randn(4, 12)
    actions = torch.randn(4, 2)
    q1, q2 = critic(maps, scalars, actions)
    assert q1.shape == (4, 1)
    assert q2.shape == (4, 1)
