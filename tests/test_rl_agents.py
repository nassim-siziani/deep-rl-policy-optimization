"""Tests unitaires validant les propriétés mathématiques des réseaux et agents RL."""

import numpy as np
import torch
from src.agents import DoubleDuelingDQNAgent, ReinforceBaselineAgent
from src.networks import DuelingQNetwork, PolicyBaselineNetwork


def test_dueling_q_network_output_shape():
    net = DuelingQNetwork(state_dim=4, action_dim=2)
    x = torch.randn(10, 4)
    q = net(x)
    assert q.shape == (10, 2)


def test_policy_network_probability_simplex():
    net = PolicyBaselineNetwork(state_dim=4, action_dim=3)
    x = torch.randn(8, 4)
    probs, values = net(x)
    assert probs.shape == (8, 3)
    assert values.shape == (8, 1)
    # La somme des probabilités d'action doit valoir exactement 1.0 pour chaque état
    np.testing.assert_allclose(probs.detach().numpy().sum(axis=1), np.ones(8), atol=1e-5)


def test_dqn_replay_buffer_and_update():
    agent = DoubleDuelingDQNAgent(state_dim=4, action_dim=2)
    for _ in range(70):
        s = np.zeros(4, dtype=np.float32)
        ns = np.ones(4, dtype=np.float32)
        agent.memory.push(s, 1, 1.0, ns, 0.0)
    loss = agent.update(batch_size=32)
    assert loss is not None
    assert loss >= 0.0


def test_reinforce_episode_update():
    agent = ReinforceBaselineAgent(state_dim=4, action_dim=2)
    for _ in range(5):
        s = np.random.randn(4).astype(np.float32)
        _ = agent.select_action(s)
        agent.rewards.append(1.0)
    loss = agent.finish_episode()
    assert isinstance(loss, float)