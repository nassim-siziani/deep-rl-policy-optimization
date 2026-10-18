"""Algorithmes d'optimisation séquentielle : Double Dueling DQN et Policy Gradient avec Baseline."""

from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from src.networks import DuelingQNetwork, PolicyBaselineNetwork


class ReplayBuffer:
    """Mémoire de rejeu (Experience Replay) brisant la corrélation temporelle des trajectoires."""

    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DoubleDuelingDQNAgent:
    """Agent implémentant Double Q-Learning avec architecture Dueling et Huber Loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.01,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        self.online_net = DuelingQNetwork(state_dim, action_dim)
        self.target_net = DuelingQNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.optimizer = optim.AdamW(self.online_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(capacity=10_000)
        self.loss_fn = nn.SmoothL1Loss()

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def update(self, batch_size: int = 64) -> float | None:
        if len(self.memory) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states_t = torch.tensor(states)
        actions_t = torch.tensor(actions).unsqueeze(1)
        rewards_t = torch.tensor(rewards).unsqueeze(1)
        next_states_t = torch.tensor(next_states)
        dones_t = torch.tensor(dones).unsqueeze(1)

        current_q = self.online_net(states_t).gather(1, actions_t)

        # Double DQN : sélection de l'action via online_net, évaluation via target_net
        with torch.no_grad():
            best_next_actions = self.online_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, best_next_actions)
            target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Mise à jour douce (Polyak averaging) du réseau cible
        for target_param, online_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)

        return float(loss.item())


class ReinforceBaselineAgent:
    """Agent Policy Gradient (REINFORCE) avec baseline d'état V(s) pour réduire la variance."""

    def __init__(self, state_dim: int, action_dim: int, lr: float = 2e-3, gamma: float = 0.99):
        self.gamma = gamma
        self.net = PolicyBaselineNetwork(state_dim, action_dim)
        self.optimizer = optim.AdamW(self.net.parameters(), lr=lr)

        self.saved_log_probs = []
        self.saved_values = []
        self.rewards = []

    def select_action(self, state: np.ndarray) -> int:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs, state_val = self.net(state_t)
        dist = Categorical(probs)
        action = dist.sample()

        self.saved_log_probs.append(dist.log_prob(action))
        self.saved_values.append(state_val.squeeze(0))
        return int(action.item())

    def finish_episode(self) -> float:
        returns = []
        discounted_sum = 0.0
        for r in reversed(self.rewards):
            discounted_sum = r + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)

        returns_t = torch.tensor(returns, dtype=torch.float32)
        if len(returns_t) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        policy_losses = []
        value_losses = []

        for log_prob, value, R in zip(self.saved_log_probs, self.saved_values, returns_t):
            advantage = R - value.item()
            policy_losses.append(-log_prob * advantage)
            value_losses.append(nn.functional.smooth_l1_loss(value, torch.tensor([R])))

        loss = torch.stack(policy_losses).sum() + 0.5 * torch.stack(value_losses).sum()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.saved_log_probs.clear()
        self.saved_values.clear()
        self.rewards.clear()
        return float(loss.item())