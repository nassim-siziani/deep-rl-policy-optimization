"""Benchmark comparatif entre Double Dueling DQN et Policy Gradient (REINFORCE + Baseline)."""

import argparse
from pathlib import Path
import random
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.agents import DoubleDuelingDQNAgent, ReinforceBaselineAgent


def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_double_dueling_dqn(episodes: int = 200, seed: int = 42) -> list[float]:
    set_seeds(seed)
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DoubleDuelingDQNAgent(state_dim, action_dim, lr=1e-3, gamma=0.99)
    rewards_history = []

    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        epsilon = max(0.02, 0.9 * (0.97 ** ep))
        ep_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state, epsilon=epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.memory.push(state, action, reward, next_state, float(terminated))
            agent.update(batch_size=64)
            state = next_state
            ep_reward += reward

        rewards_history.append(ep_reward)
        if (ep + 1) % 50 == 0:
            mean_50 = np.mean(rewards_history[-50:])
            print(f"[Double Dueling DQN] Épisode {ep+1:3d}/{episodes} | Récompense moyenne (50 dern.) : {mean_50:6.1f}")

    env.close()
    return rewards_history


def train_reinforce_baseline(episodes: int = 200, seed: int = 42) -> list[float]:
    set_seeds(seed)
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = ReinforceBaselineAgent(state_dim, action_dim, lr=3e-3, gamma=0.99)
    rewards_history = []

    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.rewards.append(reward)
            state = next_state
            ep_reward += reward

        agent.finish_episode()
        rewards_history.append(ep_reward)
        if (ep + 1) % 50 == 0:
            mean_50 = np.mean(rewards_history[-50:])
            print(f"[REINFORCE+Baseline] Épisode {ep+1:3d}/{episodes} | Récompense moyenne (50 dern.) : {mean_50:6.1f}")

    env.close()
    return rewards_history


def moving_average(values: list[float], window: int = 15) -> np.ndarray:
    if len(values) < window:
        return np.array(values)
    return np.convolve(values, np.ones(window) / window, mode="valid")


def main(episodes: int = 200, output_plot: str = "results/convergence_comparison.png"):
    print("=== Lancement du Benchmark d'Optimisation Séquentielle (Gymnasium CartPole-v1) ===")
    dqn_rewards = train_double_dueling_dqn(episodes=episodes)
    pg_rewards = train_reinforce_baseline(episodes=episodes)

    Path(output_plot).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(moving_average(dqn_rewards), label="Double Dueling DQN (Value-Based)", linewidth=2)
    plt.plot(moving_average(pg_rewards), label="REINFORCE + Baseline (Policy Gradient)", linewidth=2, linestyle="--")
    plt.axhline(500, color="gray", linestyle=":", label="Score Optimal (500)")
    plt.xlabel("Épisodes d'entraînement")
    plt.ylabel("Récompense cumulée (Moyenne mobile 15 ép.)")
    plt.title("Comparaison de Convergence : Double Dueling DQN vs Policy Gradient")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)
    print(f"\nGraphique de convergence sauvegardé dans : {output_plot}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark RL : Double Dueling DQN vs Policy Gradient.")
    parser.add_argument("--episodes", type=int, default=200, help="Nombre d'épisodes d'entraînement")
    args = parser.parse_args()
    main(episodes=args.episodes)