"""Architectures neuronales PyTorch pour l'optimisation de politiques et de fonctions de valeur."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingQNetwork(nn.Module):
    """
    Architecture Dueling DQN séparant l'estimation de la valeur de l'état V(s)
    et l'avantage relatif de chaque action A(s, a).
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        # Branche de valeur d'état V(s) -> scalaire
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Branche d'avantage A(s, a) -> vecteur de taille |A|
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        # Identifiabilité mathématique : soustraction de l'avantage moyen
        q_values = values + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_values


class PolicyBaselineNetwork(nn.Module):
    """
    Réseau conjoint Politique stochastique pi(a|s) et Baseline d'état V(s)
    pour l'algorithme de Policy Gradient avec réduction de variance.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(state)
        action_probs = F.softmax(self.policy_head(h), dim=-1)
        state_value = self.value_head(h)
        return action_probs, state_value