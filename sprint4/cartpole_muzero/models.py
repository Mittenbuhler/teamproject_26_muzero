"""Fully connected MuZero networks for CartPole's four-number observation."""

import torch
from torch import nn
import torch.nn.functional as F


def normalize_latent(state, eps=1e-5):
    """Normalize each latent vector to [0, 1], including constant vectors."""
    low = state.amin(dim=1, keepdim=True)
    high = state.amax(dim=1, keepdim=True)
    return (state - low) / (high - low).clamp_min(eps)


def _check_batch(value, dimension, name):
    if value.ndim != 2 or value.shape[1] != dimension:
        raise ValueError(f"{name} must have shape [batch, {dimension}], got {tuple(value.shape)}")


def _single_input(model, value):
    device = next(model.parameters()).device
    return torch.as_tensor(value, dtype=torch.float32, device=device).unsqueeze(0)


class RepresentationNetwork(nn.Module):
    """h: [x, x_dot, theta, theta_dot] -> a learned latent vector."""

    input_dim = 4

    def __init__(self, latent_dim=64, hidden_dim=128):
        super().__init__()
        if latent_dim < 2 or hidden_dim < 1:
            raise ValueError("latent_dim must be at least 2 and hidden_dim must be positive")
        self.latent_dim, self.hidden_dim = int(latent_dim), int(hidden_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )

    def forward(self, observation):
        _check_batch(observation, self.input_dim, "observation")
        return normalize_latent(self.encoder(observation))

    @torch.no_grad()
    def encode(self, observation):
        return self(_single_input(self, observation)).squeeze(0).cpu().numpy()


class DynamicsModel(nn.Module):
    """g: latent plus one-hot action -> next latent and immediate reward."""

    def __init__(self, latent_dim=64, action_dim=2, hidden_dim=128):
        super().__init__()
        if latent_dim < 2 or min(action_dim, hidden_dim) < 1:
            raise ValueError("invalid dynamics dimensions")
        self.latent_dim, self.action_dim = int(latent_dim), int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.trunk = nn.Sequential(
            nn.Linear(self.latent_dim + self.action_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
        )
        self.state_head = nn.Linear(self.hidden_dim, self.latent_dim)
        self.reward_head = nn.Linear(self.hidden_dim, 1)

    def forward(self, state, action):
        _check_batch(state, self.latent_dim, "latent")
        if action.ndim == 2:
            action = action.argmax(dim=1)
        action_features = F.one_hot(action.long(), self.action_dim).to(state.dtype)
        hidden = self.trunk(torch.cat((state, action_features), dim=1))
        return normalize_latent(self.state_head(hidden)), self.reward_head(hidden)

    @torch.no_grad()
    def predict(self, state, action):
        state = _single_input(self, state)
        next_state, reward = self(state, torch.tensor([action], device=state.device))
        return next_state.squeeze(0).cpu().numpy(), float(reward.item())


class PredictionNetwork(nn.Module):
    """f: latent -> policy logits and an unconstrained scalar value."""

    def __init__(self, latent_dim=64, action_dim=2, hidden_dim=128):
        super().__init__()
        if latent_dim < 2 or min(action_dim, hidden_dim) < 1:
            raise ValueError("invalid prediction dimensions")
        self.latent_dim, self.action_dim = int(latent_dim), int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.policy_hidden = nn.Linear(self.latent_dim, self.hidden_dim)
        self.policy_head = nn.Linear(self.hidden_dim, self.action_dim)
        self.value_hidden = nn.Linear(self.latent_dim, self.hidden_dim)
        self.value_head = nn.Linear(self.hidden_dim, 1)
        # Warm-up starts with neutral priors, without teaching a random policy bias.
        nn.init.zeros_(self.policy_head.weight)
        nn.init.zeros_(self.policy_head.bias)

    def forward(self, state):
        _check_batch(state, self.latent_dim, "latent")
        logits = self.policy_head(F.relu(self.policy_hidden(state)))
        value = self.value_head(F.relu(self.value_hidden(state)))
        return logits, value

    @torch.no_grad()
    def predict(self, state):
        logits, value = self(_single_input(self, state))
        return torch.softmax(logits, -1).squeeze(0).cpu().numpy(), float(value.item())
