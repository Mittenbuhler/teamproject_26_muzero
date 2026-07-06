"""Neural networks used only by the real-state CartPole experiment."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicsModel(nn.Module):
    """Predict the next four-value state from a state and discrete action."""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.state_head = nn.Linear(hidden_dim, state_dim)
        # Kept for compatibility with existing state checkpoints. Planning
        # hardcodes CartPole's +1 reward and this head is not trained or used.
        self.reward_head = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, state, action_one_hot):
        hidden = self.net(torch.cat([state, action_one_hot], dim=1))
        return self.state_head(hidden), self.reward_head(hidden)

    @torch.no_grad()
    def predict(self, state, action):
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=device
        ).view(1, -1)
        action_tensor = torch.zeros(
            (1, self.action_dim), dtype=torch.float32, device=device
        )
        action_tensor[0, int(action)] = 1.0
        next_state, reward = self.forward(state_tensor, action_tensor)
        if was_training:
            self.train()
        return next_state.squeeze(0).cpu().numpy(), float(reward.item())


class PolicyNetwork(nn.Module):
    """Predict a probability distribution over CartPole's two actions."""

    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, state):
        return F.softmax(self.net(state), dim=-1)

    @torch.no_grad()
    def action_probs(self, state):
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        tensor = torch.as_tensor(state, dtype=torch.float32, device=device).view(1, -1)
        probabilities = self.forward(tensor).squeeze(0).cpu().numpy()
        if was_training:
            self.train()
        probabilities = np.maximum(probabilities, 0.0)
        total = probabilities.sum()
        if not np.isfinite(total) or total <= 0:
            return np.ones(self.action_dim, dtype=np.float32) / self.action_dim
        return probabilities / total


class ValueNetwork(nn.Module):
    """Predict normalized future survival from a real CartPole state."""

    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, state):
        return self.net(state)

    @torch.no_grad()
    def value(self, state):
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        tensor = torch.as_tensor(state, dtype=torch.float32, device=device).view(1, -1)
        value = float(self.forward(tensor).item())
        if was_training:
            self.train()
        return value
