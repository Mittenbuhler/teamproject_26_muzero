import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicsModel(nn.Module):
    """Predicts the next latent state and immediate reward from latent + action."""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.state_head = nn.Linear(hidden_dim, state_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, state, action_one_hot):
        x = torch.cat([state, action_one_hot], dim=1)
        h = self.net(x)
        return self.state_head(h), self.reward_head(h)

    def loss_details(self, state, action_one_hot, next_state, reward):
        pred_next_state, pred_reward = self.forward(state, action_one_hot)
        state_loss = F.mse_loss(pred_next_state, next_state)
        reward_loss = F.mse_loss(pred_reward, reward)
        return state_loss + reward_loss, state_loss, reward_loss

    @torch.no_grad()
    def predict(self, state, action):
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).view(1, -1)
        action_tensor = torch.zeros((1, self.action_dim), dtype=torch.float32, device=device)
        action_tensor[0, int(action)] = 1.0
        next_state, reward = self.forward(state_tensor, action_tensor)

        if was_training:
            self.train()

        return next_state.squeeze(0).cpu().numpy(), float(reward.item())


class PolicyNetwork(nn.Module):
    """Discrete policy over actions."""

    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
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
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).view(1, -1)
        probs = self.forward(state_tensor).squeeze(0).cpu().numpy()

        if was_training:
            self.train()

        probs = np.maximum(probs, 0.0)
        total = probs.sum()
        if not np.isfinite(total) or total <= 0:
            return np.ones(self.action_dim, dtype=np.float32) / self.action_dim
        return probs / total

class ValueNetwork(nn.Module):
    """Predicts normalized return in [-1, 1]."""

    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
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
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).view(1, -1)
        value = float(self.forward(state_tensor).item())

        if was_training:
            self.train()

        return value


class ImageRepresentationNetwork(nn.Module):
    """
    Representation function x[t-4:t] -> s[t].

    Five grayscale 32x32 observations become one 32-dimensional latent state.
    Missing early observations are zero-filled by FrameStack.
    """

    def __init__(self, input_channels=5, latent_dim=32):
        super().__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, observation):
        return self.encoder(observation)

    @torch.no_grad()
    def encode(self, observation):
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        observation = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=device,
        )
        if observation.ndim == 3:
            observation = observation.unsqueeze(0)
        latent = self.forward(observation)
        if was_training:
            self.train()
        return latent.squeeze(0).cpu().numpy()
