"""Small spatial networks implementing MuZero's h, g and f functions."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def normalize_latent(x, eps=1e-5):
    """Independently min-max normalize every sample to [0, 1]."""
    flat = x.flatten(1)
    low = flat.min(1).values.view(-1, 1, 1, 1)
    high = flat.max(1).values.view(-1, 1, 1, 1)
    return (x - low) / (high - low).clamp_min(eps)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return F.relu(x + self.conv2(F.relu(self.conv1(x))))


class RepresentationNetwork(nn.Module):
    """h: a real native observation history [B,C,H,W] -> spatial latent."""
    def __init__(self, input_channels=5, latent_channels=32, observation_shape=(32, 32)):
        super().__init__()
        self.input_channels = int(input_channels)
        self.latent_channels = int(latent_channels)
        self.observation_shape = tuple(observation_shape)
        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_channels, self.latent_channels, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(self.latent_channels, self.latent_channels, 3, stride=2, padding=1), nn.ReLU(),
            ResidualBlock(self.latent_channels), ResidualBlock(self.latent_channels),
        )

    @property
    def latent_shape(self):
        with torch.no_grad():
            device = next(self.parameters()).device
            x = torch.zeros(
                1,
                self.input_channels,
                *self.observation_shape,
                device=device,
            )
            return tuple(self.forward(x).shape[1:])

    def forward(self, observation):
        return normalize_latent(self.encoder(observation))

    @torch.no_grad()
    def encode(self, observation):
        device = next(self.parameters()).device
        x = torch.as_tensor(observation, dtype=torch.float32, device=device)
        if x.ndim == 3: x = x.unsqueeze(0)
        return self(x).squeeze(0).cpu().numpy()


class DynamicsModel(nn.Module):
    """g: latent plus tiled one-hot action -> reward and next latent."""
    def __init__(self, latent_channels, action_dim, hidden_channels=32):
        super().__init__()
        self.latent_channels, self.action_dim = int(latent_channels), int(action_dim)
        self.hidden_channels = int(hidden_channels)
        self.trunk = nn.Sequential(
            nn.Conv2d(self.latent_channels + self.action_dim, hidden_channels, 3, padding=1), nn.ReLU(),
            ResidualBlock(hidden_channels),
        )
        self.state_head = nn.Conv2d(hidden_channels, self.latent_channels, 3, padding=1)
        self.reward_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(hidden_channels, 1))

    def forward(self, state, action):
        if action.ndim == 2: action = action.argmax(1)
        planes = F.one_hot(action.long(), self.action_dim).float()[:, :, None, None]
        planes = planes.expand(-1, -1, state.shape[-2], state.shape[-1])
        hidden = self.trunk(torch.cat((state, planes), 1))
        return normalize_latent(self.state_head(hidden)), self.reward_head(hidden)

    @torch.no_grad()
    def predict(self, state, action):
        device = next(self.parameters()).device
        x = torch.as_tensor(state, dtype=torch.float32, device=device)
        if x.ndim == 3: x = x.unsqueeze(0)
        nxt, reward = self(x, torch.tensor([action], device=device))
        return nxt.squeeze(0).cpu().numpy(), float(reward.item())


class PredictionNetwork(nn.Module):
    """f: a position-preserving spatial latent readout for policy and value."""
    def __init__(
        self,
        latent_shape,
        action_dim,
        hidden_channels=32,
        policy_channels=2,
        value_channels=1,
    ):
        super().__init__()
        self.latent_shape = tuple(int(dimension) for dimension in latent_shape)
        if len(self.latent_shape) != 3 or any(
            dimension <= 0 for dimension in self.latent_shape
        ):
            raise ValueError("latent_shape must be positive (channels, height, width)")
        self.latent_channels, latent_height, latent_width = self.latent_shape
        self.action_dim = int(action_dim)
        self.hidden_channels = int(hidden_channels)
        self.policy_channels = int(policy_channels)
        self.value_channels = int(value_channels)
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if min(self.hidden_channels, self.policy_channels, self.value_channels) <= 0:
            raise ValueError("prediction head channel counts must be positive")

        self.policy_conv = nn.Conv2d(
            self.latent_channels,
            self.policy_channels,
            kernel_size=1,
        )
        self.policy_head = nn.Linear(
            self.policy_channels * latent_height * latent_width,
            self.action_dim,
        )
        # Until the first valid post-warm-up policy targets arrive, expose a
        # neutral prior instead of an arbitrary initialization-specific bias.
        nn.init.zeros_(self.policy_head.weight)
        nn.init.zeros_(self.policy_head.bias)
        self.value_conv = nn.Conv2d(
            self.latent_channels,
            self.value_channels,
            kernel_size=1,
        )
        self.value_hidden = nn.Linear(
            self.value_channels * latent_height * latent_width,
            self.hidden_channels,
        )
        self.value_head = nn.Linear(self.hidden_channels, 1)

    def forward(self, state):
        if tuple(state.shape[1:]) != self.latent_shape:
            raise ValueError(
                f"Expected latent shape {self.latent_shape}, "
                f"got {tuple(state.shape[1:])}"
            )
        policy_features = F.relu(self.policy_conv(state)).flatten(1)
        value_features = F.relu(self.value_conv(state)).flatten(1)
        value_features = F.relu(self.value_hidden(value_features))
        return self.policy_head(policy_features), self.value_head(value_features)

    @torch.no_grad()
    def predict(self, state):
        device = next(self.parameters()).device
        x = torch.as_tensor(state, dtype=torch.float32, device=device)
        if x.ndim == 3: x = x.unsqueeze(0)
        logits, value = self(x)
        return torch.softmax(logits, -1).squeeze(0).cpu().numpy(), float(value.item())


# Compatibility wrappers for callers that kept the two heads separate.
class PolicyNetwork(PredictionNetwork):
    def forward(self, state): return super().forward(state)[0]
    def action_probs(self, state):
        device = next(self.parameters()).device
        x = torch.as_tensor(state, dtype=torch.float32, device=device)
        if x.ndim == 3: x = x.unsqueeze(0)
        return torch.softmax(self(x), -1).squeeze(0).detach().cpu().numpy()


class ValueNetwork(PredictionNetwork):
    def __init__(self, latent_shape, hidden_channels=32):
        super().__init__(latent_shape, 1, hidden_channels)
    def forward(self, state): return super().forward(state)[1]
    def value(self, state):
        device = next(self.parameters()).device
        x = torch.as_tensor(state, dtype=torch.float32, device=device)
        if x.ndim == 3: x = x.unsqueeze(0)
        return float(self(x).item())


# The legacy package used this name; retaining it avoids needless downstream
# breakage while the native package exposes the environment-neutral class name.
ImageRepresentationNetwork = RepresentationNetwork
