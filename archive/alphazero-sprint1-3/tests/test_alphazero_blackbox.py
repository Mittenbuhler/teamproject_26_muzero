import sys
from pathlib import Path

import numpy as np
import torch
import pytest


gym = pytest.importorskip("gymnasium")
torch = pytest.importorskip("torch")
pytest.importorskip("functorch")

# Add project root to path so alphazero package can be imported
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import alphazero.policy_player_MCTS as policy_player
import alphazero.training as training


class FrozenLakeOneHot(gym.ObservationWrapper):
    """Match FrozenLake observations to the neural-network input size."""

    def observation(self, obs):
        one_hot = np.zeros(16, dtype=np.float32)
        one_hot[obs] = 1.0
        return one_hot


def make_frozen_lake():
    """Factory passed into train(), just like the real runner does."""
    return FrozenLakeOneHot(gym.make("FrozenLake-v1", is_slippery=False))


def test_frozen_lake_training_smoke_blackbox(monkeypatch):
    """Run a tiny end-to-end FrozenLake training pass from the outside."""
    # Reduce MCTS work so this stays a smoke test, not a long training run.
    monkeypatch.setattr(policy_player, "MCTS_POLICY_EXPLORE", 2)

    config = {
        "game_name": "FrozenLake-v1",
        "input_dim": 16,
        "action_dim": 4,
        "max_reward": 1,
        "episodes": 1,
        "make_env": make_frozen_lake,
    }

    rewards, moving_average, value_losses, policy_losses = training.train(config)

    assert len(rewards) == 1
    assert len(moving_average) == 1
    assert np.isfinite(rewards).all()
    assert np.isfinite(moving_average).all()
    assert all(np.isfinite(loss) for loss in value_losses)
    assert all(np.isfinite(loss) for loss in policy_losses)


class TinyDiscreteSpace:
    """Minimal discrete action space for the NN-learning blackbox test."""

    def __init__(self, n):
        self.n = n

    def sample(self):
        return 0


class TinyObservationSpace:
    """Observation space shape expected by the AlphaZero networks."""

    shape = (2,)


class TinyEnv:
    """One-step environment so every episode quickly reaches training."""

    action_space = TinyDiscreteSpace(2)
    observation_space = TinyObservationSpace()

    @property
    def unwrapped(self):
        return self

    def reset(self):
        self.state = np.array([1.0, 0.0], dtype=np.float32)
        self.steps = 0
        return self.state.copy(), {}

    def step(self, action):
        self.steps += 1
        self.state = np.array([0.0, 1.0], dtype=np.float32)
        reward = 1.0
        terminated = self.steps >= 1
        truncated = False
        return self.state.copy(), reward, terminated, truncated, {}

    def close(self):
        pass


def flatten_parameters(module):
    """Flatten a module's parameters so before/after weights are comparable."""
    return torch.cat([
        parameter.detach().cpu().flatten()
        for parameter in module.parameters()
    ])


def test_training_updates_networks_and_mcts_uses_same_updated_objects(monkeypatch):
    """Check that learning persists and MCTS receives the trained networks."""
    # A batch size of 1 means episode 2 already performs an optimizer update.
    # Episode 3 then proves the next MCTS tree sees the updated network objects.
    monkeypatch.setattr(training, "BATCH_SIZE", 1)
    monkeypatch.setattr(training, "BUFFER_SIZE", 8)

    created_networks = []
    original_networks_class = training.AlphaZeroNetworks

    class RecordingNetworks(original_networks_class):
        """Store initial parameters from the internally created networks."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.initial_policy_parameters = flatten_parameters(
                self.policy_network
            )
            self.initial_value_parameters = flatten_parameters(
                self.value_network
            )
            created_networks.append(self)

    seen_policy_ids = []
    seen_value_ids = []
    seen_policy_parameters = []

    def fake_policy_player(mytree):
        # This is the blackbox observation point: training.py passes its
        # networks into MCTSNode, and Policy_Player_MCTS receives that tree.
        seen_policy_ids.append(id(mytree.policy_network))
        seen_value_ids.append(id(mytree.value_network))
        seen_policy_parameters.append(flatten_parameters(mytree.policy_network))

        policy = np.array([0.0, 1.0], dtype=np.float32)
        return (
            mytree,
            0,
            np.array([0.0, 1.0], dtype=np.float32),
            policy,
            np.array(mytree.observation, dtype=np.float32),
        )

    monkeypatch.setattr(training, "AlphaZeroNetworks", RecordingNetworks)
    monkeypatch.setattr(training, "Policy_Player_MCTS", fake_policy_player)

    config = {
        "game_name": "TinyEnv",
        "input_dim": 2,
        "action_dim": 2,
        "max_reward": 1,
        "episodes": 3,
        "make_env": TinyEnv,
    }

    rewards, moving_average, value_losses, policy_losses = training.train(config)

    networks = created_networks[0]
    final_policy_parameters = flatten_parameters(networks.policy_network)
    final_value_parameters = flatten_parameters(networks.value_network)

    assert len(rewards) == 3
    assert len(moving_average) == 3
    assert len(value_losses) >= 1
    assert len(policy_losses) >= 1

    assert not torch.allclose(
        networks.initial_policy_parameters,
        final_policy_parameters,
    )
    assert not torch.allclose(
        networks.initial_value_parameters,
        final_value_parameters,
    )

    assert all(
        policy_id == id(networks.policy_network)
        for policy_id in seen_policy_ids
    )
    assert all(
        value_id == id(networks.value_network)
        for value_id in seen_value_ids
    )

    assert not torch.allclose(
        seen_policy_parameters[0],
        seen_policy_parameters[-1],
    )
