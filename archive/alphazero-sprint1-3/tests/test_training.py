import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("torch")
pytest.importorskip("functorch")

import alphazero.training as training


class TinyDiscreteSpace:
    """Minimal discrete action space used by TinyEnv."""

    def __init__(self, n):
        self.n = n

    def sample(self):
        return 0


class TinyObservationSpace:
    """Minimal observation space with the shape expected by training.py."""

    shape = (2,)


class TinyEnv:
    """One-step deterministic environment for a fast training smoke test."""

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
        reward = 1.0 if action == 0 else 0.0
        terminated = self.steps >= 1
        truncated = False
        return self.state.copy(), reward, terminated, truncated, {}

    def close(self):
        pass


def test_train_returns_episode_metrics_for_tiny_environment(monkeypatch):
    """The training loop should return metrics and perform one NN update."""
    # Use tiny buffers so a two-episode run is enough to trigger training.
    monkeypatch.setattr(training, "BATCH_SIZE", 1)
    monkeypatch.setattr(training, "BUFFER_SIZE", 4)

    def fake_policy_player(mytree):
        # Avoid expensive MCTS here; this test focuses on training.py plumbing.
        policy = np.array([1.0, 0.0], dtype=np.float32)
        return (
            mytree,
            0,
            np.array([0.0, 1.0], dtype=np.float32),
            policy,
            np.array(mytree.observation, dtype=np.float32),
        )

    monkeypatch.setattr(training, "Policy_Player_MCTS", fake_policy_player)

    config = {
        "game_name": "TinyEnv",
        "input_dim": 2,
        "action_dim": 2,
        "max_reward": 1,
        "episodes": 2,
        "make_env": TinyEnv,
    }

    rewards, moving_average, value_losses, policy_losses = training.train(config)

    assert rewards == [1.0, 1.0]
    assert moving_average == [1.0, 1.0]
    assert len(value_losses) == 1
    assert len(policy_losses) == 1
    assert np.isfinite(value_losses).all()
    assert np.isfinite(policy_losses).all()
