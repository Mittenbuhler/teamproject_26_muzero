import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("matplotlib")
pytest.importorskip("torch")
pytest.importorskip("functorch")

from alphazero.FrozenLake_alphazero import make_env


def test_frozen_lake_make_env_returns_one_hot_observations():
    """FrozenLake runner should wrap integer states as one-hot vectors."""
    env = make_env()
    try:
        observation, _ = env.reset()

        assert env.action_space.n == 4
        assert observation.shape == (16,)
        assert np.isclose(observation.sum(), 1.0)
        assert set(np.unique(observation)).issubset({0.0, 1.0})
    finally:
        env.close()
