import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("torch")
pytest.importorskip("functorch")

from alphazero.CartPole_alphazero import make_env


def test_cartpole_make_env_matches_expected_spaces():
    """CartPole runner should create observations/actions matching its config."""
    env = make_env()
    try:
        observation, _ = env.reset()

        assert env.action_space.n == 2
        assert observation.shape == (4,)
    finally:
        env.close()
