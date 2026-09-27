import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("torch")
pytest.importorskip("functorch")

from alphazero.mcts_agent_policyValue import MCTSNode
from alphazero.value_and_policy_NN import AlphaZeroNetworks, ActionSpaceType
import alphazero.policy_player_MCTS as policy_player


class FrozenLakeOneHot(gym.ObservationWrapper):
    """Convert FrozenLake's integer observation to a one-hot vector."""

    def observation(self, obs):
        one_hot = np.zeros(16, dtype=np.float32)
        one_hot[obs] = 1.0
        return one_hot


def make_frozen_lake():
    """Use a deterministic environment for predictable MCTS tests."""
    return FrozenLakeOneHot(gym.make("FrozenLake-v1", is_slippery=False))


def test_policy_player_returns_tree_action_observations_and_policy(monkeypatch):
    """Policy_Player_MCTS should return everything needed by training.py."""
    # Keep the test fast; we only need enough exploration to create children.
    monkeypatch.setattr(policy_player, "MCTS_POLICY_EXPLORE", 2)

    env = make_frozen_lake()
    observation, _ = env.reset()
    networks = AlphaZeroNetworks(
        action_space_type=ActionSpaceType.DISCRETE,
        action_dim=4,
        hidden_states=16,
        input_dim=16,
        device="cpu",
    )
    root = MCTSNode(
        env,
        False,
        None,
        observation,
        0,
        "FrozenLake-v1",
        env_factory=make_frozen_lake,
        policy_network=networks.policy_network,
        value_network=networks.value_network,
    )

    try:
        next_tree, action, next_observation, policy, prev_observation = (
            policy_player.Policy_Player_MCTS(root)
        )

        assert action in range(env.action_space.n)
        assert next_tree.parent is None
        assert next_observation.shape == (16,)
        assert prev_observation.shape == (16,)
        assert policy.shape == (4,)
        assert np.isclose(policy.sum(), 1.0)
    finally:
        env.close()
