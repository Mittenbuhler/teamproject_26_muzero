from platform import node

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
torch = pytest.importorskip("torch")
pytest.importorskip("functorch")

from alphazero.mcts_agent_policyValue import MCTSAgent, MCTSNode
from alphazero.value_and_policy_NN import AlphaZeroNetworks, ActionSpaceType


class FrozenLakeOneHot(gym.ObservationWrapper):
    """Convert FrozenLake's integer state into the NN input vector."""

    def observation(self, obs):
        one_hot = np.zeros(16, dtype=np.float32)
        one_hot[obs] = 1.0
        return one_hot


def make_frozen_lake():
    """Use deterministic FrozenLake so MCTS behavior is easier to test."""
    return FrozenLakeOneHot(gym.make("FrozenLake-v1", is_slippery=False))


def make_networks():
    """Create small CPU networks so the MCTS tests stay lightweight."""
    return AlphaZeroNetworks(
        action_space_type=ActionSpaceType.DISCRETE,
        action_dim=4,
        hidden_states=16,
        input_dim=16,
        device="cpu",
    )


def make_root_node():
    """Build a root MCTS node with networks and a wrapped environment."""
    env = make_frozen_lake()
    observation, _ = env.reset()
    networks = make_networks()
    node = MCTSNode(
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
    return node, env


class FixedPolicyNetwork(torch.nn.Module):
    """Tiny policy network with deterministic priors for expansion tests."""

    def __init__(self, priors):
        super().__init__()
        self.dense = torch.nn.Linear(16, 4)
        self.register_buffer("priors", torch.tensor(priors, dtype=torch.float32))

    def forward(self, state):
        return self.priors.unsqueeze(0).repeat(state.shape[0], 1)


class FixedValueNetwork(torch.nn.Module):
    """Tiny value network that returns a known leaf value."""

    def __init__(self, value):
        super().__init__()
        self.dense = torch.nn.Linear(16, 1)
        self.value = value
        self.calls = 0

    def get_value(self, state):
        self.calls += 1
        return self.value


def test_mcts_expands_children_and_builds_visit_count_policy():
    """After exploration, the root should have one child per legal action."""
    node, env = make_root_node()
    try:
        node.explore()
        node.explore()

        assert node.child is not None
        assert set(node.child) == {0, 1, 2, 3}
        assert node.N == 2

        policy = node.visit_count_policy()
        assert policy.shape == (4,)
        assert np.isclose(policy.sum(), 1.0)
        assert np.all(policy >= 0.0)
    finally:
        env.close()


def test_selection_picks_child_with_highest_ucb_score(monkeypatch):
    """Selection should follow the child with the best UCB/PUCT score."""
    node, env = make_root_node()
    try:
        node.create_child()

        low_score_child = node.child[0]
        high_score_child = node.child[1]
        low_score_child.N = 10
        low_score_child.T = 0.0
        high_score_child.N = 1
        high_score_child.T = 5.0

        def chosen_rollout():
            return 7.0

        def wrong_rollout():
            raise AssertionError("Selection picked the lower-scoring child")

        monkeypatch.setattr(high_score_child, "rollout", chosen_rollout)
        monkeypatch.setattr(low_score_child, "rollout", wrong_rollout)

        # Verhindere, dass das High-Score-Child Enkelkinder generiert
        monkeypatch.setattr(high_score_child, "create_child", lambda: None)

        node.explore()

        assert high_score_child.N == 2
        assert high_score_child.T == 12.0
        assert low_score_child.N == 10
    finally:
        env.close()


def test_expansion_creates_one_child_per_action_with_policy_priors():
    """Expansion should create children and copy NN policy priors into them."""
    env = make_frozen_lake()
    observation, _ = env.reset()
    priors = np.array([0.7, 0.1, 0.15, 0.05], dtype=np.float32)
    node = MCTSNode(
        env,
        False,
        None,
        observation,
        0,
        "FrozenLake-v1",
        env_factory=make_frozen_lake,
        policy_network=FixedPolicyNetwork(priors),
    )

    try:
        node.create_child()

        assert set(node.child) == {0, 1, 2, 3}
        for action, child in node.child.items():
            assert child.parent is node
            assert child.action_index == action
            assert np.isclose(child.prior_probability, priors[action])
    finally:
        env.close()


def test_rollout_uses_value_network_instead_of_random_environment_rollout(
    monkeypatch,
):
    """With a value NN, rollout should evaluate the leaf directly."""
    node, env = make_root_node()
    value_network = FixedValueNetwork(value=0.25)
    node.value_network = value_network
    node.reward = 0.5
    node.discount = 0.9

    def fail_if_random_rollout_is_used(game):
        raise AssertionError("Random environment rollout should not happen")

    monkeypatch.setattr(node, "clone_env_state", fail_if_random_rollout_is_used)

    try:
        value = node.rollout()

        assert np.isclose(value, 0.5 + 0.9 * 0.25)
        assert value_network.calls == 1
    finally:
        env.close()


def test_backpropagation_updates_leaf_and_ancestors(monkeypatch):
    """Backpropagation should add rollout value to the leaf and root."""
    node, env = make_root_node()
    try:
        node.create_child()
        selected_child = node.child[2]
        selected_child.N = 0
        selected_child.T = 0.0

        for action, child in node.child.items():
            if action != 2:
                child.N = 100
                child.T = -100.0
            child.prior_probability = 0.0
        selected_child.prior_probability = 1.0

        monkeypatch.setattr(selected_child, "rollout", lambda: 3.5)

        node.explore()

        assert selected_child.N == 1
        assert selected_child.T == 3.5
        assert node.N == 1
        assert node.T == 3.5
    finally:
        env.close()


def test_mcts_next_returns_valid_action_after_search():
    """The selected next action must be valid for the environment."""
    node, env = make_root_node()
    try:
        node.explore()
        node.explore()

        next_node, action = node.next()

        assert action in range(env.action_space.n)
        assert next_node.action_index == action
        assert next_node.parent is node
    finally:
        env.close()


def test_mcts_agent_returns_action_and_policy():
    """The public MCTSAgent API should return both action and policy target."""
    env = make_frozen_lake()
    observation, _ = env.reset()
    networks = make_networks()
    agent = MCTSAgent(
        game_name="FrozenLake-v1",
        env_factory=make_frozen_lake,
        explore_iterations=2,
        networks=networks,
    )

    try:
        action, policy = agent.get_action(
            env,
            observation,
            done=False,
            return_policy=True,
        )

        assert action in range(env.action_space.n)
        assert policy.shape == (4,)
        assert np.isclose(policy.sum(), 1.0)
    finally:
        env.close()


def test_selection_phase_prefers_child_with_highest_ucb_score():
    """Selection should prefer the child with the best UCB value."""
    node, env = make_root_node()
    try:
        node.create_child()
        node.N = 10

        weak_child = node.child[0]
        strong_child = node.child[1]
        weak_child.N = 5
        weak_child.T = 0.0
        strong_child.N = 2
        strong_child.T = 3.0

        best_child = max(node.child.values(), key=lambda child: child.getUCBscore())

        assert best_child is strong_child
    finally:
        env.close()


def test_expansion_phase_creates_one_child_per_action():
    """Expansion should add exactly one child node for each legal action."""
    env = make_frozen_lake()
    observation, _ = env.reset()
    node = MCTSNode(
        env,
        False,
        None,
        observation,
        0,
        "FrozenLake-v1",
        env_factory=make_frozen_lake,
    )

    try:
        node.create_child()

        assert node.child is not None
        assert set(node.child) == {0, 1, 2, 3}
        assert all(child.parent is node for child in node.child.values())
    finally:
        env.close()


def test_rollout_phase_returns_terminal_reward_without_random_steps(monkeypatch):
    """Rollout should return the node reward immediately for terminal states."""
    node, env = make_root_node()
    node.done = True
    node.reward = 2.5

    def fail_if_clone_env_state_is_used(*args, **kwargs):
        raise AssertionError("Random rollout should not be used for terminal states")

    monkeypatch.setattr(node, "clone_env_state", fail_if_clone_env_state_is_used)

    try:
        assert node.rollout() == 2.5
    finally:
        env.close()


def test_backpropagation_phase_updates_leaf_and_ancestors(monkeypatch):
    """Backpropagation should increase the visit counts and totals on the path."""
    root, env = make_root_node()
    try:
        root.create_child()
        leaf = root.child[0]
        leaf.N = 0
        leaf.T = 0.0

        monkeypatch.setattr(leaf, "rollout", lambda: 4.0)
        root.explore()

        assert leaf.N == 1
        assert leaf.T == 4.0
        assert root.N == 1
        assert root.T == 4.0
    finally:
        env.close()
