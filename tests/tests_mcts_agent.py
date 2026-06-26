import numpy as np
import pytest

pytest.importorskip("gymnasium")
torch = pytest.importorskip("torch")
pytest.importorskip("functorch")

import alphazero.mcts_agent_policyValue as mcts_agent


class DummyEnv:
    """Minimal environment stub so MCTSNode can derive an action space."""

    class _ActionSpace:
        def __init__(self, n):
            self.n = n

    class _ObservationSpace:
        def __init__(self):
            self.shape = (2,)

    action_space = _ActionSpace(2)
    observation_space = _ObservationSpace()


def test_clone_env_state_deep_isolation():
    """clone_env_state should create independent copies of environment state."""
    gym = pytest.importorskip("gymnasium")

    cartpole_env = gym.make("CartPole-v1")
    frozenlake_env = gym.make("FrozenLake-v1", is_slippery=False)

    try:
        cartpole_env.reset()
        frozenlake_env.reset()

        cartpole_node = mcts_agent.MCTSNode(
            cartpole_env,
            False,
            None,
            np.zeros(4, dtype=np.float32),
            0,
            "CartPole-v1",
        )
        cloned_cartpole = cartpole_node.clone_env_state(cartpole_env)
        cartpole_state_before = cartpole_env.unwrapped.state.copy()
        cloned_cartpole.step(1)
        assert np.array_equal(cartpole_env.unwrapped.state, cartpole_state_before)

        frozenlake_node = mcts_agent.MCTSNode(
            frozenlake_env,
            False,
            None,
            np.zeros(16, dtype=np.float32),
            0,
            "FrozenLake-v1",
        )
        cloned_frozenlake = frozenlake_node.clone_env_state(frozenlake_env)
        frozenlake_state_before = frozenlake_env.unwrapped.s
        cloned_frozenlake.step(0)
        assert frozenlake_env.unwrapped.s == frozenlake_state_before
    finally:
        cartpole_env.close()
        frozenlake_env.close()


def test_mcts_terminal_state_handling():
    """Rollout should stop on terminal states and skip the value network."""
    gym = pytest.importorskip("gymnasium")

    env = gym.make("CartPole-v1")
    observation, _ = env.reset()

    class RaisingValueNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, state):
            self.calls += 1
            raise AssertionError("The value network should not be queried")

    value_network = RaisingValueNetwork()
    node = mcts_agent.MCTSNode(
        env,
        True,
        None,
        observation,
        0,
        "CartPole-v1",
        reward=1.0,
        value_network=value_network,
    )

    try:
        value = node.rollout()

        assert value == 1.0
        assert node.child is None
        assert value_network.calls == 0
    finally:
        env.close()


def test_mcts_multi_step_discounting_backprop():
    """Backpropagation should discount rollout values at each ancestor level."""
    root = mcts_agent.MCTSNode(
        DummyEnv(),
        False,
        None,
        np.array([0.0], dtype=np.float32),
        0,
        "Dummy",
        discount=0.5,
    )
    child = mcts_agent.MCTSNode(
        DummyEnv(),
        False,
        root,
        np.array([1.0], dtype=np.float32),
        0,
        "Dummy",
        discount=0.5,
    )
    grandchild = mcts_agent.MCTSNode(
        DummyEnv(),
        False,
        child,
        np.array([2.0], dtype=np.float32),
        0,
        "Dummy",
        discount=0.5,
    )

    child.parent = root
    grandchild.parent = child

    def fake_rollout():
        return 1.0

    grandchild.rollout = fake_rollout

    grandchild.explore()

    assert np.isclose(grandchild.T, 1.0)
    assert np.isclose(child.T, 0.5)
    assert np.isclose(root.T, 0.25)
