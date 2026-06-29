import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("torch")
pytest.importorskip("functorch")

import alphazero.policy_player_MCTS as policy_player
from alphazero.mcts_agent_policyValue import MCTSNode


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


class DummyRootNode(MCTSNode):
    """Small helper that exposes visit-count statistics for policy tests."""

    def __init__(self, visit_counts):
        super().__init__(
            DummyEnv(),
            False,
            None,
            np.zeros(2, dtype=np.float32),
            0,
            "Dummy",
        )
        self.child = {}
        for action, visits in visit_counts.items():
            child = MCTSNode(
                DummyEnv(),
                False,
                self,
                np.zeros(2, dtype=np.float32),
                action,
                "Dummy",
            )
            child.N = visits
            self.child[action] = child


def test_policy_player_temperature_control(monkeypatch):
    """Temperature should switch between deterministic argmax and stochastic sampling."""
    root = DummyRootNode({0: 80, 1: 20})

    monkeypatch.setattr(policy_player, "MCTS_POLICY_EXPLORE", 1)

    def fake_explore():
        return None

    monkeypatch.setattr(root, "explore", fake_explore)

    def fake_next():
        return root.child[0], 0

    monkeypatch.setattr(root, "next", fake_next)

    monkeypatch.setattr(root, "visit_count_policy", lambda temperature=1.0: np.array([1.0, 0.0], dtype=np.float32))

    policy_player.Policy_Player_MCTS(root)

    assert root.child[0].N == 80
    assert root.child[1].N == 20
