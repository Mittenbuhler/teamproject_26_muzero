"""Modular Monte Carlo tree search over CartPole's real four-value state."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Protocol

import numpy as np


class TransitionModel(Protocol):
    def step(self, state: np.ndarray, action: int) -> tuple[np.ndarray, float, bool]: ...


class PriorModel(Protocol):
    def probabilities(self, state: np.ndarray) -> np.ndarray: ...


class LeafEvaluator(Protocol):
    def evaluate(self, state: np.ndarray, remaining_depth: int) -> float: ...


class ExactCartPoleDynamics:
    """The deterministic transition used by Gymnasium's CartPoleEnv."""

    def __init__(
        self,
        gravity=9.8,
        masscart=1.0,
        masspole=0.1,
        length=0.5,
        force_mag=10.0,
        tau=0.02,
        theta_threshold_radians=12 * 2 * math.pi / 360,
        x_threshold=2.4,
        kinematics_integrator="euler",
    ):
        self.gravity = gravity
        self.masscart = masscart
        self.masspole = masspole
        self.total_mass = masspole + masscart
        self.length = length
        self.polemass_length = masspole * length
        self.force_mag = force_mag
        self.tau = tau
        self.theta_threshold_radians = theta_threshold_radians
        self.x_threshold = x_threshold
        self.kinematics_integrator = kinematics_integrator

    @classmethod
    def from_env(cls, env):
        source = env.unwrapped
        return cls(
            gravity=source.gravity,
            masscart=source.masscart,
            masspole=source.masspole,
            length=source.length,
            force_mag=source.force_mag,
            tau=source.tau,
            theta_threshold_radians=source.theta_threshold_radians,
            x_threshold=source.x_threshold,
            kinematics_integrator=source.kinematics_integrator,
        )

    def is_terminal(self, state):
        x, _, theta, _ = np.asarray(state)
        return bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

    def step(self, state, action):
        x, x_dot, theta, theta_dot = map(float, state)
        force = self.force_mag if int(action) == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        next_state = np.asarray((x, x_dot, theta, theta_dot), dtype=np.float32)
        return next_state, 1.0, self.is_terminal(next_state)


class LearnedCartPoleDynamics:
    def __init__(self, network, terminal_test):
        self.network = network
        self.terminal_test = terminal_test

    def step(self, state, action):
        next_state, _predicted_reward = self.network.predict(state, action)
        next_state = np.asarray(next_state, dtype=np.float32)
        # CartPole awards exactly +1 per real transition. Hardcoding it isolates
        # state-transition quality and prevents reward-model bias from compounding.
        return next_state, 1.0, bool(self.terminal_test(next_state))


class UniformPrior:
    uses_learned_prior = False

    def __init__(self, action_dim=2):
        self.action_dim = action_dim

    def probabilities(self, _state):
        return np.full(self.action_dim, 1.0 / self.action_dim, dtype=np.float32)


class NetworkPrior:
    uses_learned_prior = True

    def __init__(self, network):
        self.network = network

    def probabilities(self, state):
        return self.network.action_probs(state)


class RandomRolloutEvaluator:
    """Classical MCTS default policy, returning search-depth-normalized reward."""

    def __init__(self, transitions, search_depth, action_dim=2, seed=0):
        self.transitions = transitions
        self.search_depth = search_depth
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)

    def evaluate(self, state, remaining_depth):
        reward_sum = 0.0
        current = np.asarray(state, dtype=np.float32)
        for _ in range(remaining_depth):
            action = int(self.rng.integers(self.action_dim))
            current, reward, done = self.transitions.step(current, action)
            reward_sum += reward
            if done:
                break
        return reward_sum / self.search_depth


class NetworkValueEvaluator:
    """Uses ValueNetwork as a broad, search-depth-independent leaf value."""

    def __init__(self, network, value_horizon=500):
        self.network = network
        self.value_horizon = value_horizon

    def evaluate(self, state, remaining_depth):
        _ = remaining_depth
        return float(np.clip(self.network.value(state), 0.0, 1.0))


@dataclass
class SearchNode:
    state: np.ndarray
    parent: "SearchNode | None" = None
    action: int | None = None
    reward: float = 0.0
    done: bool = False
    prior: float = 1.0
    depth: int = 0
    children: dict[int, "SearchNode"] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0

    @property
    def mean_value(self):
        return self.value_sum / self.visits if self.visits else 0.0


class ModularMCTS:
    """PUCT MCTS whose transition, prior, and leaf value are replaceable."""

    def __init__(
        self,
        transitions: TransitionModel,
        priors: PriorModel,
        evaluator: LeafEvaluator,
        simulations=64,
        search_depth=30,
        bootstrap_after=0,
        exploration_c=1.4,
        discount=1.0,
        action_dim=2,
        seed=0,
    ):
        if simulations <= 0 or search_depth <= 0:
            raise ValueError("simulations and search_depth must be positive")
        if bootstrap_after < 0 or bootstrap_after > search_depth:
            raise ValueError("bootstrap_after must be between 0 and search_depth")
        self.transitions = transitions
        self.priors = priors
        self.evaluator = evaluator
        self.simulations = simulations
        self.search_depth = search_depth
        self.bootstrap_after = bootstrap_after
        self.exploration_c = exploration_c
        self.discount = discount
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)

    def search(self, root_state):
        root = SearchNode(np.asarray(root_state, dtype=np.float32))
        self._expand(root)
        for _ in range(self.simulations):
            node = root
            path = [root]
            if self.bootstrap_after == 0:
                while node.children and not node.done and node.depth < self.search_depth:
                    node = self._select(node)
                    path.append(node)
                if not node.done and node.depth < self.search_depth:
                    self._expand(node)
            else:
                while not node.done and node.depth < self.search_depth:
                    if node.children:
                        node = self._select(node)
                        path.append(node)
                        continue
                    self._expand(node)
                    if node.depth >= self.bootstrap_after:
                        break
                    if not node.children:
                        break
                    node = self._select(node)
                    path.append(node)
            remaining = max(0, self.search_depth - node.depth)
            leaf_value = 0.0 if node.done else self.evaluator.evaluate(node.state, remaining)
            self._backup(path, leaf_value)
        return root

    def _select(self, node):
        sqrt_parent = math.sqrt(max(1, node.visits))
        scores = {}
        for action, child in node.children.items():
            q = child.reward / self.search_depth + self.discount * child.mean_value
            if getattr(self.priors, "uses_learned_prior", False):
                # PUCT is the policy-enabled replacement for ordinary UCT.
                u = self.exploration_c * child.prior * sqrt_parent / (1 + child.visits)
            elif child.visits == 0:
                u = float("inf")
            else:
                u = self.exploration_c * math.sqrt(
                    math.log(max(2, node.visits)) / child.visits
                )
            scores[action] = q + u
        maximum = max(scores.values())
        tied = [action for action, score in scores.items() if np.isclose(score, maximum)]
        return node.children[int(self.rng.choice(tied))]

    def _expand(self, node):
        probabilities = np.asarray(self.priors.probabilities(node.state), dtype=np.float64)
        if probabilities.shape != (self.action_dim,) or not np.all(np.isfinite(probabilities)):
            raise ValueError("prior model returned invalid action probabilities")
        probabilities = np.maximum(probabilities, 0.0)
        probabilities = probabilities / probabilities.sum() if probabilities.sum() else np.full(
            self.action_dim, 1.0 / self.action_dim
        )
        for action in range(self.action_dim):
            next_state, reward, done = self.transitions.step(node.state, action)
            node.children[action] = SearchNode(
                state=np.asarray(next_state, dtype=np.float32),
                parent=node,
                action=action,
                reward=float(reward),
                done=bool(done),
                prior=float(probabilities[action]),
                depth=node.depth + 1,
            )

    def _backup(self, path, value):
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value
            value = node.reward / self.search_depth + self.discount * value


def select_mcts_action(root):
    visits = np.asarray([root.children[action].visits for action in sorted(root.children)])
    return int(np.flatnonzero(visits == visits.max())[0])
