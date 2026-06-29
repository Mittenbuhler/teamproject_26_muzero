import random

import numpy as np


class MCTSNode:
    """Model-based MCTS node over latent states, with no environment cloning."""

    def __init__(
        self,
        state,
        parent=None,
        action=None,
        reward=0.0,
        done=False,
        prior=1.0,
        action_dim=2,
        terminal_fn=None,
        depth=0,
    ):
        self.state = np.asarray(state, dtype=np.float32)
        self.parent = parent
        self.action = action
        self.reward = float(reward)
        self.done = bool(done)
        self.prior = float(prior)
        self.action_dim = action_dim
        self.terminal_fn = terminal_fn
        self.depth = depth

        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0

    @property
    def mean_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self):
        return bool(self.children)


class ModelBasedMCTS:
    """
    MCTS where:
      - selection uses visit statistics and policy priors
      - expansion uses the learned dynamics model
      - the value NN evaluates newly reached leaf states
    """

    def __init__(
        self,
        dynamics_model,
        policy_network,
        value_network,
        action_dim,
        terminal_fn,
        simulations=60,
        discount=0.997,
        exploration_c=1.4,
        reward_scale=1.0,
    ):
        self.dynamics_model = dynamics_model
        self.policy_network = policy_network
        self.value_network = value_network
        self.action_dim = action_dim
        self.terminal_fn = terminal_fn
        self.simulations = simulations
        self.discount = discount
        self.exploration_c = exploration_c
        self.reward_scale = reward_scale

    def search(self, root_state):
        root = MCTSNode(
            root_state,
            action_dim=self.action_dim,
            terminal_fn=self.terminal_fn,
        )
        self.expand(root)

        for _ in range(self.simulations):
            node = root
            path = [node]

            while node.is_expanded() and not node.done:
                node = self.select_child(node)
                path.append(node)

            if not node.done:
                self.expand(node)

            leaf_value = self.evaluate_leaf(node)
            self.backpropagate(path, leaf_value)

        return root

    def select_child(self, node):
        parent_visits = max(node.visit_count, 1)

        def score(child):
            q_score = child.reward + self.discount * child.mean_value
            prior_score = (
                self.exploration_c
                * child.prior
                * np.sqrt(parent_visits)
                / (1 + child.visit_count)
            )
            return q_score + prior_score

        max_score = max(score(child) for child in node.children.values())
        best_actions = [
            action for action, child in node.children.items()
            if score(child) == max_score
        ]
        return node.children[random.choice(best_actions)]

    def expand(self, node):
        if node.done:
            return

        priors = self.policy_network.action_probs(node.state)
        for action in range(self.action_dim):
            next_state, reward = self.dynamics_model.predict(node.state, action)
            reward *= self.reward_scale
            done = self.terminal_fn(next_state)
            child = MCTSNode(
                next_state,
                parent=node,
                action=action,
                reward=reward,
                done=done,
                prior=float(priors[action]),
                action_dim=self.action_dim,
                terminal_fn=self.terminal_fn,
                depth=node.depth + 1,
            )
            node.children[action] = child

    def evaluate_leaf(self, node):
        if node.done:
            return 0.0
        return float(self.value_network.value(node.state))

    def backpropagate(self, path, value):
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = node.reward + self.discount * value


def visit_count_policy(root, temperature=1.0):
    counts = np.asarray(
        [root.children[action].visit_count for action in range(root.action_dim)],
        dtype=np.float32,
    )
    if counts.sum() <= 0:
        return np.ones(root.action_dim, dtype=np.float32) / root.action_dim
    if temperature <= 0:
        policy = np.zeros(root.action_dim, dtype=np.float32)
        policy[int(np.argmax(counts))] = 1.0
        return policy
    counts = counts ** (1.0 / temperature)
    return counts / counts.sum()


def select_action(root, temperature=0.0):
    policy = visit_count_policy(root, temperature=temperature)
    if temperature <= 0:
        return int(np.argmax(policy)), policy
    return int(np.random.choice(len(policy), p=policy)), policy
