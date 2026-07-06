import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from legacy_muzero.buffers import LatentReplayBuffer
from legacy_muzero.diagnose_policy_behavior import policy_reactivity_test
from legacy_muzero.image_observation import FrameStack
from legacy_muzero.mcts import ModelBasedMCTS
from legacy_muzero.models import DynamicsModel, ImageRepresentationNetwork, PolicyNetwork, ValueNetwork
from legacy_muzero.train_policy_value import (
    bootstrapped_value_targets,
    discounted_return_scale,
    full_episode_value_targets,
    latent_terminal,
    load_latent_checkpoint,
    parse_simulations,
    performance_gated_temperature,
    save_latent_checkpoint,
    shaped_environment_reward,
    train_latent_step,
)


class LatentPipelineTest(unittest.TestCase):
    def test_policy_reactivity_gate_distinguishes_response_from_uniform_output(self):
        def trajectories(danger_probability):
            episodes = []
            for _ in range(20):
                trajectory = []
                for step in range(30):
                    state_coordinate = (step - 24.5) / 4.5
                    if danger_probability == 0.5:
                        right_probability = 0.5
                    else:
                        right_probability = np.clip(
                            0.5 + (danger_probability - 0.5) * state_coordinate,
                            0.01,
                            0.99,
                        )
                    policy = [1.0 - right_probability, right_probability]
                    trajectory.append(
                        {
                            "network_policy": policy,
                            "environment_state": np.asarray(
                                [state_coordinate, 0.0, state_coordinate / 10, 0.0]
                            ),
                            "terminated": step == 29,
                        }
                    )
                episodes.append(trajectory)
            return episodes

        reactive = policy_reactivity_test(
            trajectories(0.8),
            danger_window=10,
            permutation_count=100,
        )
        uniform = policy_reactivity_test(
            trajectories(0.5),
            danger_window=10,
            permutation_count=100,
        )
        self.assertTrue(reactive["passed"])
        self.assertFalse(uniform["passed"])

    def test_fixed_simulations_and_performance_gated_temperature(self):
        self.assertEqual(parse_simulations("50"), (50, 50))
        self.assertEqual(
            performance_gated_temperature(
                episode=1000,
                anneal_episodes=100,
                minimum=0.25,
                hold_temperature=0.5,
            ),
            0.5,
        )
        self.assertAlmostEqual(
            performance_gated_temperature(
                episode=40,
                anneal_episodes=100,
                minimum=0.25,
                hold_temperature=0.5,
                unlock_episode=40,
                unlock_temperature=0.8,
            ),
            0.8,
        )
        self.assertEqual(
            performance_gated_temperature(
                episode=200,
                anneal_episodes=100,
                minimum=0.25,
                hold_temperature=0.5,
                unlock_episode=40,
                unlock_temperature=0.8,
            ),
            0.25,
        )

    def test_frame_stack_zero_fills_then_counts_to_five(self):
        stack = FrameStack(stack_size=5, observation_shape=(1, 32, 32))
        first = np.ones((1, 32, 32), dtype=np.float32)
        observation = stack.reset(first)

        self.assertEqual(observation.shape, (5, 32, 32))
        self.assertEqual(stack.real_frame_count, 1)
        self.assertTrue(np.all(observation[:4] == 0.0))
        self.assertTrue(np.all(observation[4] == 1.0))

        for value in range(2, 6):
            observation = stack.append(
                np.full((1, 32, 32), value, dtype=np.float32)
            )
        self.assertEqual(stack.real_frame_count, 5)
        self.assertEqual(observation[:, 0, 0].tolist(), [1, 2, 3, 4, 5])

    def test_representation_maps_five_frames_to_32_latents(self):
        representation = ImageRepresentationNetwork(input_channels=5, latent_dim=32)
        observations = torch.rand(3, 5, 32, 32)
        latents = representation(observations)
        self.assertEqual(tuple(latents.shape), (3, 32))
        latents.square().mean().backward()
        self.assertTrue(
            all(parameter.grad is not None for parameter in representation.parameters())
        )

    def test_joint_step_trains_all_four_networks(self):
        torch.manual_seed(2)
        representation = ImageRepresentationNetwork(5, 32)
        dynamics = DynamicsModel(32, 2, hidden_dim=16)
        policy = PolicyNetwork(32, 2, hidden_dim=16)
        value = ValueNetwork(32, hidden_dim=16)
        buffer = LatentReplayBuffer(capacity=8)
        episode = []
        for index in range(4):
            observation = np.full((5, 32, 32), index / 4, dtype=np.float32)
            next_observation = np.full(
                (5, 32, 32), (index + 1) / 4, dtype=np.float32
            )
            episode.append(
                {
                    "observation": observation,
                    "action": index % 2,
                    "next_observation": next_observation,
                    "policy": [0.75, 0.25],
                    "value": 0.2,
                    "reward": 1.0,
                    "terminated": index == 3,
                }
            )
        buffer.add_episode(episode)

        networks = [representation, dynamics, policy, value]
        optimizer = torch.optim.Adam(
            [parameter for network in networks for parameter in network.parameters()],
            lr=1e-3,
        )
        before = [next(network.parameters()).detach().clone() for network in networks]
        losses = train_latent_step(
            representation,
            dynamics,
            policy,
            value,
            buffer,
            optimizer,
            batch_size=4,
            unroll_steps=3,
            action_dim=2,
            consistency_weight=0.25,
            device=torch.device("cpu"),
        )

        self.assertTrue(all(np.isfinite(loss) for loss in losses.values()))
        for old_parameter, network in zip(before, networks):
            self.assertFalse(torch.equal(old_parameter, next(network.parameters())))

    def test_replay_samples_contiguous_unrolls_and_masks_episode_end(self):
        buffer = LatentReplayBuffer(capacity=8)
        episode = []
        for index in range(2):
            episode.append(
                {
                    "observation": np.full((5, 32, 32), index, dtype=np.float32),
                    "action": index,
                    "next_observation": np.full(
                        (5, 32, 32), index + 1, dtype=np.float32
                    ),
                    "policy": [0.5, 0.5],
                    "value": float(index),
                    "reward": 1.0,
                    "terminated": index == 1,
                }
            )
        buffer.add_episode(episode)

        sample = buffer.sample(
            batch_size=2,
            unroll_steps=3,
            action_dim=2,
            device=torch.device("cpu"),
        )
        observations, actions, next_observations, _, _, _, terminals, masks = sample
        self.assertEqual(tuple(observations.shape), (2, 5, 32, 32))
        self.assertEqual(tuple(actions.shape), (2, 3, 2))
        self.assertEqual(tuple(next_observations.shape), (2, 3, 5, 32, 32))
        self.assertTrue(torch.all(masks[:, 0] == 1.0))
        self.assertTrue(torch.all(masks.sum(dim=1) <= 2.0))
        self.assertTrue(torch.any(terminals == 1.0))

    def test_latent_mcts_expands_with_dynamics_policy_and_value(self):
        dynamics = DynamicsModel(32, 2, hidden_dim=16)
        policy = PolicyNetwork(32, 2, hidden_dim=16)
        value = ValueNetwork(32, hidden_dim=16)
        mcts = ModelBasedMCTS(
            dynamics,
            policy,
            value,
            action_dim=2,
            terminal_fn=latent_terminal,
            simulations=2,
            reward_scale=1 / 500,
        )
        root = mcts.search(np.zeros(32, dtype=np.float32))
        self.assertEqual(set(root.children), {0, 1})
        self.assertEqual(root.children[0].state.shape, (32,))
        self.assertEqual(root.visit_count, 2)

    def test_dynamics_has_only_state_and_reward_outputs(self):
        dynamics = DynamicsModel(32, 2, hidden_dim=16)
        state = torch.zeros((3, 32))
        actions = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        )
        next_state, reward = dynamics(state, actions)
        self.assertEqual(next_state.shape, (3, 32))
        self.assertEqual(reward.shape, (3, 1))
        self.assertFalse(hasattr(dynamics, "terminal_head"))
        prediction = dynamics.predict(np.zeros(32, dtype=np.float32), 0)
        self.assertEqual(len(prediction), 2)

    def test_bootstrapped_target_uses_real_rewards_and_future_search_value(self):
        trajectory = [
            {"reward": 1.0, "search_value": 0.1},
            {"reward": 1.0, "search_value": 0.2},
            {"reward": 1.0, "search_value": 0.3},
        ]
        targets = bootstrapped_value_targets(
            trajectory,
            discount=0.9,
            bootstrap_steps=2,
            max_steps=10,
        )
        scale = discounted_return_scale(0.9, 10)
        self.assertAlmostEqual(targets[0], 1.0 / scale + 0.9 / scale + 0.81 * 0.3)

    def test_terminal_penalty_makes_delayed_failure_more_valuable(self):
        discount = 0.997
        terminal_penalty = -25.0
        self.assertEqual(
            shaped_environment_reward(1.0, terminated=True, terminal_penalty=-25.0),
            terminal_penalty,
        )
        self.assertEqual(
            shaped_environment_reward(1.0, terminated=False, terminal_penalty=-25.0),
            1.0,
        )

        immediate_failure = [{"reward": terminal_penalty, "search_value": 0.0}]
        delayed_failure = [
            {"reward": 1.0, "search_value": 0.0} for _ in range(20)
        ] + [{"reward": terminal_penalty, "search_value": 0.0}]
        immediate_value = full_episode_value_targets(
            immediate_failure,
            discount=discount,
            max_steps=500,
        )[0]
        delayed_value = full_episode_value_targets(
            delayed_failure,
            discount=discount,
            max_steps=500,
        )[0]
        self.assertGreater(delayed_value, immediate_value)

    def test_full_episode_target_uses_every_reward_without_bootstrap(self):
        trajectory = [
            {"reward": 1.0, "search_value": -0.9},
            {"reward": 1.0, "search_value": -0.9},
            {"reward": -25.0, "search_value": 0.9},
        ]
        targets = full_episode_value_targets(
            trajectory,
            discount=0.997,
            max_steps=500,
        )
        scale = discounted_return_scale(0.997, 500)
        expected = (1.0 + 0.997 - 25.0 * 0.997**2) / scale
        self.assertAlmostEqual(targets[0], expected)

    def test_unified_checkpoint_round_trip(self):
        representation = ImageRepresentationNetwork(5, 32)
        dynamics = DynamicsModel(32, 2, hidden_dim=16)
        policy = PolicyNetwork(32, 2, hidden_dim=16)
        value = ValueNetwork(32, hidden_dim=16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latent.pt"
            save_latent_checkpoint(
                path,
                representation,
                dynamics,
                policy,
                value,
                {"rewards": [1.0]},
                value_discount=0.997,
                max_steps=500,
                terminal_penalty=-25.0,
                value_target_mode="full-episode",
                unroll_steps=5,
            )
            loaded = load_latent_checkpoint(path, device=torch.device("cpu"))
            self.assertEqual(loaded[0].input_channels, 5)
            self.assertEqual(loaded[1].state_dim, 32)
            self.assertEqual(loaded[4]["history"]["rewards"], [1.0])
            self.assertEqual(loaded[4]["value_discount"], 0.997)
            self.assertEqual(loaded[4]["terminal_penalty"], -25.0)
            self.assertEqual(loaded[4]["value_target_mode"], "full-episode")
            self.assertEqual(loaded[4]["unroll_steps"], 5)
            self.assertEqual(loaded[4]["temperature_hold"], 0.5)
            self.assertEqual(loaded[4]["training_version"], 10)

    def test_version_9_checkpoint_discards_removed_terminal_head(self):
        representation = ImageRepresentationNetwork(5, 32)
        dynamics = DynamicsModel(32, 2, hidden_dim=16)
        policy = PolicyNetwork(32, 2, hidden_dim=16)
        value = ValueNetwork(32, hidden_dim=16)
        old_dynamics = dict(dynamics.state_dict())
        old_dynamics["terminal_head.weight"] = torch.zeros((1, 16))
        old_dynamics["terminal_head.bias"] = torch.zeros(1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "version9.pt"
            torch.save(
                {
                    "training_version": 9,
                    "representation_state_dict": representation.state_dict(),
                    "dynamics_state_dict": old_dynamics,
                    "policy_state_dict": policy.state_dict(),
                    "value_state_dict": value.state_dict(),
                    "image_size": 32,
                    "stack_size": 5,
                    "latent_dim": 32,
                    "action_dim": 2,
                    "hidden_dim": 16,
                    "history": {"rewards": []},
                },
                path,
            )
            loaded = load_latent_checkpoint(path, device=torch.device("cpu"))
            self.assertEqual(loaded[4]["training_version"], 9)
            self.assertFalse(hasattr(loaded[1], "terminal_head"))


if __name__ == "__main__":
    unittest.main()
