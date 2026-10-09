import sys
import tempfile
import types
import unittest
import warnings
from pathlib import Path
from unittest import mock

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

from minatar_muzero.buffers import LatentReplayBuffer
from minatar_muzero.diagnose import _screenshot_history_panel, record_episode_gif
from minatar_muzero.environment import (
    MinAtarAdapter,
    ObservationHistory,
    make_minatar_env,
    minatar_env_id,
    _render_to_rgb,
)
from minatar_muzero.mcts import MCTSNode, ModelBasedMCTS, visit_count_policy
from minatar_muzero.models import DynamicsModel, PredictionNetwork
from minatar_muzero.train import (
    CHECKPOINT_VERSION,
    CHECKPOINT_FORMAT,
    build_argument_parser,
    build_networks,
    companion_checkpoint_path,
    load_latent_checkpoint,
    save_latent_checkpoint,
    train_latent_step,
    train_muzero,
)


GAME_SPECS = {
    "asterix": (4, 5),
    "breakout": (4, 3),
    "freeway": (7, 3),
    "seaquest": (10, 6),
    "space_invaders": (6, 4),
}


def replay_episode(channels=3, action_dim=3, length=4, policy_valid=True):
    transitions = []
    for index in range(length):
        policy = np.full(action_dim, 0.1 / max(1, action_dim - 1), np.float32)
        policy[index % action_dim] = 0.9
        transitions.append(
            {
                "observation": np.full(
                    (channels, 10, 10), index / 10, np.float32
                ),
                "action": index % action_dim,
                "reward": float(index + 1),
                "policy": policy,
                "policy_valid": policy_valid,
                "value": float(10 + index),
                "terminated": index == length - 1,
                "truncated": False,
            }
        )
    return transitions


class FakeNativeMinAtar:
    """Small HWC boolean environment shaped like Gymnasium MinAtar."""

    def __init__(self, channels=4, action_dim=3):
        self.channels = channels
        self.action_space = gym.spaces.Discrete(action_dim)
        self.observation_space = gym.spaces.Box(
            0, 1, shape=(10, 10, channels), dtype=np.uint8
        )
        self.steps = 0
        self.actions = []
        self.closed = False

    def _observation(self):
        observation = np.zeros((10, 10, self.channels), dtype=bool)
        for channel in range(self.channels):
            observation[(channel + self.steps) % 10, channel % 10, channel] = True
        return observation

    def reset(self, seed=None, options=None):
        del seed, options
        self.steps = 0
        return self._observation(), {"native": True}

    def step(self, action):
        self.actions.append(int(action))
        self.steps += 1
        return self._observation(), 1.0, self.steps >= 2, False, {}

    def render(self):
        # Pixels deliberately differ from the native feature grids.
        screenshot = np.empty((10, 10, 3), np.float32)
        screenshot[:] = (self.steps / 4, 0.5, 1.0)
        return screenshot

    def close(self):
        self.closed = True


class FakeMinAtarAdapter(MinAtarAdapter):
    """Real screenshot adapter around a two-step game for actor/GIF tests."""

    def __init__(self, game="breakout", image_size=10, **kwargs):
        channels, action_dim = GAME_SPECS[game]
        self.native = FakeNativeMinAtar(channels, action_dim)
        with mock.patch("minatar_muzero.environment.gym.make", return_value=self.native):
            super().__init__(game, image_size=image_size, **kwargs)

    @property
    def actions(self):
        return self.native.actions

    @property
    def closed(self):
        return self.native.closed


class ToyDynamics:
    def __init__(self, rewards):
        self.rewards = rewards
        self.calls = 0

    def predict(self, state, action):
        self.calls += 1
        state = np.asarray(state, np.float32)
        return state + action + 1, float(self.rewards[action])


class ToyPrediction:
    def __init__(self, action_dim, values=None, priors=None):
        self.values = values or {}
        self.priors = np.asarray(
            priors if priors is not None else np.ones(action_dim) / action_dim,
            np.float32,
        )

    def predict(self, state):
        key = float(np.asarray(state).mean())
        return self.priors, float(self.values.get(key, 0.0))


class FakeGifRepresentation:
    def __init__(self):
        self.observations = []

    def encode(self, observation):
        self.observations.append(np.asarray(observation).copy())
        return np.asarray([[[np.asarray(observation).mean()]]], np.float32)


class FakeGifPrediction:
    def __init__(self, action_dim=3):
        self.action_dim = action_dim

    def predict(self, _latent):
        policy = np.full(self.action_dim, 0.1 / (self.action_dim - 1), np.float32)
        policy[0] = 0.9
        return policy, 0.5


class FakeGifMCTS:
    def __init__(self, action_dim=3):
        self.action_dim = action_dim
        self.noise = []

    def search(self, _latent, add_exploration_noise=False):
        self.noise.append(add_exploration_noise)
        root = MCTSNode()
        root.children = {
            action: MCTSNode(1 / self.action_dim)
            for action in range(self.action_dim)
        }
        for action, child in root.children.items():
            child.visit_count = action + 1
        return root


class ScreenshotMuZeroPipelineTest(unittest.TestCase):
    def make_models(self, input_shape=(3, 10, 10), action_dim=3):
        return build_networks(
            input_shape,
            action_dim,
            latent_channels=8,
            device=torch.device("cpu"),
        )

    def test_minatar_ids_default_to_minimal_action_v1(self):
        self.assertEqual(minatar_env_id("breakout"), "MinAtar/Breakout-v1")
        self.assertEqual(
            minatar_env_id("space_invaders"), "MinAtar/SpaceInvaders-v1"
        )
        self.assertEqual(
            minatar_env_id("MinAtar/Seaquest-v0"), "MinAtar/Seaquest-v0"
        )

    def test_all_installed_minatar_v1_games_construct_and_step(self):
        for game, (_channels, action_dim) in GAME_SPECS.items():
            with self.subTest(game=game):
                environment = make_minatar_env(game, render_cell_size=2)
                try:
                    observation, _ = environment.reset(seed=9)
                    next_observation, reward, terminated, truncated, _ = (
                        environment.step(0)
                    )
                    self.assertEqual(observation.shape, (3, 32, 32))
                    self.assertEqual(next_observation.shape, observation.shape)
                    self.assertEqual(environment.action_dim, action_dim)
                    self.assertTrue(np.isfinite(reward))
                    self.assertIsInstance(terminated, bool)
                    self.assertIsInstance(truncated, bool)
                    self.assertEqual(environment.render().shape, (20, 20, 3))
                    expected_pixels = _render_to_rgb(environment._env.render())
                    expected_pixels = np.asarray(Image.fromarray(expected_pixels).resize(
                        (32, 32), Image.Resampling.NEAREST
                    ))
                    np.testing.assert_array_equal(
                        next_observation, np.moveaxis(expected_pixels, -1, 0) / np.float32(255)
                    )
                finally:
                    environment.close()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            full_action_environment = make_minatar_env("MinAtar/Breakout-v0")
        try:
            self.assertEqual(full_action_environment.action_dim, 6)
        finally:
            full_action_environment.close()

    def test_adapter_registers_minatar_and_uses_rendered_rgb_pixels(self):
        native = FakeNativeMinAtar(channels=7, action_dim=3)
        register_envs = mock.Mock()
        minatar_module = types.ModuleType("minatar")
        minatar_gym_module = types.ModuleType("minatar.gym")
        minatar_gym_module.register_envs = register_envs
        minatar_module.gym = minatar_gym_module
        with mock.patch.dict(
            sys.modules,
            {"minatar": minatar_module, "minatar.gym": minatar_gym_module},
        ), mock.patch.dict(
            gym.envs.registry,
            {},
            clear=True,
        ), mock.patch("minatar_muzero.environment.gym.make", return_value=native) as make:
            environment = MinAtarAdapter("freeway", render_cell_size=2)
            observation, info = environment.reset(seed=4)
            next_observation, reward, terminated, truncated, _ = environment.step(2)
            rendered = environment.render()
            environment.close()

        register_envs.assert_called_once_with()
        make.assert_called_once_with(
            "MinAtar/Freeway-v1",
            render_mode="rgb_array",
            disable_env_checker=True,
            sticky_action_prob=0.1,
            difficulty_ramping=True,
        )
        self.assertEqual(observation.shape, (3, 32, 32))
        self.assertEqual(next_observation.shape, observation.shape)
        self.assertEqual(observation.dtype, np.float32)
        self.assertTrue(observation.flags.c_contiguous)
        self.assertTrue(((observation >= 0) & (observation <= 1)).all())
        self.assertEqual(environment.action_dim, 3)
        self.assertEqual(info, {"native": True})
        self.assertEqual((reward, terminated, truncated), (1.0, False, False))
        self.assertEqual(rendered.shape, (20, 20, 3))
        self.assertEqual(rendered.dtype, np.uint8)
        self.assertTrue(native.closed)
        np.testing.assert_array_equal(observation[:, 0, 0], [0, np.float32(128/255), 1])
        np.testing.assert_array_equal(
            next_observation[:, 0, 0], [np.float32(64/255), np.float32(128/255), 1]
        )

    def test_native_state_and_observation_space_cannot_leak_into_input(self):
        native = FakeNativeMinAtar()
        # Any inspection/conversion of the native observation must fail.
        hidden_state = mock.Mock()
        hidden_state.__array__ = mock.Mock(side_effect=AssertionError("native state read"))
        native.observation_space = None
        with mock.patch.object(native, "_observation", return_value=hidden_state), \
             mock.patch("minatar_muzero.environment.gym.make", return_value=native):
            env = MinAtarAdapter(image_size=16, render_cell_size=1)
            first, _ = env.reset()
            with mock.patch.object(native, "_observation", return_value=np.full((7, 2), np.nan)):
                second, _ = env.reset()
            np.testing.assert_array_equal(first, second)
            with mock.patch.object(native, "render", return_value=np.zeros((5, 8, 3), np.uint8)):
                changed, _ = env.reset()
            self.assertFalse(np.array_equal(first, changed))
            np.testing.assert_array_equal(changed, 0)
            env.close()

    def test_render_is_cached_and_missing_pixels_never_fall_back_to_state(self):
        native = FakeNativeMinAtar()
        with mock.patch("minatar_muzero.environment.gym.make", return_value=native):
            env = MinAtarAdapter(render_cell_size=1)
        with mock.patch.object(native, "render", wraps=native.render) as render:
            env.reset()
            before = env.render()
            env.render()
            self.assertEqual(render.call_count, 1)
            env.step(0)
            self.assertEqual(render.call_count, 2)
            self.assertFalse(np.array_equal(before, env.render()))
        with mock.patch.object(native, "render", return_value=None):
            with self.assertRaisesRegex(ValueError, "no screenshot"):
                env.step(0)
            with self.assertRaises(RuntimeError):
                env.render()
        env.close()

    def test_screenshot_conversion_accepts_rgb_rgba_and_rejects_invalid_pixels(self):
        expected = np.array([[[0, 128, 255]]], np.uint8)
        for frame in (expected, expected.astype(float),
                      np.array([[[0., 0.5, 1.]]]),
                      np.array([[[0, 128, 255, 255]]], np.uint8)):
            np.testing.assert_array_equal(_render_to_rgb(frame), expected)
        for frame in (None, np.zeros((10, 10)), np.zeros((0, 10, 3)),
                      np.full((10, 10, 3), np.nan), np.full((10, 10, 3), 256)):
            with self.subTest(frame_type=type(frame)), self.assertRaises(ValueError):
                _render_to_rgb(frame)

    def test_observation_history_concatenates_time_without_mixing_channels(self):
        history = ObservationHistory(3, (4, 10, 10))
        first = np.stack(
            [np.full((10, 10), channel + 1, np.float32) for channel in range(4)]
        )
        second = first + 10
        root = history.reset(first)
        self.assertEqual(root.shape, (12, 10, 10))
        np.testing.assert_array_equal(root[:8], 0)
        np.testing.assert_array_equal(root[8:], first)
        recurrent_root = history.append(second)
        np.testing.assert_array_equal(recurrent_root[:4], 0)
        np.testing.assert_array_equal(recurrent_root[4:8], first)
        np.testing.assert_array_equal(recurrent_root[8:], second)
        self.assertEqual(history.real_frame_count, 2)
        self.assertEqual(history.output_shape, (12, 10, 10))

    def test_default_history_is_four_screenshots_and_reset_clears_previous_episode(self):
        history = ObservationHistory()
        observation = np.random.default_rng(2).random((3, 32, 32), dtype=np.float32)
        for _ in range(5):
            history.append(observation)
        reset = history.reset(observation)
        np.testing.assert_array_equal(reset[:9], 0)
        np.testing.assert_array_equal(reset[9:], observation)
        self.assertEqual(history.output_shape, (12, 32, 32))

    def test_networks_derive_shape_and_action_count_for_multiple_games(self):
        for _game, (_native_channels, action_dim) in GAME_SPECS.items():
            channels = 12
            representation, dynamics, prediction = self.make_models(
                (channels, 32, 32), action_dim
            )
            observation = torch.rand(2, channels, 32, 32)
            root = representation(observation)
            logits, value = prediction(root)
            recurrent, reward = dynamics(
                root, torch.tensor([0, action_dim - 1])
            )
            recurrent_logits, recurrent_value = prediction(recurrent)
            self.assertEqual(root.shape[1:], representation.latent_shape)
            self.assertEqual(logits.shape, (2, action_dim))
            self.assertEqual(value.shape, (2, 1))
            self.assertEqual(recurrent.shape, root.shape)
            self.assertEqual(reward.shape, (2, 1))
            self.assertEqual(recurrent_logits.shape, (2, action_dim))
            self.assertEqual(recurrent_value.shape, (2, 1))
            self.assertTrue(torch.isfinite(root).all())
            self.assertTrue(((root >= 0) & (root <= 1)).all())

    def test_history_length_changes_only_configured_input_channels(self):
        representation, _, _ = self.make_models((6, 10, 10), 3)
        self.assertEqual(representation.input_channels, 6)
        self.assertEqual(representation.observation_shape, (10, 10))
        self.assertEqual(representation(torch.rand(1, 6, 10, 10)).shape[1:],
                         representation.latent_shape)

    def test_dynamics_responds_to_different_actions(self):
        _, dynamics, _ = self.make_models((3, 10, 10), 5)
        state = torch.rand(2, 8, 3, 3)
        left, _ = dynamics(state, torch.zeros(2, dtype=torch.long))
        right, _ = dynamics(state, torch.full((2,), 4, dtype=torch.long))
        self.assertEqual(left.shape, state.shape)
        self.assertFalse(torch.allclose(left, right))

    def test_replay_alignment_with_six_actions_and_screenshot_history(self):
        replay = LatentReplayBuffer()
        replay.add_episode(replay_episode(channels=12, action_dim=6))
        observations, actions, policies, values, rewards, pm, vm, rm = replay.sample(
            1, 3, 6, starts=[0]
        )
        self.assertEqual(observations.shape, (1, 12, 10, 10))
        self.assertEqual(actions.tolist(), [[0, 1, 2]])
        self.assertEqual(policies.shape, (1, 4, 6))
        self.assertEqual(values[:, :, 0].tolist(), [[10, 11, 12, 13]])
        self.assertEqual(rewards[:, :, 0].tolist(), [[1, 2, 3]])
        self.assertTrue(torch.all(pm == 1)); self.assertTrue(torch.all(vm == 1))
        self.assertTrue(torch.all(rm == 1))

    def test_replay_masks_warmup_policy_but_keeps_model_targets(self):
        replay = LatentReplayBuffer()
        replay.add_episode(
            replay_episode(channels=3, action_dim=3, length=2, policy_valid=False)
        )
        _, _, _, values, rewards, pm, vm, rm = replay.sample(1, 1, 3, starts=[0])
        self.assertEqual(pm[:, :, 0].tolist(), [[0, 0]])
        self.assertEqual(vm[:, :, 0].tolist(), [[1, 1]])
        self.assertEqual(rm[:, :, 0].tolist(), [[1]])
        self.assertEqual(values[:, :, 0].tolist(), [[10, 11]])
        self.assertEqual(rewards[:, :, 0].tolist(), [[1]])

    def test_replay_terminal_padding_never_crosses_episode(self):
        replay = LatentReplayBuffer()
        replay.add_episode(replay_episode(length=1))
        other = replay_episode(length=1); other[0]["reward"] = 99
        replay.add_episode(other)
        for _ in range(10):
            _, actions, _, values, rewards, pm, vm, _ = replay.sample(
                1, 3, 3, starts=[0]
            )
            self.assertIn(rewards[0, 0, 0].item(), (1, 99))
            self.assertEqual(rewards[0, 1:, 0].tolist(), [0, 0])
            self.assertEqual(actions[0, 1:].tolist(), [0, 0])
            self.assertEqual(values[0, 1:, 0].tolist(), [0, 0, 0])
            self.assertEqual(pm[0, 1:, 0].tolist(), [0, 0, 0])
            self.assertEqual(vm[0, :, 0].tolist(), [1, 1, 1, 1])

    def test_joint_training_step_reaches_all_three_networks(self):
        torch.manual_seed(5)
        representation, dynamics, prediction = self.make_models((6, 10, 10), 3)
        replay = LatentReplayBuffer()
        replay.add_episode(replay_episode(channels=6, action_dim=3))
        optimizer = torch.optim.Adam(
            [*representation.parameters(), *dynamics.parameters(), *prediction.parameters()],
            lr=1e-3,
        )
        result = train_latent_step(
            representation, dynamics, prediction, replay, optimizer,
            4, 3, 3, torch.device("cpu")
        )
        self.assertEqual(result["prediction_calls"], 4)
        for network in (representation, dynamics, prediction):
            gradients = [p.grad for p in network.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)

    def test_fixed_screenshot_batch_can_reduce_all_losses(self):
        torch.manual_seed(0)
        representation, dynamics, prediction = self.make_models((3, 10, 10), 3)
        replay = LatentReplayBuffer()
        replay.add_episode(replay_episode(channels=3, action_dim=3, length=2))
        optimizer = torch.optim.Adam(
            [*representation.parameters(), *dynamics.parameters(), *prediction.parameters()],
            lr=5e-3,
        )
        first = train_latent_step(
            representation, dynamics, prediction, replay, optimizer,
            2, 1, 3, torch.device("cpu")
        )
        last = first
        for _ in range(50):
            last = train_latent_step(
                representation, dynamics, prediction, replay, optimizer,
                2, 1, 3, torch.device("cpu")
            )
        for loss in ("policy_loss", "value_loss", "reward_loss"):
            self.assertLess(last[loss], first[loss])

    def test_mcts_prefers_higher_reward_with_more_than_two_actions(self):
        mcts = ModelBasedMCTS(
            ToyDynamics([0, 0, 2]), ToyPrediction(3), 3,
            simulations=40, discount=0.9
        )
        root = mcts.search(np.zeros((1, 1, 1), np.float32))
        self.assertGreater(root.children[2].visit_count, root.children[0].visit_count)

    def test_mcts_prefers_higher_leaf_value_and_expands_one_edge_per_simulation(self):
        dynamics = ToyDynamics([0, 0, 0])
        prediction = ToyPrediction(3, values={1.0: 0, 2.0: 0, 3.0: 4})
        mcts = ModelBasedMCTS(dynamics, prediction, 3, simulations=6, discount=0.9)
        root = mcts.search(np.zeros((1, 1, 1), np.float32))
        self.assertGreater(root.children[2].visit_count, root.children[0].visit_count)
        self.assertEqual(dynamics.calls, 6)
        self.assertEqual(mcts.expansions_last_search, 6)

    def test_screenshot_checkpoint_roundtrip_preserves_preprocessing_and_actions(self):
        representation, dynamics, prediction = self.make_models((12, 32, 32), 6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seaquest.pt"
            save_latent_checkpoint(
                path, representation, dynamics, prediction,
                {
                    "game": "seaquest",
                    "env_id": "MinAtar/Seaquest-v1",
                    "base_observation_shape": (3, 32, 32),
                    "history_length": 4,
                },
            )
            loaded_rep, loaded_dyn, loaded_pred, metadata = load_latent_checkpoint(path)
            self.assertEqual(metadata["checkpoint_version"], CHECKPOINT_VERSION)
            self.assertEqual(metadata["game"], "seaquest")
            self.assertEqual(metadata["checkpoint_format"], CHECKPOINT_FORMAT)
            self.assertEqual(metadata["observation_preprocessing"]["source"], "render_rgb_array")
            self.assertEqual(tuple(metadata["base_observation_shape"]), (3, 32, 32))
            self.assertEqual(metadata["history_length"], 4)
            self.assertEqual(loaded_rep.input_channels, 12)
            self.assertEqual(loaded_dyn.action_dim, 6)
            self.assertEqual(loaded_pred.action_dim, 6)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_legacy_checkpoint_is_rejected_informatively(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save({"checkpoint_version": 14}, path)
            with self.assertRaises(ValueError) as raised:
                load_latent_checkpoint(path)
            message = str(raised.exception).lower()
            for fragment in ("legacy", "version 14", "native", "fresh"):
                self.assertIn(fragment, message)

    def test_parser_defaults_to_breakout_and_four_rgb_screenshots(self):
        arguments = build_argument_parser().parse_args([])
        self.assertEqual(arguments.game, "breakout")
        self.assertEqual(arguments.history_length, 4)
        self.assertEqual(arguments.image_size, 32)
        self.assertNotIn("stack_size", vars(arguments))

    def test_screenshot_panel_preserves_rgb_and_chronological_order(self):
        levels = (16, 48, 96, 160, 224, 240)
        observation = np.stack(
            [np.full((10, 10), level / 255, np.float32) for level in levels]
        )
        panel = _screenshot_history_panel(
            observation,
            output_height=300,
            history_length=2,
            real_frame_count=2,
        )
        pixels = np.asarray(panel)
        centres = []
        for color in (levels[:3], levels[3:]):
            locations = np.argwhere(np.all(pixels == color, axis=-1))
            self.assertGreater(len(locations), 20)
            centres.append(float(np.median(locations[:, 1])))
        self.assertEqual(centres, sorted(centres))

    def test_composite_gif_contains_live_screenshots_in_raw_and_mcts_modes(self):
        prediction = FakeGifPrediction(action_dim=3)
        with tempfile.TemporaryDirectory() as directory:
            raw_env = FakeMinAtarAdapter()
            raw_representation = FakeGifRepresentation()
            raw_path = Path(directory) / "raw.gif"
            with mock.patch("minatar_muzero.diagnose.make_minatar_env", return_value=raw_env):
                raw = record_episode_gif(
                    raw_path, "breakout", raw_representation, prediction, None,
                    base_observation_shape=(3, 10, 10), history_length=2,
                    mode="raw", seed=3, max_steps=4, fps=10, output_width=120,
                )
            self.assertEqual(raw_env.actions, [0, 0])
            self.assertEqual(len(raw_representation.observations), 2)
            first = np.broadcast_to(
                np.array([0, 128, 255], np.float32)[:, None, None] / 255, (3, 10, 10)
            )
            second = np.broadcast_to(
                np.array([64, 128, 255], np.float32)[:, None, None] / 255, (3, 10, 10)
            )
            np.testing.assert_allclose(
                raw_representation.observations[0], np.concatenate([np.zeros_like(first), first])
            )
            np.testing.assert_allclose(
                raw_representation.observations[1], np.concatenate([first, second])
            )
            self.assertGreater(raw["gif_width"], raw["gameplay_width"])
            self.assertEqual(raw["screenshot_channels"], 3)
            self.assertEqual(raw["history_length"], 2)
            with Image.open(raw_path) as image:
                self.assertEqual(image.format, "GIF")
                self.assertEqual(image.n_frames, 3)
                self.assertGreater(image.width, 120)
                recorded = []
                for index in range(image.n_frames):
                    image.seek(index)
                    recorded.append(np.asarray(image.convert("RGB"))[:, 120:])
                self.assertFalse(np.array_equal(recorded[0], recorded[1]))
                self.assertFalse(np.array_equal(recorded[1], recorded[-1]))
                for index, color in enumerate(((0, 128, 255), (64, 128, 255), (128, 128, 255))):
                    self.assertGreater(int(np.all(recorded[index] == color, axis=-1).sum()), 20)

            mcts_env = FakeMinAtarAdapter()
            mcts = FakeGifMCTS(action_dim=3)
            mcts_path = Path(directory) / "mcts.gif"
            with mock.patch("minatar_muzero.diagnose.make_minatar_env", return_value=mcts_env):
                result = record_episode_gif(
                    mcts_path, "breakout", FakeGifRepresentation(), prediction, mcts,
                    base_observation_shape=(3, 10, 10), history_length=1,
                    mode="mcts", seed=3, max_steps=4, fps=10, output_width=120,
                )
            self.assertEqual(mcts_env.actions, [2, 2])
            self.assertEqual(mcts.noise, [False, False])
            self.assertGreater(result["gif_width"], result["gameplay_width"])

    def test_training_replay_evaluation_and_resume_use_screenshots_for_two_games(self):
        for game, (_channels, action_dim) in {
            "breakout": GAME_SPECS["breakout"],
            "seaquest": GAME_SPECS["seaquest"],
        }.items():
            with self.subTest(game=game), tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / game / "best.pt"
                environments = []
                def factory(game, **kwargs):
                    env = FakeMinAtarAdapter(game, **kwargs)
                    environments.append(env)
                    return env
                arguments = dict(
                        game=game, episodes=1, max_steps=2, history_length=2, image_size=16,
                        latent_channels=4, simulations=2, batch_size=2,
                        buffer_capacity=20, warmup_episodes=1,
                        updates_per_episode=1, updates_per_transition=0,
                        unroll_steps=1, evaluation_interval=1,
                        evaluation_episodes=1, checkpoint_path=checkpoint,
                        reward_plot_path=Path(directory) / game / "rewards.png",
                        seed=7, device=torch.device("cpu"),
                )
                with mock.patch("minatar_muzero.train.make_minatar_env", side_effect=factory), \
                     mock.patch("minatar_muzero.train.save_reward_plot", return_value=Path(directory) / "reward.png"):
                    train_muzero(**arguments)
                    train_muzero(**arguments, reuse_checkpoint=True)
                    for changed in ({"image_size": 32}, {"history_length": 1}, {"game": "asterix"}):
                        with self.assertRaisesRegex(ValueError, "does not match"):
                            train_muzero(**{**arguments, **changed}, reuse_checkpoint=True)
                data = torch.load(
                    companion_checkpoint_path(checkpoint, "latest"),
                    weights_only=False,
                )
                self.assertEqual(tuple(data["base_observation_shape"]),
                                 (3, 16, 16))
                self.assertEqual(data["input_channels"], 6)
                self.assertEqual(data["action_dim"], action_dim)
                self.assertEqual(data["game"], game)
                self.assertEqual(data["completed_episodes"], 2)
                self.assertTrue(all(env.closed for env in environments))
                self.assertTrue(all(env.image_size == 16 for env in environments[:4]))
                replay = LatentReplayBuffer()
                replay.load_state_dict(data["replay_state"])
                root, *_ = replay.sample(1, 1, action_dim, starts=[0])
                self.assertEqual(tuple(root.shape), (1, 6, 16, 16))
                np.testing.assert_array_equal(root[0, :3], 0)
                np.testing.assert_array_equal(root[0, 3:, 0, 0],
                                              [0, np.float32(128/255), 1])

    def test_native_checkpoint_and_inconsistent_screenshot_metadata_are_rejected(self):
        networks = self.make_models((6, 16, 16))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_latent_checkpoint(path, *networks)
            original = torch.load(path, weights_only=False)
            changes = (
                {"checkpoint_format": "native_minatar_vanilla_muzero", "checkpoint_version": 1},
                {"observation_preprocessing": {}},
                {"observation_preprocessing": {**original["observation_preprocessing"], "color_mode": "L"}},
                {"history_length": 4},
                {"base_observation_shape": (6, 16, 16)},
            )
            for changed in changes:
                with self.subTest(changed=changed):
                    torch.save({**original, **changed}, path)
                    with self.assertRaises(ValueError):
                        load_latent_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
