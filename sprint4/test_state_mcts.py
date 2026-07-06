import tempfile
import unittest
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from modular_state_cartpole import (
    build_mcts,
    component_configurations,
    load_report_history,
    parse_models,
    write_loss_graph,
    write_report_history,
)
from state_mcts import (
    ExactCartPoleDynamics,
    LearnedCartPoleDynamics,
    ModularMCTS,
    NetworkValueEvaluator,
    RandomRolloutEvaluator,
    UniformPrior,
    select_mcts_action,
)
from state_training import (
    collect_state_dataset,
    curriculum_horizon,
    dynamics_curriculum,
    load_component,
    save_component,
    train_dynamics,
    train_policy,
    train_value,
)


class ConstantPolicy:
    def __init__(self, probabilities=(0.1, 0.9)):
        self.output = np.asarray(probabilities, dtype=np.float32)

    def action_probs(self, _state):
        return self.output


class ConstantValue:
    def __init__(self, value=0.75):
        self.output = value
        self.calls = 0

    def value(self, _state):
        self.calls += 1
        return self.output


class CountingDynamics:
    def __init__(self):
        self.calls = 0

    def predict(self, state, action):
        self.calls += 1
        next_state = np.asarray(state, dtype=np.float32).copy()
        next_state[0] += 0.01 if action else -0.01
        return next_state, 1.0


class ActionSensitiveTransitions:
    def step(self, state, action):
        next_state = np.asarray(state, dtype=np.float32).copy()
        next_state[0] += 1.0
        return next_state, 1.0, action == 0


class StateMCTSTest(unittest.TestCase):
    def test_exact_dynamics_matches_gym_transition(self):
        env = gym.make("CartPole-v1")
        exact = ExactCartPoleDynamics.from_env(env)
        rng = np.random.default_rng(4)
        try:
            for _ in range(30):
                state = np.asarray(
                    [
                        rng.uniform(-1.5, 1.5),
                        rng.uniform(-1.0, 1.0),
                        rng.uniform(-0.15, 0.15),
                        rng.uniform(-1.0, 1.0),
                    ],
                    dtype=np.float32,
                )
                action = int(rng.integers(2))
                env.reset()
                env.unwrapped.state = state.copy()
                gym_state, gym_reward, gym_done, _, _ = env.step(action)
                state_prediction, reward_prediction, done_prediction = exact.step(
                    state, action
                )
                np.testing.assert_allclose(state_prediction, gym_state, atol=1e-6)
                self.assertEqual(reward_prediction, gym_reward)
                self.assertEqual(done_prediction, gym_done)
        finally:
            env.close()

    def test_classical_mcts_prefers_action_with_longer_survival(self):
        transitions = ActionSensitiveTransitions()
        mcts = ModularMCTS(
            transitions,
            UniformPrior(),
            RandomRolloutEvaluator(transitions, search_depth=6, seed=2),
            simulations=80,
            search_depth=6,
            seed=2,
        )
        root = mcts.search(np.zeros(4, dtype=np.float32))
        self.assertGreater(root.children[1].visits, root.children[0].visits)
        self.assertEqual(select_mcts_action(root), 1)

    def test_each_toggle_replaces_only_its_component(self):
        env = gym.make("CartPole-v1")
        exact = ExactCartPoleDynamics.from_env(env)
        env.close()
        models = {
            "dynamics": CountingDynamics(),
            "policy": ConstantPolicy(),
            "value": ConstantValue(),
        }
        value_only = build_mcts(
            frozenset({"value"}), models, exact, simulations=2, search_depth=4, seed=0
        )
        value_root = value_only.search(np.zeros(4, dtype=np.float32))
        self.assertEqual(models["value"].calls, 2)
        self.assertAlmostEqual(value_root.children[0].prior, 0.5)
        self.assertEqual(models["dynamics"].calls, 0)

        policy_only = build_mcts(
            frozenset({"policy"}), models, exact, simulations=2, search_depth=4, seed=1
        )
        policy_root = policy_only.search(np.zeros(4, dtype=np.float32))
        self.assertAlmostEqual(policy_root.children[1].prior, 0.9, places=5)
        self.assertEqual(models["dynamics"].calls, 0)

        dynamics_only = build_mcts(
            frozenset({"dynamics"}), models, exact, simulations=2, search_depth=4, seed=2
        )
        dynamics_only.search(np.zeros(4, dtype=np.float32))
        self.assertGreater(models["dynamics"].calls, 0)

    def test_learned_dynamics_hardcodes_cartpole_reward(self):
        dynamics = CountingDynamics()
        dynamics.predict = lambda state, action: (np.asarray(state), -99.0)
        transitions = LearnedCartPoleDynamics(dynamics, lambda _state: False)
        _, reward, done = transitions.step(np.zeros(4, dtype=np.float32), 0)
        self.assertEqual(reward, 1.0)
        self.assertFalse(done)

    def test_dynamics_curriculum_ends_at_mcts_depth(self):
        self.assertEqual(dynamics_curriculum(30), (5, 10, 20, 30))
        self.assertEqual(dynamics_curriculum(8), (5, 8))
        self.assertEqual(dynamics_curriculum(3), (3,))
        self.assertEqual(curriculum_horizon(29, 30, 30), 30)

    def test_value_target_horizon_is_rescaled_not_retrained_for_search_depth(self):
        value = ConstantValue(0.5)
        shallow = NetworkValueEvaluator(value, search_depth=10, value_horizon=100)
        deep = NetworkValueEvaluator(value, search_depth=20, value_horizon=100)
        self.assertEqual(shallow.evaluate(np.zeros(4), remaining_depth=10), 1.0)
        self.assertEqual(deep.evaluate(np.zeros(4), remaining_depth=10), 0.5)

    def test_dataset_can_skip_value_targets_for_policy_or_dynamics_only(self):
        dataset = collect_state_dataset(
            samples=40,
            value_horizon=500,
            include_value_targets=False,
            seed=5,
        )
        self.assertTrue(np.all(dataset.value_targets == 0.0))

    def test_model_selection_and_ablation_are_complete(self):
        selected = parse_models("value,dynamics,policy")
        configurations = component_configurations(selected, ablation=True)
        self.assertEqual(len(configurations), 8)
        self.assertIn(frozenset(), configurations)
        self.assertIn(selected, configurations)
        self.assertEqual(parse_models("none"), frozenset())

    def test_report_history_migrates_legacy_and_preserves_every_run(self):
        legacy = {
            "selected_models": [],
            "training_metrics": {},
            "evaluations": {"baseline": {"mean_reward": 123.0}},
            "settings": {"seed": 0},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(legacy), encoding="utf-8")
            history = load_report_history(path)
            self.assertEqual(history["schema_version"], 2)
            self.assertEqual(len(history["runs"]), 1)
            self.assertEqual(history["runs"][0]["evaluations"], legacy["evaluations"])

            history["runs"].append(
                {
                    "run_id": "second-run",
                    "status": "completed",
                    "selected_models": ["dynamics"],
                    "training_metrics": {},
                    "evaluations": {"dynamics": {"mean_reward": 50.0}},
                    "settings": {"seed": 1},
                }
            )
            write_report_history(path, history)
            reloaded = load_report_history(path)
            self.assertEqual(len(reloaded["runs"]), 2)
            self.assertEqual(reloaded["runs"][0]["evaluations"], legacy["evaluations"])
            self.assertEqual(reloaded["runs"][1]["run_id"], "second-run")
            self.assertFalse(path.with_suffix(".json.tmp").exists())

            # Simulate dashboard edits while an experiment still holds an
            # older in-memory history object.
            disk_history = json.loads(path.read_text(encoding="utf-8"))
            disk_history["dashboard_folders"] = [
                {"id": "references", "name": "References"}
            ]
            disk_history["dashboard_run_folders"] = {
                "legacy-imported-run": "references"
            }
            disk_history["runs"][0]["display_name"] = "Baseline reference"
            path.write_text(json.dumps(disk_history), encoding="utf-8")
            write_report_history(path, history)
            merged = load_report_history(path)
            self.assertEqual(merged["runs"][0]["display_name"], "Baseline reference")
            self.assertEqual(merged["dashboard_folders"], disk_history["dashboard_folders"])
            self.assertEqual(
                merged["dashboard_run_folders"],
                disk_history["dashboard_run_folders"],
            )

    def test_all_three_trainers_run_independently_and_checkpoint(self):
        dataset = collect_state_dataset(samples=240, value_horizon=8, seed=3)
        self.assertEqual(dataset.states.shape, (240, 4))
        self.assertEqual(dataset.episode_ids.shape, (240,))
        trainers = {
            "dynamics": train_dynamics,
            "policy": train_policy,
            "value": train_value,
        }
        with tempfile.TemporaryDirectory() as directory:
            for offset, (component, trainer) in enumerate(trainers.items()):
                trainer_kwargs = {"rollout_horizon": 8} if component == "dynamics" else {}
                loss_history = []
                model, metrics = trainer(
                    dataset,
                    hidden_dim=16,
                    epochs=2,
                    batch_size=64,
                    seed=10 + offset,
                    device=torch.device("cpu"),
                    loss_history=loss_history,
                    **trainer_kwargs,
                )
                self.assertEqual(len(loss_history), 2)
                self.assertEqual([entry["epoch"] for entry in loss_history], [1, 2])
                self.assertTrue(
                    all(np.isfinite(entry["training_loss"]) for entry in loss_history)
                )
                self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
                if component == "dynamics":
                    self.assertEqual(metrics["training_rollout_horizon"], 8.0)
                    self.assertIn("rollout_mae_h1", metrics)
                path = Path(directory) / f"{component}.pt"
                save_component(
                    path,
                    component,
                    model,
                    metrics,
                    value_horizon=8,
                    loss_history=loss_history,
                )
                loaded, checkpoint = load_component(
                    path, component, device=torch.device("cpu")
                )
                self.assertEqual(checkpoint["component"], component)
                self.assertEqual(checkpoint["loss_history"], loss_history)
                if component == "dynamics":
                    self.assertEqual(checkpoint["dynamics_training_version"], 2)
                    self.assertEqual(checkpoint["training_horizon"], 8)
                if component == "value":
                    self.assertEqual(checkpoint["value_target_horizon"], 8)
                self.assertEqual(
                    set(model.state_dict()),
                    set(loaded.state_dict()),
                )

    def test_model_loss_graph_is_written_in_run_component_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            output = write_loss_graph(
                directory,
                "test-run",
                "value",
                [{"epoch": 1, "training_loss": 0.25}],
                {"mae": 0.1},
                "trained",
                {"train_epochs": 1},
            )
            self.assertEqual(output, Path(directory) / "test-run" / "value.svg")
            svg = output.read_text(encoding="utf-8")
            self.assertIn("<svg", svg)
            self.assertIn("Value training loss", svg)
            self.assertIn("Epoch 1: 0.25", svg)
            self.assertFalse(output.with_suffix(".json").exists())


if __name__ == "__main__":
    unittest.main()
