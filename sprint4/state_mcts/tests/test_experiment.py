import tempfile
import unittest
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from state_mcts.run_state_mcts_experiment import (
    build_mcts,
    component_configurations,
    load_report_history,
    parse_models,
    write_loss_graph,
    write_report_history,
)
from state_mcts.train_policy_value_from_mcts import (
    discounted_return_normalizer,
    discounted_return_targets,
    load_report_history as load_distillation_report_history,
    parse_args as parse_distillation_args,
    visit_distribution,
    write_report_history as write_distillation_report_history,
)
from state_mcts.mcts_search import (
    ExactCartPoleDynamics,
    LearnedCartPoleDynamics,
    ModularMCTS,
    NetworkValueEvaluator,
    RandomRolloutEvaluator,
    UniformPrior,
    select_mcts_action,
)
from state_mcts.train_policy_value_dynamic_from_data import (
    collect_state_dataset,
    curriculum_horizon,
    dynamics_curriculum,
    load_component,
    save_component,
    train_dynamics,
    train_policy,
    train_value,
)
from state_mcts.diagnose_value_action_alignment import (
    select_records_to_store,
    summarize_records,
    value_implied_action,
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


class RightBetterTransitions:
    def step(self, state, action):
        next_state = np.asarray(state, dtype=np.float32).copy()
        next_state[0] = 1.0 if action else 0.0
        return next_state, 1.0, False


class NonTerminalDepthTransitions:
    def step(self, state, _action):
        next_state = np.asarray(state, dtype=np.float32).copy()
        next_state[0] += 1.0
        return next_state, 1.0, False


class DepthRecordingEvaluator:
    def __init__(self):
        self.depths = []

    def evaluate(self, state, _remaining_depth):
        self.depths.append(int(state[0]))
        return 0.0


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

    def test_bootstrap_after_delays_leaf_evaluation_depth(self):
        evaluator = DepthRecordingEvaluator()
        mcts = ModularMCTS(
            NonTerminalDepthTransitions(),
            UniformPrior(),
            evaluator,
            simulations=12,
            search_depth=5,
            bootstrap_after=3,
            seed=0,
        )
        mcts.search(np.zeros(4, dtype=np.float32))
        self.assertTrue(evaluator.depths)
        self.assertGreaterEqual(min(evaluator.depths), 3)

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

    def test_value_leaf_estimate_is_independent_of_search_depth(self):
        value = ConstantValue(0.5)
        evaluator = NetworkValueEvaluator(value, value_horizon=100)
        self.assertEqual(evaluator.evaluate(np.zeros(4), remaining_depth=10), 0.5)
        self.assertEqual(evaluator.evaluate(np.zeros(4), remaining_depth=0), 0.5)

    def test_discounted_return_targets_are_normalized_return_to_go(self):
        self.assertEqual(discounted_return_normalizer(1.0, 4), 4.0)
        targets = discounted_return_targets([1.0, 1.0, 1.0], discount=1.0, horizon=4)
        np.testing.assert_allclose(targets, [0.75, 0.5, 0.25])

        discounted = discounted_return_targets(
            [1.0, 1.0],
            discount=0.5,
            horizon=3,
        )
        normalizer = 1.0 + 0.5 + 0.25
        np.testing.assert_allclose(
            discounted,
            [(1.0 + 0.5) / normalizer, 1.0 / normalizer],
        )

    def test_distillation_value_target_cli_defaults(self):
        default_args = parse_distillation_args(["--models", "value"])
        self.assertIsNone(default_args.value_horizon)
        self.assertEqual(default_args.discount, 1.0)

        discounted_args = parse_distillation_args(
            ["--models", "value", "--discount", "0.99", "--value-horizon", "500"]
        )
        self.assertAlmostEqual(discounted_args.discount, 0.99)
        self.assertEqual(discounted_args.value_horizon, 500)

    def test_mcts_visit_distribution_is_normalized_policy_target(self):
        transitions = ActionSensitiveTransitions()
        root = ModularMCTS(
            transitions,
            UniformPrior(),
            RandomRolloutEvaluator(transitions, search_depth=6, seed=2),
            simulations=80,
            search_depth=6,
            seed=2,
        ).search(np.zeros(4, dtype=np.float32))
        target = visit_distribution(root)
        self.assertEqual(target.shape, (2,))
        self.assertAlmostEqual(float(target.sum()), 1.0)
        self.assertGreater(target[1], target[0])

    def test_value_mcts_diagnostic_scores_value_implied_action(self):
        action, scores, next_values, dones = value_implied_action(
            ConstantValue(0.0),
            RightBetterTransitions(),
            np.zeros(4, dtype=np.float32),
            search_depth=10,
        )
        self.assertEqual(action, 0)
        self.assertEqual(dones, [False, False])
        np.testing.assert_allclose(next_values, [0.0, 0.0])
        np.testing.assert_allclose(scores, [0.1, 0.1])

        class StateValue:
            def value(self, state):
                return float(state[0])

        action, scores, _, _ = value_implied_action(
            StateValue(),
            RightBetterTransitions(),
            np.zeros(4, dtype=np.float32),
            search_depth=10,
        )
        self.assertEqual(action, 1)
        self.assertGreater(scores[1], scores[0])

    def test_value_mcts_diagnostic_summary_reports_confident_agreement(self):
        records = [
            {
                "agreement": True,
                "value_margin": 0.2,
                "mcts_selected_visit_fraction": 0.8,
                "mcts_action": 1,
                "value_action": 1,
            },
            {
                "agreement": False,
                "value_margin": 0.1,
                "mcts_selected_visit_fraction": 0.5,
                "mcts_action": 0,
                "value_action": 1,
            },
            {
                "agreement": False,
                "value_margin": 0.3,
                "mcts_selected_visit_fraction": 0.75,
                "mcts_action": 1,
                "value_action": 0,
            },
        ]
        summary = summarize_records(records)
        self.assertEqual(summary["samples"], 3)
        self.assertEqual(summary["wrong_action_count"], 2)
        self.assertAlmostEqual(summary["agreement"], 1 / 3)
        self.assertAlmostEqual(
            summary["agreement_when_mcts_visit_fraction_at_least_0.70"],
            0.5,
        )

    def test_value_mcts_diagnostic_can_store_high_confidence_records(self):
        records = [
            {"agreement": True, "mcts_selected_visit_fraction": 0.55},
            {"agreement": False, "mcts_selected_visit_fraction": 0.8},
            {"agreement": True, "mcts_selected_visit_fraction": 0.9},
        ]
        selected = select_records_to_store(
            records,
            limit=5,
            mode="high-confidence",
            confidence_threshold=0.7,
        )
        self.assertEqual(selected, records[1:])
        selected = select_records_to_store(
            records,
            limit=5,
            mode="high-confidence-disagreements",
            confidence_threshold=0.7,
        )
        self.assertEqual(selected, [records[1]])

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

    def test_distillation_report_history_migrates_legacy_and_appends_runs(self):
        legacy = {
            "schema_version": 1,
            "run_id": "old-distillation",
            "selected_models": ["value"],
            "evaluations": {"hybrid_mcts_value_distilled": {"mean_reward": 500.0}},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state_mcts_value_only_report.json"
            path.write_text(json.dumps(legacy), encoding="utf-8")
            history = load_distillation_report_history(path)
            self.assertEqual(history["schema_version"], 2)
            self.assertEqual(len(history["runs"]), 1)
            self.assertEqual(history["runs"][0]["run_id"], "old-distillation")

            history["runs"].append(
                {
                    "run_id": "new-distillation",
                    "status": "completed",
                    "selected_models": ["policy"],
                    "evaluations": {"hybrid_mcts_policy_distilled": {"mean_reward": 450.0}},
                }
            )
            write_distillation_report_history(path, history)
            reloaded = load_distillation_report_history(path)
            self.assertEqual(len(reloaded["runs"]), 2)
            self.assertEqual(reloaded["runs"][0]["run_id"], "old-distillation")
            self.assertEqual(reloaded["runs"][1]["run_id"], "new-distillation")
            self.assertFalse(path.with_suffix(".json.tmp").exists())

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
