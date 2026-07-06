"""Sequential policy diagnostics: reactivity first, correctness only after it passes."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from .diagnose_latent_muzero import collect_diagnostic_episodes, describe
from .train_policy_value import load_latent_checkpoint
from .utils import ensure_dir


REACTIVITY_CRITERIA = {
    "minimum_eligible_episodes": 20,
    "minimum_danger_p_right_standard_deviation": 0.02,
    "minimum_episode_held_out_r_squared": 0.10,
    "maximum_episode_permutation_p_value": 0.01,
}


def bootstrap_mean_interval(values, seed=0, draws=10_000):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return [None, None]
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def polynomial_state_features(states, mean, standard_deviation):
    standardized = (states - mean) / standard_deviation
    columns = [standardized]
    columns.append(np.square(standardized))
    columns.append(
        np.column_stack(
            [
                standardized[:, left] * standardized[:, right]
                for left in range(standardized.shape[1])
                for right in range(left + 1, standardized.shape[1])
            ]
        )
    )
    return np.column_stack(columns)


def episode_folds(groups, fold_count, seed):
    episodes = np.unique(groups)
    generator = np.random.default_rng(seed)
    episodes = generator.permutation(episodes)
    return np.array_split(episodes, min(fold_count, len(episodes)))


def cross_validated_state_prediction(
    states,
    targets,
    groups,
    folds,
    ridge_alpha=1.0,
):
    predictions = np.zeros_like(targets, dtype=np.float64)
    baselines = np.zeros_like(targets, dtype=np.float64)

    for test_episodes in folds:
        test_mask = np.isin(groups, test_episodes)
        train_mask = ~test_mask
        train_states = states[train_mask]
        mean = train_states.mean(axis=0)
        standard_deviation = train_states.std(axis=0)
        standard_deviation[standard_deviation < 1e-8] = 1.0
        train_features = polynomial_state_features(
            train_states,
            mean,
            standard_deviation,
        )
        test_features = polynomial_state_features(
            states[test_mask],
            mean,
            standard_deviation,
        )
        target_mean = float(targets[train_mask].mean())
        regularizer = ridge_alpha * np.eye(train_features.shape[1])
        coefficients = np.linalg.solve(
            train_features.T @ train_features + regularizer,
            train_features.T @ (targets[train_mask] - target_mean),
        )
        predictions[test_mask] = target_mean + test_features @ coefficients
        baselines[test_mask] = target_mean

    model_squared_error = float(np.square(targets - predictions).sum())
    baseline_squared_error = float(np.square(targets - baselines).sum())
    r_squared = (
        1.0 - model_squared_error / baseline_squared_error
        if baseline_squared_error > 1e-12
        else 0.0
    )
    correlation = (
        float(np.corrcoef(predictions, targets)[0, 1])
        if predictions.std() > 0 and targets.std() > 0
        else None
    )
    return {
        "r_squared_vs_train_mean_baseline": float(r_squared),
        "prediction_target_correlation": correlation,
        "prediction_rmse": float(np.sqrt(np.square(targets - predictions).mean())),
        "constant_baseline_rmse": float(
            np.sqrt(np.square(targets - baselines).mean())
        ),
    }


def policy_reactivity_test(
    trajectories,
    danger_window=10,
    permutation_seed=0,
    permutation_count=500,
    policy_key="network_policy",
    policy_name="policy network",
):
    """Test whether a dangerous-state policy output depends on physical state."""
    state_episodes = []
    probability_episodes = []

    for trajectory in trajectories:
        if not trajectory or not trajectory[-1]["terminated"]:
            continue
        if len(trajectory) < danger_window:
            continue
        danger = trajectory[-danger_window:]
        if any("environment_state" not in transition for transition in danger):
            continue
        state_episodes.append(
            np.asarray(
                [transition["environment_state"] for transition in danger],
                dtype=np.float64,
            )
        )
        probability_episodes.append(
            np.asarray(
                [transition[policy_key][1] for transition in danger],
                dtype=np.float64,
            )
        )

    eligible_episodes = len(state_episodes)
    if eligible_episodes:
        states = np.concatenate(state_episodes)
        probabilities = np.concatenate(probability_episodes)
        groups = np.repeat(np.arange(eligible_episodes), danger_window)
        probability_matrix = np.asarray(probability_episodes)
    else:
        states = np.empty((0, 4), dtype=np.float64)
        probabilities = np.empty(0, dtype=np.float64)
        groups = np.empty(0, dtype=np.int64)
        probability_matrix = np.empty((0, danger_window), dtype=np.float64)

    if eligible_episodes >= 2:
        folds = episode_folds(groups, fold_count=5, seed=permutation_seed)
        observed = cross_validated_state_prediction(
            states,
            probabilities,
            groups,
            folds,
        )
        generator = np.random.default_rng(permutation_seed)
        null_r_squared = []
        for _ in range(permutation_count):
            episode_permutation = generator.permutation(eligible_episodes)
            permuted_matrix = probability_matrix[episode_permutation].copy()
            for episode_index in range(eligible_episodes):
                shift = int(generator.integers(0, danger_window))
                permuted_matrix[episode_index] = np.roll(
                    permuted_matrix[episode_index],
                    shift,
                )
            permuted_probabilities = permuted_matrix.reshape(-1)
            null_r_squared.append(
                cross_validated_state_prediction(
                    states,
                    permuted_probabilities,
                    groups,
                    folds,
                )["r_squared_vs_train_mean_baseline"]
            )
        null_r_squared = np.asarray(null_r_squared, dtype=np.float64)
        permutation_p_value = float(
            (1 + (null_r_squared >= observed["r_squared_vs_train_mean_baseline"]).sum())
            / (permutation_count + 1)
        )
    else:
        observed = {
            "r_squared_vs_train_mean_baseline": float("-inf"),
            "prediction_target_correlation": None,
            "prediction_rmse": None,
            "constant_baseline_rmse": None,
        }
        null_r_squared = np.empty(0, dtype=np.float64)
        permutation_p_value = 1.0

    probability_std = float(probabilities.std()) if probabilities.size else 0.0
    checks = {
        "enough_eligible_episodes": (
            eligible_episodes >= REACTIVITY_CRITERIA["minimum_eligible_episodes"]
        ),
        "danger_policy_output_varies": (
            probability_std
            >= REACTIVITY_CRITERIA[
                "minimum_danger_p_right_standard_deviation"
            ]
        ),
        "physical_state_predicts_held_out_policy_output": (
            observed["r_squared_vs_train_mean_baseline"]
            >= REACTIVITY_CRITERIA["minimum_episode_held_out_r_squared"]
        ),
        "state_dependence_beats_episode_permutation": (
            permutation_p_value
            <= REACTIVITY_CRITERIA["maximum_episode_permutation_p_value"]
        ),
    }
    return {
        "question": (
            f"Within dangerous pre-failure states, does the {policy_name} output "
            "change predictably with the underlying physical state?"
        ),
        "policy_source": policy_key,
        "passed": bool(all(checks.values())),
        "criteria": REACTIVITY_CRITERIA,
        "checks": checks,
        "collected_episodes": len(trajectories),
        "eligible_terminated_episodes": eligible_episodes,
        "excluded_episodes": len(trajectories) - eligible_episodes,
        "danger_window": danger_window,
        "danger_transition_count": int(len(probabilities)),
        "danger_p_right": describe(probabilities),
        "episode_held_out_state_model": observed,
        "episode_permutation_test": {
            "permutations": permutation_count,
            "p_value": permutation_p_value,
            "method": (
                "Permute complete policy sequences between episodes and apply "
                "an independent circular shift within each sequence."
            ),
            "null_r_squared": describe(null_r_squared),
            "null_99th_percentile": (
                float(np.quantile(null_r_squared, 0.99))
                if null_r_squared.size
                else None
            ),
        },
        "diagnostic_state_usage": (
            "CartPole's [x, x_dot, theta, theta_dot] is used only as an "
            "after-the-fact diagnostic label and is never passed to MuZero."
        ),
    }


def teacher_student_diagnosis(network_test, mcts_test):
    network_passed = network_test["passed"]
    mcts_passed = mcts_test["passed"]
    if mcts_passed and not network_passed:
        conclusion = "policy_imitation_bottleneck"
        explanation = (
            "MCTS has state-dependent preferences, but the policy network does "
            "not reproduce them consistently on held-out episodes."
        )
    elif not mcts_passed and not network_passed:
        conclusion = "mcts_teacher_is_not_state_coherent"
        explanation = (
            "Neither MCTS visits nor the policy network form a generalizable "
            "danger-state response; changing only the policy head is unlikely "
            "to fix the source signal."
        )
    elif mcts_passed and network_passed:
        conclusion = "teacher_and_student_are_state_coherent"
        explanation = (
            "Both MCTS and the policy network show generalizable state-dependent "
            "preferences in dangerous states."
        )
    else:
        conclusion = "policy_differs_from_current_mcts_teacher"
        explanation = (
            "The policy network is state-dependent while current MCTS visits are "
            "not; this may indicate stale learned behavior or unstable search."
        )
    return {"conclusion": conclusion, "explanation": explanation}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Test policy reactivity first; correctness is only tested after "
            "reactivity passes."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/legacy_muzero/latent_muzero_terminal_v9.pt",
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--danger-window", type=int, default=10)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default="artifacts/legacy_muzero/diagnostics/policy_reactivity.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    representation, dynamics, policy, value, checkpoint = load_latent_checkpoint(
        args.checkpoint,
        device=device,
    )
    trajectories = collect_diagnostic_episodes(
        checkpoint,
        representation,
        dynamics,
        policy,
        value,
        episodes=args.episodes,
        simulations=args.simulations,
        seed=args.seed,
        terminal_penalty=checkpoint.get("terminal_penalty", -25.0),
        device=device,
    )
    reactivity = policy_reactivity_test(
        trajectories,
        danger_window=args.danger_window,
        permutation_seed=args.seed,
        permutation_count=args.permutations,
    )
    mcts_reactivity = policy_reactivity_test(
        trajectories,
        danger_window=args.danger_window,
        permutation_seed=args.seed,
        permutation_count=args.permutations,
        policy_key="policy",
        policy_name="MCTS visit policy",
    )
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_episode": len(checkpoint.get("history", {}).get("rewards", [])),
        "reactivity_test": reactivity,
        "mcts_reactivity_test": mcts_reactivity,
        "teacher_student_diagnosis": teacher_student_diagnosis(
            reactivity,
            mcts_reactivity,
        ),
        "correctness_test": {
            "status": (
                "eligible_for_implementation"
                if reactivity["passed"]
                else "skipped_because_reactivity_did_not_pass"
            )
        },
    }
    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\n=== Policy reactivity gate ===")
    print(json.dumps(reactivity, indent=2))
    print("\n=== MCTS teacher reactivity ===")
    print(json.dumps(mcts_reactivity, indent=2))
    print("\n=== Teacher/student diagnosis ===")
    print(json.dumps(report["teacher_student_diagnosis"], indent=2))
    print("saved policy behavior report:", output_path.resolve())
    if not reactivity["passed"]:
        print("reactivity gate failed; correctness test was not implemented or run")
        raise SystemExit(2)
    print("reactivity gate passed; the correctness test may now be implemented")


if __name__ == "__main__":
    main()
