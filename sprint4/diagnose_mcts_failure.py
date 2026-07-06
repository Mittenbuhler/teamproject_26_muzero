"""Read-only root-cause diagnostics for incoherent latent MCTS preferences."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from diagnose_latent_muzero import collect_diagnostic_episodes, describe
from diagnose_policy_behavior import (
    cross_validated_state_prediction,
    episode_folds,
)
from image_observation import ScreenshotConfig, make_image_env
from train_policy_value import load_latent_checkpoint
from utils import ensure_dir


STATE_NAMES = ("cart_position", "cart_velocity", "pole_angle", "pole_angular_velocity")


def correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or left.std() == 0 or right.std() == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def held_out_latent_state_probe(trajectories, seed=0, steps=400):
    latents = []
    states = []
    groups = []
    for episode_index, trajectory in enumerate(trajectories):
        for transition in trajectory:
            latents.append(transition["latent"])
            states.append(transition["environment_state"])
            groups.append(episode_index)
    latents = np.asarray(latents, dtype=np.float32)
    states = np.asarray(states, dtype=np.float32)
    groups = np.asarray(groups, dtype=np.int64)
    folds = episode_folds(groups, fold_count=5, seed=seed)
    predictions = np.zeros_like(states)
    baselines = np.zeros_like(states)

    for fold_index, test_episodes in enumerate(folds):
        train_mask = ~np.isin(groups, test_episodes)
        test_mask = ~train_mask
        latent_mean = latents[train_mask].mean(axis=0)
        latent_std = latents[train_mask].std(axis=0)
        latent_std[latent_std < 1e-6] = 1.0
        state_mean = states[train_mask].mean(axis=0)
        state_std = states[train_mask].std(axis=0)
        state_std[state_std < 1e-6] = 1.0
        train_x = torch.as_tensor(
            (latents[train_mask] - latent_mean) / latent_std,
            dtype=torch.float32,
        )
        train_y = torch.as_tensor(
            (states[train_mask] - state_mean) / state_std,
            dtype=torch.float32,
        )
        test_x = torch.as_tensor(
            (latents[test_mask] - latent_mean) / latent_std,
            dtype=torch.float32,
        )
        torch.manual_seed(seed + fold_index)
        probe = torch.nn.Sequential(
            torch.nn.Linear(latents.shape[1], 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, states.shape[1]),
        )
        optimizer = torch.optim.Adam(probe.parameters(), lr=3e-3, weight_decay=1e-4)
        for _ in range(steps):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(probe(train_x), train_y)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            standardized_prediction = probe(test_x).numpy()
        predictions[test_mask] = standardized_prediction * state_std + state_mean
        baselines[test_mask] = state_mean

    result = {}
    for index, name in enumerate(STATE_NAMES):
        model_error = np.square(states[:, index] - predictions[:, index]).sum()
        baseline_error = np.square(states[:, index] - baselines[:, index]).sum()
        result[name] = {
            "episode_held_out_r_squared": float(
                1.0 - model_error / max(baseline_error, 1e-12)
            ),
            "correlation": correlation(predictions[:, index], states[:, index]),
            "prediction_rmse": float(
                np.sqrt(np.square(states[:, index] - predictions[:, index]).mean())
            ),
            "constant_baseline_rmse": float(
                np.sqrt(np.square(states[:, index] - baselines[:, index]).mean())
            ),
        }
    return {
        "probe": "two-layer MLP trained only on four episode folds",
        "transitions": len(states),
        "folds": len(folds),
        "training_steps_per_fold": steps,
        "state_components": result,
        "minimum_r_squared": min(item["episode_held_out_r_squared"] for item in result.values()),
    }


def smooth_l1_per_row(predictions, targets):
    error = np.abs(predictions - targets)
    return np.where(error < 1.0, 0.5 * np.square(error), error - 0.5).mean(axis=1)


def cosine_per_row(left, right):
    return (left * right).sum(axis=1) / (
        np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1) + 1e-12
    )


def one_step_counterfactual_dynamics(
    trajectories,
    representation,
    dynamics,
    checkpoint,
    danger_window=10,
    maximum_states=200,
    seed=0,
):
    candidates = [
        transition
        for trajectory in trajectories
        if trajectory and trajectory[-1]["terminated"]
        for transition in trajectory[-danger_window:]
    ]
    random.Random(seed).shuffle(candidates)
    candidates = candidates[:maximum_states]
    image_size = checkpoint["image_size"]
    env = make_image_env(
        "CartPole-v1",
        screenshot_config=ScreenshotConfig(width=image_size, height=image_size),
    )
    records = []

    try:
        for state_index, transition in enumerate(candidates):
            action_records = []
            for action in range(checkpoint["action_dim"]):
                env.reset(seed=seed + state_index)
                env.unwrapped.state = np.asarray(
                    transition["environment_state"],
                    dtype=np.float64,
                ).copy()
                env.unwrapped.steps_beyond_terminated = None
                env.env._elapsed_steps = transition["elapsed_steps"]
                next_frame, reward, terminated, truncated, _ = env.step(action)
                real_next_state = np.asarray(
                    env.unwrapped.state,
                    dtype=np.float64,
                ).copy()
                next_observation = np.concatenate(
                    [transition["observation"][1:], next_frame],
                    axis=0,
                )
                target_next_latent = representation.encode(next_observation)
                (
                    predicted_next_latent,
                    predicted_reward,
                    terminal_probability,
                ) = dynamics.predict(transition["latent"], action)
                action_records.append(
                    {
                        "action": action,
                        "predicted_next_latent": predicted_next_latent,
                        "target_next_latent": target_next_latent,
                        "predicted_reward": predicted_reward,
                        "real_reward": float(reward),
                        "real_next_state": real_next_state,
                        "terminal_probability": terminal_probability,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                    }
                )
            records.append(
                {
                    "current_latent": transition["latent"],
                    "actions": action_records,
                }
            )
    finally:
        env.close()

    flat_actions = [action for record in records for action in record["actions"]]
    predicted = np.asarray(
        [action["predicted_next_latent"] for action in flat_actions],
        dtype=np.float64,
    )
    targets = np.asarray(
        [action["target_next_latent"] for action in flat_actions],
        dtype=np.float64,
    )
    identity = np.repeat(
        np.asarray([record["current_latent"] for record in records], dtype=np.float64),
        checkpoint["action_dim"],
        axis=0,
    )
    probabilities = np.asarray(
        [action["terminal_probability"] for action in flat_actions],
        dtype=np.float64,
    )
    labels = np.asarray(
        [action["terminated"] for action in flat_actions],
        dtype=np.bool_,
    )
    threshold_predictions = probabilities >= checkpoint.get("termination_threshold", 0.5)

    predicted_effects = []
    actual_effects = []
    physical_state_effects = []
    terminal_contrast_correct = []
    terminal_probability_margins = []
    for record in records:
        left, right = record["actions"]
        predicted_effects.append(
            right["predicted_next_latent"] - left["predicted_next_latent"]
        )
        actual_effects.append(right["target_next_latent"] - left["target_next_latent"])
        physical_state_effects.append(
            right["real_next_state"] - left["real_next_state"]
        )
        if left["terminated"] != right["terminated"]:
            terminating = right if right["terminated"] else left
            surviving = left if right["terminated"] else right
            margin = (
                terminating["terminal_probability"]
                - surviving["terminal_probability"]
            )
            terminal_probability_margins.append(margin)
            terminal_contrast_correct.append(margin > 0)

    predicted_effects = np.asarray(predicted_effects, dtype=np.float64)
    actual_effects = np.asarray(actual_effects, dtype=np.float64)
    physical_state_effects = np.asarray(physical_state_effects, dtype=np.float64)
    effect_cosine = cosine_per_row(predicted_effects, actual_effects)
    return {
        "sampled_danger_states": len(records),
        "real_action_transitions": len(flat_actions),
        "actual_terminal_transitions": int(labels.sum()),
        "latent_prediction": {
            "learned_smooth_l1": describe(smooth_l1_per_row(predicted, targets)),
            "identity_smooth_l1": describe(smooth_l1_per_row(identity, targets)),
            "learned_mae": describe(np.abs(predicted - targets).mean(axis=1)),
            "identity_mae": describe(np.abs(identity - targets).mean(axis=1)),
            "learned_cosine": describe(cosine_per_row(predicted, targets)),
            "identity_cosine": describe(cosine_per_row(identity, targets)),
        },
        "action_effect": {
            "real_hidden_state_right_minus_left": {
                name: describe(physical_state_effects[:, index])
                for index, name in enumerate(STATE_NAMES)
            },
            "predicted_l2_norm": describe(np.linalg.norm(predicted_effects, axis=1)),
            "actual_l2_norm": describe(np.linalg.norm(actual_effects, axis=1)),
            "predicted_actual_cosine": describe(effect_cosine),
            "positive_cosine_fraction": float((effect_cosine > 0).mean()),
        },
        "termination": {
            "recall": float(threshold_predictions[labels].mean()) if labels.any() else None,
            "false_positive_rate": (
                float(threshold_predictions[~labels].mean()) if (~labels).any() else None
            ),
            "terminal_probability": describe(probabilities[labels]),
            "nonterminal_probability": describe(probabilities[~labels]),
            "one_action_terminal_state_count": len(terminal_contrast_correct),
            "terminating_action_rank_accuracy": (
                float(np.mean(terminal_contrast_correct))
                if terminal_contrast_correct
                else None
            ),
            "terminating_minus_surviving_probability": describe(
                terminal_probability_margins
            ),
        },
    }


def mcts_component_diagnostics(trajectories, danger_window=10, seed=0):
    component_names = (
        "prior",
        "visit_share",
        "predicted_reward",
        "terminal_probability",
        "direct_child_value",
        "backed_up_child_value",
        "search_q",
    )
    episode_components = {name: [] for name in component_names}
    state_episodes = []

    for trajectory in trajectories:
        if not trajectory or not trajectory[-1]["terminated"] or len(trajectory) < danger_window:
            continue
        danger = trajectory[-danger_window:]
        state_episodes.append(
            np.asarray([transition["environment_state"] for transition in danger])
        )
        for name in component_names:
            values = []
            for transition in danger:
                left, right = transition["mcts_action_stats"]
                if name in {"prior", "visit_share"}:
                    values.append(right[name])
                else:
                    values.append(right[name] - left[name])
            episode_components[name].append(np.asarray(values, dtype=np.float64))

    states = np.concatenate(state_episodes)
    episode_count = len(state_episodes)
    groups = np.repeat(np.arange(episode_count), danger_window)
    folds = episode_folds(groups, fold_count=5, seed=seed)
    flattened = {
        name: np.concatenate(episodes)
        for name, episodes in episode_components.items()
    }
    state_dependence = {
        name: cross_validated_state_prediction(
            states,
            values,
            groups,
            folds,
        )
        for name, values in flattened.items()
    }
    visit_advantage = 2.0 * flattened["visit_share"] - 1.0
    correlations = {
        name: correlation(visit_advantage, values)
        for name, values in flattened.items()
        if name != "visit_share"
    }
    sign_agreement = {}
    for name, values in flattened.items():
        if name in {"visit_share", "predicted_reward"}:
            continue
        signed_values = 2.0 * values - 1.0 if name == "prior" else values
        sign_agreement[name] = float(
            np.mean(np.sign(visit_advantage) == np.sign(signed_values))
        )
    return {
        "danger_states": len(states),
        "component_distributions": {
            name: describe(values) for name, values in flattened.items()
        },
        "episode_held_out_state_dependence": state_dependence,
        "correlation_with_visit_advantage": correlations,
        "sign_agreement_with_visit_advantage": sign_agreement,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/latent_muzero_terminal_v9.pt",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--danger-window", type=int, default=10)
    parser.add_argument("--counterfactual-states", type=int, default=200)
    parser.add_argument("--probe-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default="artifacts/mcts_failure_diagnostics.json",
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
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_episode": len(checkpoint.get("history", {}).get("rewards", [])),
        "representation_state_probe": held_out_latent_state_probe(
            trajectories,
            seed=args.seed,
            steps=args.probe_steps,
        ),
        "one_step_counterfactual_dynamics": one_step_counterfactual_dynamics(
            trajectories,
            representation,
            dynamics,
            checkpoint,
            danger_window=args.danger_window,
            maximum_states=args.counterfactual_states,
            seed=args.seed,
        ),
        "mcts_components": mcts_component_diagnostics(
            trajectories,
            danger_window=args.danger_window,
            seed=args.seed,
        ),
    }
    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print("\n=== Representation state probe ===")
    print(json.dumps(report["representation_state_probe"], indent=2))
    print("\n=== Real one-step counterfactual dynamics ===")
    print(json.dumps(report["one_step_counterfactual_dynamics"], indent=2))
    print("\n=== Dangerous-state MCTS components ===")
    print(json.dumps(report["mcts_components"], indent=2))
    print("saved MCTS failure report:", output_path.resolve())


if __name__ == "__main__":
    main()
