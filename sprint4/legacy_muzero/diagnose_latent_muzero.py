"""Read-only diagnostics for the image-latent MuZero training pipeline."""

import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from .image_observation import ScreenshotConfig, make_image_env
from .mcts import ModelBasedMCTS, select_action, visit_count_policy
from .train_policy_value import (
    bootstrapped_value_targets,
    full_episode_value_targets,
    load_latent_checkpoint,
    make_frame_stack,
    make_latent_mcts,
    shaped_environment_reward,
    train_latent_step,
)
from .utils import ensure_dir


def finite_float(value):
    value = float(value)
    return value if np.isfinite(value) else None


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": finite_float(values.mean()),
        "std": finite_float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": finite_float(values.min()),
        "median": finite_float(np.median(values)),
        "max": finite_float(values.max()),
    }


def safe_correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or np.isclose(left.std(), 0) or np.isclose(right.std(), 0):
        return None
    return finite_float(np.corrcoef(left, right)[0, 1])


def entropy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    return float(-(probabilities * np.log(probabilities + 1e-12)).sum())


def error_metrics(predictions, targets):
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if predictions.size == 0:
        return {"count": 0}
    errors = predictions - targets
    return {
        "count": int(predictions.size),
        "mae": finite_float(np.abs(errors).mean()),
        "rmse": finite_float(np.sqrt(np.square(errors).mean())),
        "bias": finite_float(errors.mean()),
        "correlation": safe_correlation(predictions, targets),
        "prediction": describe(predictions),
        "target": describe(targets),
    }


class UniformToyPolicy:
    def action_probs(self, _state):
        return np.asarray([0.5, 0.5], dtype=np.float32)


class ZeroToyValue:
    def value(self, _state):
        return 0.0


class BranchToyValue:
    def value(self, state):
        return 1.0 if state[1] > 0.5 else -1.0


class RewardToyDynamics:
    def predict(self, state, action):
        return (
            np.asarray([state[0] + 1, action], dtype=np.float32),
            float(action),
            0.0,
        )


class ValueToyDynamics:
    def predict(self, state, action):
        branch = action if state[0] == 0 else state[1]
        return np.asarray([state[0] + 1, branch], dtype=np.float32), 0.0, 0.0


def run_toy_mcts_checks():
    reward_search = ModelBasedMCTS(
        RewardToyDynamics(),
        UniformToyPolicy(),
        ZeroToyValue(),
        action_dim=2,
        terminal_fn=lambda state: state[0] >= 1,
        simulations=32,
        discount=0.9,
        exploration_c=1.0,
    )
    reward_root = reward_search.search(np.asarray([0.0, 0.0], dtype=np.float32))
    reward_visits = [reward_root.children[action].visit_count for action in range(2)]

    value_search = ModelBasedMCTS(
        ValueToyDynamics(),
        UniformToyPolicy(),
        BranchToyValue(),
        action_dim=2,
        terminal_fn=lambda _state: False,
        simulations=32,
        discount=0.9,
        exploration_c=1.0,
    )
    value_root = value_search.search(np.asarray([0.0, 0.0], dtype=np.float32))
    value_visits = [value_root.children[action].visit_count for action in range(2)]

    return {
        "reward_preferred_action": int(np.argmax(reward_visits)),
        "reward_visits": reward_visits,
        "reward_check_passed": bool(reward_visits[1] > reward_visits[0]),
        "value_preferred_action": int(np.argmax(value_visits)),
        "value_visits": value_visits,
        "value_check_passed": bool(value_visits[1] > value_visits[0]),
    }


def collect_diagnostic_episodes(
    checkpoint,
    representation,
    dynamics,
    policy,
    value,
    episodes,
    simulations,
    seed,
    terminal_penalty,
    device,
):
    image_size = checkpoint["image_size"]
    stack_size = checkpoint["stack_size"]
    max_steps = checkpoint["max_steps"]
    discount = checkpoint["value_discount"]
    env = make_image_env(
        "CartPole-v1",
        screenshot_config=ScreenshotConfig(width=image_size, height=image_size),
    )
    env.action_space.seed(seed + 10_000)
    mcts = make_latent_mcts(
        dynamics,
        policy,
        value,
        action_dim=checkpoint["action_dim"],
        simulations=simulations,
        discount=discount,
        max_steps=max_steps,
        terminal_penalty=terminal_penalty,
        termination_threshold=checkpoint.get("termination_threshold", 0.5),
        use_learned_termination=(
            checkpoint.get("training_version") >= 9
            and checkpoint.get("learned_termination_ready", True)
        ),
    )
    trajectories = []

    try:
        for episode_index in range(episodes):
            evaluation_seed = seed + 10_000 + episode_index
            random.seed(evaluation_seed)
            np.random.seed(evaluation_seed)
            observation, _ = env.reset(seed=evaluation_seed)
            frame_stack = make_frame_stack(stack_size, image_size)
            stacked_observation = frame_stack.reset(observation)
            trajectory = []

            for _ in range(max_steps):
                environment_state = np.asarray(
                    env.unwrapped.state,
                    dtype=np.float64,
                ).copy()
                elapsed_steps = int(getattr(env.env, "_elapsed_steps", 0))
                latent = representation.encode(stacked_observation)
                network_policy = policy.action_probs(latent)
                predicted_value = value.value(latent)
                root = mcts.search(latent)
                search_policy = visit_count_policy(root, temperature=1.0)
                mcts_action_stats = []
                for candidate_action in range(checkpoint["action_dim"]):
                    child = root.children[candidate_action]
                    (
                        _,
                        direct_reward,
                        terminal_probability,
                    ) = dynamics.predict(latent, candidate_action)
                    mcts_action_stats.append(
                        {
                            "action": candidate_action,
                            "prior": float(child.prior),
                            "visit_share": float(search_policy[candidate_action]),
                            "visit_count": int(child.visit_count),
                            "predicted_reward": float(direct_reward),
                            "terminal_probability": float(terminal_probability),
                            "search_reward": float(child.reward),
                            "done": bool(child.done),
                            "direct_child_value": float(value.value(child.state)),
                            "backed_up_child_value": float(child.mean_value),
                            "search_q": float(
                                child.reward + discount * child.mean_value
                            ),
                        }
                    )
                action, _ = select_action(root, temperature=0.0)
                (
                    predicted_next_latent,
                    predicted_reward,
                    predicted_terminal_probability,
                ) = dynamics.predict(latent, action)

                next_observation, real_reward, terminated, truncated, _ = env.step(action)
                next_stacked_observation = frame_stack.append(next_observation)
                target_next_latent = representation.encode(next_stacked_observation)
                value_reward = shaped_environment_reward(
                    real_reward,
                    terminated=terminated,
                    terminal_penalty=terminal_penalty,
                )
                trajectory.append(
                    {
                        "observation": stacked_observation.copy(),
                        "action": int(action),
                        "next_observation": next_stacked_observation.copy(),
                        "policy": search_policy.copy(),
                        "network_policy": network_policy.copy(),
                        "predicted_value": float(predicted_value),
                        "search_value": float(root.mean_value),
                        "mcts_action_stats": mcts_action_stats,
                        "reward": float(real_reward),
                        "value_reward": float(value_reward),
                        "real_reward": float(real_reward),
                        "predicted_reward": float(predicted_reward),
                        "predicted_terminal_probability": float(
                            predicted_terminal_probability
                        ),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "environment_state": environment_state,
                        "elapsed_steps": elapsed_steps,
                        "latent": latent.copy(),
                        "predicted_next_latent": predicted_next_latent.copy(),
                        "target_next_latent": target_next_latent.copy(),
                    }
                )
                stacked_observation = next_stacked_observation
                if terminated or truncated:
                    break

            if checkpoint.get("value_target_mode", "full-episode") == "n-step":
                value_targets = bootstrapped_value_targets(
                    trajectory,
                    discount=discount,
                    bootstrap_steps=checkpoint.get("bootstrap_steps", 10),
                    max_steps=max_steps,
                )
            else:
                value_targets = full_episode_value_targets(
                    trajectory,
                    discount=discount,
                    max_steps=max_steps,
                )
            for transition, target in zip(trajectory, value_targets):
                transition["value"] = float(target)
            trajectories.append(trajectory)
            print(
                f"diagnostic evaluation {episode_index + 1:2d}/{episodes} "
                f"reward={sum(item['real_reward'] for item in trajectory):5.1f}"
            )
    finally:
        env.close()

    return trajectories


def representation_diagnostics(transitions):
    latents = np.asarray([item["latent"] for item in transitions], dtype=np.float64)
    centered = latents - latents.mean(axis=0, keepdims=True)
    dimension_stds = centered.std(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    variance_weights = np.square(singular_values)
    if variance_weights.sum() > 0:
        variance_weights /= variance_weights.sum()
        effective_rank = math.exp(
            -float((variance_weights * np.log(variance_weights + 1e-12)).sum())
        )
    else:
        effective_rank = 0.0

    sample = latents[: min(500, len(latents))]
    if len(sample) > 1:
        differences = sample[1:] - sample[:-1]
        consecutive_distances = np.linalg.norm(differences, axis=1)
    else:
        consecutive_distances = []

    return {
        "latent_norm": describe(np.linalg.norm(latents, axis=1)),
        "dimension_std": describe(dimension_stds),
        "active_dimensions_std_gt_1e-3": int((dimension_stds > 1e-3).sum()),
        "active_dimensions_std_gt_1e-2": int((dimension_stds > 1e-2).sum()),
        "effective_rank": finite_float(effective_rank),
        "consecutive_state_distance": describe(consecutive_distances),
    }


def policy_diagnostics(transitions):
    network_policies = np.asarray(
        [item["network_policy"] for item in transitions], dtype=np.float64
    )
    search_policies = np.asarray(
        [item["policy"] for item in transitions], dtype=np.float64
    )
    network_entropies = [entropy(item) for item in network_policies]
    search_entropies = [entropy(item) for item in search_policies]
    cross_entropies = -(
        search_policies * np.log(network_policies + 1e-12)
    ).sum(axis=1)
    target_entropies = np.asarray(search_entropies)
    kl_divergences = cross_entropies - target_entropies
    return {
        "network_entropy": describe(network_entropies),
        "mcts_visit_entropy": describe(search_entropies),
        "mcts_max_visit_share": describe(search_policies.max(axis=1)),
        "cross_entropy": describe(cross_entropies),
        "target_entropy_lower_bound": describe(target_entropies),
        "network_mcts_kl_divergence": describe(kl_divergences),
        "network_mcts_argmax_agreement": finite_float(
            (
                network_policies.argmax(axis=1)
                == search_policies.argmax(axis=1)
            ).mean()
        ),
        "mcts_near_uniform_fraction_max_share_lt_0_6": finite_float(
            (search_policies.max(axis=1) < 0.6).mean()
        ),
        "network_near_uniform_fraction_max_prob_lt_0_6": finite_float(
            (network_policies.max(axis=1) < 0.6).mean()
        ),
        "mean_l1_policy_change_by_search": finite_float(
            np.abs(network_policies - search_policies).sum(axis=1).mean()
        ),
    }


def value_diagnostics(transitions):
    predictions = np.asarray([item["predicted_value"] for item in transitions])
    targets = np.asarray([item["value"] for item in transitions])
    search_values = [item["search_value"] for item in transitions]
    return {
        "network_vs_episode_target": error_metrics(predictions, targets),
        "mcts_vs_episode_target": error_metrics(search_values, targets),
        "network_vs_mcts": error_metrics(predictions, search_values),
        "zero_prediction_rmse": finite_float(np.sqrt(np.square(targets).mean())),
        "target_mean_prediction_rmse": finite_float(targets.std()),
    }


def reward_diagnostics(transitions):
    normal = [item for item in transitions if not item["terminated"]]
    terminal = [item for item in transitions if item["terminated"]]

    def metrics(items):
        return error_metrics(
            [item["predicted_reward"] for item in items],
            [item["reward"] for item in items],
        )

    targets = np.asarray([item["reward"] for item in transitions])
    always_one_rmse = float(np.sqrt(np.square(1.0 - targets).mean()))
    all_metrics = metrics(transitions)
    all_metrics["always_predict_one_rmse"] = always_one_rmse
    all_metrics["terminal_fraction"] = float(len(terminal) / len(transitions))
    return {
        "all": all_metrics,
        "ordinary": metrics(normal),
        "terminal": metrics(terminal),
    }


def termination_diagnostics(transitions, threshold=0.5):
    probabilities = np.asarray(
        [item["predicted_terminal_probability"] for item in transitions]
    )
    targets = np.asarray([item["terminated"] for item in transitions], dtype=np.float64)
    predictions = probabilities >= threshold
    positives = targets == 1
    negatives = ~positives
    return {
        "probability_vs_target": error_metrics(probabilities, targets),
        "threshold": threshold,
        "recall": finite_float(predictions[positives].mean()) if positives.any() else None,
        "false_positive_rate": (
            finite_float(predictions[negatives].mean()) if negatives.any() else None
        ),
        "terminal_probability": describe(probabilities[positives]),
        "nonterminal_probability": describe(probabilities[negatives]),
    }


def one_step_latent_diagnostics(transitions):
    current = np.asarray([item["latent"] for item in transitions], dtype=np.float64)
    predictions = np.asarray(
        [item["predicted_next_latent"] for item in transitions], dtype=np.float64
    )
    targets = np.asarray(
        [item["target_next_latent"] for item in transitions], dtype=np.float64
    )
    smooth_l1 = np.where(
        np.abs(predictions - targets) < 1,
        0.5 * np.square(predictions - targets),
        np.abs(predictions - targets) - 0.5,
    ).mean(axis=1)
    cosine = (predictions * targets).sum(axis=1) / (
        np.linalg.norm(predictions, axis=1)
        * np.linalg.norm(targets, axis=1)
        + 1e-12
    )

    def comparison(candidate):
        absolute_error = np.abs(candidate - targets)
        smooth_l1_error = np.where(
            absolute_error < 1,
            0.5 * np.square(absolute_error),
            absolute_error - 0.5,
        ).mean(axis=1)
        cosine_similarity = (candidate * targets).sum(axis=1) / (
            np.linalg.norm(candidate, axis=1)
            * np.linalg.norm(targets, axis=1)
            + 1e-12
        )
        return {
            "smooth_l1": describe(smooth_l1_error),
            "mae": describe(absolute_error.mean(axis=1)),
            "cosine_similarity": describe(cosine_similarity),
            "norm": describe(np.linalg.norm(candidate, axis=1)),
        }

    return {
        "dynamics_prediction": {
            "smooth_l1": describe(smooth_l1),
            "mae": describe(np.abs(predictions - targets).mean(axis=1)),
            "cosine_similarity": describe(cosine),
            "norm": describe(np.linalg.norm(predictions, axis=1)),
        },
        "identity_current_latent_baseline": comparison(current),
        "zero_latent_baseline": comparison(np.zeros_like(targets)),
        "target_norm": describe(np.linalg.norm(targets, axis=1)),
    }


@torch.no_grad()
def action_sensitivity_diagnostics(transitions, dynamics, policy, value, device):
    latents = torch.as_tensor(
        np.asarray([item["latent"] for item in transitions]),
        dtype=torch.float32,
        device=device,
    )
    action_zero = torch.zeros((len(latents), 2), dtype=torch.float32, device=device)
    action_one = torch.zeros((len(latents), 2), dtype=torch.float32, device=device)
    action_zero[:, 0] = 1.0
    action_one[:, 1] = 1.0
    next_zero, reward_zero, terminal_zero = dynamics(latents, action_zero)
    next_one, reward_one, terminal_one = dynamics(latents, action_one)
    policy_zero = policy(next_zero)
    policy_one = policy(next_one)
    value_zero = value(next_zero).squeeze(1)
    value_one = value(next_one).squeeze(1)
    return {
        "next_latent_l2_difference": describe(
            torch.linalg.vector_norm(next_zero - next_one, dim=1).cpu().numpy()
        ),
        "reward_absolute_difference": describe(
            torch.abs(reward_zero - reward_one).squeeze(1).cpu().numpy()
        ),
        "termination_probability_absolute_difference": describe(
            torch.abs(
                torch.sigmoid(terminal_zero) - torch.sigmoid(terminal_one)
            ).squeeze(1).cpu().numpy()
        ),
        "next_policy_l1_difference": describe(
            torch.abs(policy_zero - policy_one).sum(dim=1).cpu().numpy()
        ),
        "next_value_absolute_difference": describe(
            torch.abs(value_zero - value_one).cpu().numpy()
        ),
    }


@torch.no_grad()
def multistep_rollout_diagnostics(
    trajectories,
    dynamics,
    policy,
    value,
    horizons,
    max_starts,
    device,
):
    starts = [
        (episode_index, start)
        for episode_index, trajectory in enumerate(trajectories)
        for start in range(len(trajectory))
    ]
    random.Random(12345).shuffle(starts)
    starts = starts[:max_starts]
    current = torch.as_tensor(
        np.asarray(
            [trajectories[episode][start]["latent"] for episode, start in starts]
        ),
        dtype=torch.float32,
        device=device,
    )
    active = starts
    results = {}

    for horizon in range(1, horizons + 1):
        valid_positions = [
            index
            for index, (episode, start) in enumerate(active)
            if start + horizon - 1 < len(trajectories[episode])
        ]
        if not valid_positions:
            break
        current = current[valid_positions]
        active = [active[index] for index in valid_positions]
        actions = torch.zeros((len(active), 2), dtype=torch.float32, device=device)
        reward_targets = []
        latent_targets = []
        for index, (episode, start) in enumerate(active):
            transition = trajectories[episode][start + horizon - 1]
            actions[index, transition["action"]] = 1.0
            reward_targets.append(transition["reward"])
            latent_targets.append(transition["target_next_latent"])

        current, predicted_rewards, predicted_terminal_logits = dynamics(
            current,
            actions,
        )
        target_tensor = torch.as_tensor(
            np.asarray(latent_targets), dtype=torch.float32, device=device
        )
        predicted_latent_array = current.cpu().numpy()
        target_latent_array = target_tensor.cpu().numpy()
        latent_mae = np.abs(
            predicted_latent_array - target_latent_array
        ).mean(axis=1)
        latent_smooth_l1 = torch.nn.functional.smooth_l1_loss(
            current, target_tensor, reduction="none"
        ).mean(dim=1).cpu().numpy()
        latent_cosine = (
            (predicted_latent_array * target_latent_array).sum(axis=1)
            / (
                np.linalg.norm(predicted_latent_array, axis=1)
                * np.linalg.norm(target_latent_array, axis=1)
                + 1e-12
            )
        )
        reward_predictions = predicted_rewards.squeeze(1).cpu().numpy()
        terminal_predictions = torch.sigmoid(
            predicted_terminal_logits
        ).squeeze(1).cpu().numpy()
        terminal_targets = [
            float(trajectories[episode][start + horizon - 1]["terminated"])
            for episode, start in active
        ]

        next_policy_cross_entropies = []
        next_value_predictions = []
        next_value_targets = []
        next_policy_predictions = policy(current).cpu().numpy()
        next_policy_entropies = [
            entropy(probabilities) for probabilities in next_policy_predictions
        ]
        value_predictions = value(current).squeeze(1).cpu().numpy()
        for index, (episode, start) in enumerate(active):
            next_index = start + horizon
            if next_index < len(trajectories[episode]):
                transition = trajectories[episode][next_index]
                next_policy_cross_entropies.append(
                    -float(
                        (
                            transition["policy"]
                            * np.log(next_policy_predictions[index] + 1e-12)
                        ).sum()
                    )
                )
                next_value_predictions.append(value_predictions[index])
                next_value_targets.append(transition["value"])

        results[str(horizon)] = {
            "count": len(active),
            "latent_mae": describe(latent_mae),
            "latent_smooth_l1": describe(latent_smooth_l1),
            "latent_cosine_similarity": describe(latent_cosine),
            "predicted_latent_norm": describe(
                np.linalg.norm(predicted_latent_array, axis=1)
            ),
            "target_latent_norm": describe(
                np.linalg.norm(target_latent_array, axis=1)
            ),
            "predicted_mean_dimension_std": finite_float(
                predicted_latent_array.std(axis=0).mean()
            ),
            "target_mean_dimension_std": finite_float(
                target_latent_array.std(axis=0).mean()
            ),
            "reward": error_metrics(reward_predictions, reward_targets),
            "termination": error_metrics(terminal_predictions, terminal_targets),
            "predicted_policy_entropy": describe(next_policy_entropies),
            "next_policy_cross_entropy": describe(next_policy_cross_entropies),
            "next_value": error_metrics(next_value_predictions, next_value_targets),
        }

    return results


class FixedBatchBuffer:
    def __init__(self, batch):
        self.batch = batch

    def sample(self, _batch_size, unroll_steps, action_dim, device=None):
        del unroll_steps, action_dim
        if device is None:
            return self.batch
        return tuple(tensor.to(device) for tensor in self.batch)


def make_fixed_overfit_batch(
    trajectories,
    batch_size,
    unroll_steps,
    action_dim,
    device,
    terminal_balanced=True,
):
    if terminal_balanced:
        selected = []
        for trajectory in trajectories:
            selected.append((trajectory, max(0, len(trajectory) - unroll_steps)))
            selected.append((trajectory, 0))
    else:
        selected = [
            (trajectory, start)
            for trajectory in trajectories
            for start in range(len(trajectory))
        ]
        random.Random(24680).shuffle(selected)
    while len(selected) < batch_size:
        trajectory = trajectories[len(selected) % len(trajectories)]
        maximum_start = max(0, len(trajectory) - unroll_steps)
        start = (len(selected) * 7) % (maximum_start + 1)
        selected.append((trajectory, start))
    selected = selected[:batch_size]

    observations = []
    next_observations = []
    actions = np.zeros((batch_size, unroll_steps, action_dim), dtype=np.float32)
    policies = np.zeros((batch_size, unroll_steps, action_dim), dtype=np.float32)
    values = np.zeros((batch_size, unroll_steps, 1), dtype=np.float32)
    rewards = np.zeros((batch_size, unroll_steps, 1), dtype=np.float32)
    terminals = np.zeros((batch_size, unroll_steps, 1), dtype=np.float32)
    masks = np.zeros((batch_size, unroll_steps, 1), dtype=np.float32)

    for batch_index, (trajectory, start) in enumerate(selected):
        observations.append(trajectory[start]["observation"])
        padded_next = trajectory[start]["next_observation"]
        sequence_next = []
        for step in range(unroll_steps):
            index = start + step
            if index < len(trajectory):
                transition = trajectory[index]
                padded_next = transition["next_observation"]
                actions[batch_index, step, transition["action"]] = 1.0
                policies[batch_index, step] = transition["policy"]
                values[batch_index, step, 0] = transition["value"]
                rewards[batch_index, step, 0] = transition["reward"]
                terminals[batch_index, step, 0] = transition["terminated"]
                masks[batch_index, step, 0] = 1.0
            sequence_next.append(padded_next)
        next_observations.append(sequence_next)

    arrays = (
        observations,
        actions,
        next_observations,
        policies,
        values,
        rewards,
        terminals,
        masks,
    )
    return tuple(
        torch.as_tensor(np.asarray(array), dtype=torch.float32, device=device)
        for array in arrays
    )


def calculate_unrolled_losses(
    networks,
    batch,
    unroll_steps,
):
    representation, dynamics, policy, value = networks
    (
        observations,
        actions,
        next_observations,
        target_policies,
        target_values,
        target_rewards,
        target_terminals,
        masks,
    ) = batch
    batch_size = observations.shape[0]
    latents = representation(observations)
    with torch.no_grad():
        target_next_latents = representation(
            next_observations.flatten(0, 1)
        ).view(batch_size, unroll_steps, -1)

    losses = {
        name: torch.zeros((), device=observations.device)
        for name in ("policy", "value", "reward", "termination", "consistency")
    }
    valid_steps = masks.sum().clamp_min(1.0)
    for step in range(unroll_steps):
        mask = masks[:, step]
        predicted_policy = policy(latents)
        predicted_value = value(latents)
        (
            predicted_next_latents,
            predicted_rewards,
            predicted_terminal_logits,
        ) = dynamics(latents, actions[:, step])
        losses["policy"] += (
            -(
                target_policies[:, step]
                * torch.log(predicted_policy + 1e-8)
            ).sum(dim=1, keepdim=True)
            * mask
        ).sum()
        losses["value"] += (
            torch.nn.functional.mse_loss(
                predicted_value,
                target_values[:, step],
                reduction="none",
            )
            * mask
        ).sum()
        losses["reward"] += (
            torch.nn.functional.mse_loss(
                predicted_rewards,
                target_rewards[:, step],
                reduction="none",
            )
            * mask
        ).sum()
        losses["termination"] += (
            torch.nn.functional.binary_cross_entropy_with_logits(
                predicted_terminal_logits,
                target_terminals[:, step],
                reduction="none",
                pos_weight=torch.as_tensor(25.0, device=observations.device),
            )
            * mask
        ).sum()
        losses["consistency"] += (
            torch.nn.functional.smooth_l1_loss(
                predicted_next_latents,
                target_next_latents[:, step],
                reduction="none",
            ).mean(dim=1, keepdim=True)
            * mask
        ).sum()
        latents = predicted_next_latents

    return {name: loss / valid_steps for name, loss in losses.items()}


def loss_gradient_diagnostics(
    trajectories,
    representation,
    dynamics,
    policy,
    value,
    batch_size,
    unroll_steps,
    action_dim,
    consistency_weight,
    device,
):
    networks = [
        copy.deepcopy(network).to(device).train()
        for network in (representation, dynamics, policy, value)
    ]
    names = ("representation", "dynamics", "policy", "value")
    parameter_groups = [list(network.parameters()) for network in networks]
    parameters = [parameter for group in parameter_groups for parameter in group]

    def analyze_batch(terminal_balanced, selected_batch_size):
        batch = make_fixed_overfit_batch(
            trajectories,
            selected_batch_size,
            unroll_steps,
            action_dim,
            device,
            terminal_balanced=terminal_balanced,
        )
        losses = calculate_unrolled_losses(networks, batch, unroll_steps)
        weighted_losses = {
            **losses,
            "consistency": consistency_weight * losses["consistency"],
        }
        gradient_norms = {}
        for loss_name, loss in weighted_losses.items():
            gradients = torch.autograd.grad(
                loss,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            offset = 0
            per_network = {}
            total_squared = 0.0
            for network_name, group in zip(names, parameter_groups):
                group_gradients = gradients[offset:offset + len(group)]
                offset += len(group)
                squared = sum(
                    float(torch.square(gradient).sum())
                    for gradient in group_gradients
                    if gradient is not None
                )
                per_network[network_name] = math.sqrt(squared)
                total_squared += squared
            per_network["all"] = math.sqrt(total_squared)
            gradient_norms[loss_name] = per_network
        masks = batch[-1]
        terminals = batch[-2]
        terminal_count = int((terminals * masks).sum().item())
        return {
            "batch_size": selected_batch_size,
            "valid_transitions": int(masks.sum().item()),
            "terminal_transitions": terminal_count,
            "component_values": {
                name: float(loss.detach()) for name, loss in losses.items()
            },
            "weighted_gradient_l2_norms": gradient_norms,
        }

    return {
        "consistency_weight": consistency_weight,
        "representative_batch": analyze_batch(False, 64),
        "terminal_balanced_batch": analyze_batch(True, batch_size),
    }


def isolated_consistency_overfit_diagnostic(
    trajectories,
    representation,
    dynamics,
    steps,
    batch_size,
    unroll_steps,
    action_dim,
    device,
):
    representation = copy.deepcopy(representation).to(device).eval()
    dynamics = copy.deepcopy(dynamics).to(device).train()
    for parameter in representation.parameters():
        parameter.requires_grad_(False)
    batch = make_fixed_overfit_batch(
        trajectories,
        batch_size,
        unroll_steps,
        action_dim,
        device,
    )
    observations, actions, next_observations, _, _, _, _, masks = batch
    with torch.no_grad():
        initial_latents = representation(observations)
        target_latents = representation(next_observations.flatten(0, 1)).view(
            batch_size,
            unroll_steps,
            -1,
        )
    optimizer = torch.optim.Adam(dynamics.parameters(), lr=1e-3)
    snapshots = {}

    for step_index in range(1, steps + 1):
        optimizer.zero_grad()
        latents = initial_latents
        total = torch.zeros((), device=device)
        valid_steps = masks.sum().clamp_min(1.0)
        for rollout_step in range(unroll_steps):
            latents, _, _ = dynamics(latents, actions[:, rollout_step])
            total += (
                torch.nn.functional.smooth_l1_loss(
                    latents,
                    target_latents[:, rollout_step],
                    reduction="none",
                ).mean(dim=1, keepdim=True)
                * masks[:, rollout_step]
            ).sum()
        loss = total / valid_steps
        loss.backward()
        optimizer.step()
        if step_index in {1, 10, 50, 100, steps}:
            snapshots[str(step_index)] = float(loss.detach())

    return {
        "steps": steps,
        "snapshots": snapshots,
        "final_to_initial_ratio": finite_float(
            snapshots[str(steps)] / snapshots["1"]
        ),
    }


def fixed_batch_overfit_diagnostic(
    trajectories,
    representation,
    dynamics,
    policy,
    value,
    steps,
    batch_size,
    unroll_steps,
    action_dim,
    consistency_weight,
    device,
):
    networks = [
        copy.deepcopy(network).to(device).train()
        for network in (representation, dynamics, policy, value)
    ]
    batch = make_fixed_overfit_batch(
        trajectories,
        batch_size,
        unroll_steps,
        action_dim,
        device,
    )
    fixed_buffer = FixedBatchBuffer(batch)
    optimizer = torch.optim.Adam(
        [parameter for network in networks for parameter in network.parameters()],
        lr=1e-3,
    )
    snapshots = {}
    initial_parameters = [
        [parameter.detach().clone() for parameter in network.parameters()]
        for network in networks
    ]

    for step in range(1, steps + 1):
        losses = train_latent_step(
            *networks,
            fixed_buffer,
            optimizer,
            batch_size,
            unroll_steps,
            action_dim,
            consistency_weight,
            device,
        )
        if step in {1, 10, 50, 100, steps}:
            snapshots[str(step)] = losses

    parameter_changes = {}
    for name, network, initial in zip(
        ("representation", "dynamics", "policy", "value"),
        networks,
        initial_parameters,
    ):
        squared_change = sum(
            float(torch.square(parameter.detach() - old).sum())
            for parameter, old in zip(network.parameters(), initial)
        )
        parameter_changes[name] = math.sqrt(squared_change)

    first = snapshots["1"]
    final = snapshots[str(steps)]
    ratios = {
        name: finite_float(final[name] / first[name]) if first[name] else None
        for name in first
    }
    return {
        "steps": steps,
        "batch_size": batch_size,
        "unroll_steps": unroll_steps,
        "snapshots": snapshots,
        "final_to_initial_ratio": ratios,
        "parameter_l2_change": parameter_changes,
    }


def print_report(report):
    print("\n=== Structural checks ===")
    print(json.dumps(report["toy_mcts"], indent=2))
    print("\n=== Evaluation ===")
    print(json.dumps(report["evaluation"], indent=2))
    print("\n=== Representation ===")
    print(json.dumps(report["representation"], indent=2))
    print("\n=== Policy and MCTS ===")
    print(json.dumps(report["policy"], indent=2))
    print("\n=== Value calibration ===")
    print(json.dumps(report["value"], indent=2))
    print("\n=== Reward prediction ===")
    print(json.dumps(report["reward"], indent=2))
    print("\n=== Termination prediction ===")
    print(json.dumps(report["termination"], indent=2))
    print("\n=== One-step latent prediction ===")
    print(json.dumps(report["one_step_latent"], indent=2))
    print("\n=== Dynamics action sensitivity ===")
    print(json.dumps(report["action_sensitivity"], indent=2))
    print("\n=== Multi-step rollout ===")
    print(json.dumps(report["multistep"], indent=2))
    print("\n=== Fixed-batch overfit ===")
    print(json.dumps(report["fixed_batch_overfit"], indent=2))
    print("\n=== Per-loss gradient norms ===")
    print(json.dumps(report["loss_gradients"], indent=2))
    print("\n=== Isolated consistency overfit ===")
    print(json.dumps(report["isolated_consistency_overfit"], indent=2))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose a latent MuZero checkpoint without modifying it."
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/legacy_muzero/latent_muzero_cartpole.pt",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout-starts", type=int, default=512)
    parser.add_argument("--overfit-steps", type=int, default=150)
    parser.add_argument("--overfit-batch-size", type=int, default=16)
    parser.add_argument("--consistency-weight", type=float, default=0.25)
    parser.add_argument(
        "--output",
        default="artifacts/legacy_muzero/diagnostics/latent_muzero_diagnostics.json",
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
    terminal_penalty = checkpoint.get("terminal_penalty", -25.0)
    trajectories = collect_diagnostic_episodes(
        checkpoint,
        representation,
        dynamics,
        policy,
        value,
        episodes=args.episodes,
        simulations=args.simulations,
        seed=args.seed,
        terminal_penalty=terminal_penalty,
        device=device,
    )
    transitions = [item for trajectory in trajectories for item in trajectory]
    episode_rewards = [
        sum(item["real_reward"] for item in trajectory)
        for trajectory in trajectories
    ]

    report = {
        "checkpoint": {
            "path": str(Path(args.checkpoint).resolve()),
            "training_version": checkpoint.get("training_version"),
            "saved_episodes": len(checkpoint.get("history", {}).get("rewards", [])),
            "best_episode": checkpoint.get("history", {}).get(
                "best_evaluation_episode"
            ),
            "best_reward": checkpoint.get("history", {}).get(
                "best_evaluation_reward"
            ),
            "value_target_mode": checkpoint.get("value_target_mode"),
            "terminal_penalty": terminal_penalty,
            "unroll_steps": checkpoint.get("unroll_steps", 1),
            "device": str(device),
        },
        "toy_mcts": run_toy_mcts_checks(),
        "evaluation": {
            **describe(episode_rewards),
            "rewards": episode_rewards,
            "simulations": args.simulations,
            "seed_start": args.seed + 10_000,
        },
        "representation": representation_diagnostics(transitions),
        "policy": policy_diagnostics(transitions),
        "value": value_diagnostics(transitions),
        "reward": reward_diagnostics(transitions),
        "termination": termination_diagnostics(
            transitions,
            threshold=checkpoint.get("termination_threshold", 0.5),
        ),
        "one_step_latent": one_step_latent_diagnostics(transitions),
        "action_sensitivity": action_sensitivity_diagnostics(
            transitions,
            dynamics,
            policy,
            value,
            device,
        ),
        "multistep": multistep_rollout_diagnostics(
            trajectories,
            dynamics,
            policy,
            value,
            horizons=checkpoint.get("unroll_steps", 5),
            max_starts=args.rollout_starts,
            device=device,
        ),
        "fixed_batch_overfit": fixed_batch_overfit_diagnostic(
            trajectories,
            representation,
            dynamics,
            policy,
            value,
            steps=args.overfit_steps,
            batch_size=args.overfit_batch_size,
            unroll_steps=checkpoint.get("unroll_steps", 5),
            action_dim=checkpoint["action_dim"],
            consistency_weight=args.consistency_weight,
            device=device,
        ),
        "loss_gradients": loss_gradient_diagnostics(
            trajectories,
            representation,
            dynamics,
            policy,
            value,
            batch_size=args.overfit_batch_size,
            unroll_steps=checkpoint.get("unroll_steps", 5),
            action_dim=checkpoint["action_dim"],
            consistency_weight=args.consistency_weight,
            device=device,
        ),
        "isolated_consistency_overfit": isolated_consistency_overfit_diagnostic(
            trajectories,
            representation,
            dynamics,
            steps=100,
            batch_size=args.overfit_batch_size,
            unroll_steps=checkpoint.get("unroll_steps", 5),
            action_dim=checkpoint["action_dim"],
            device=device,
        ),
    }
    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_report(report)
    print("\nsaved diagnostic report:", output_path.resolve())


if __name__ == "__main__":
    main()
