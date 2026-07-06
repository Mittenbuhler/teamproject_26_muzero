"""Independent supervised training for vector-state CartPole components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from .models import DynamicsModel, PolicyNetwork, ValueNetwork
from .search import ExactCartPoleDynamics


@dataclass
class StateDataset:
    states: np.ndarray
    actions: np.ndarray
    next_states: np.ndarray
    rewards: np.ndarray
    policy_actions: np.ndarray
    value_targets: np.ndarray
    episode_ids: np.ndarray

    def split(self, validation_fraction=0.2, seed=0):
        """Split whole episodes so neighboring transitions cannot leak."""
        rng = np.random.default_rng(seed)
        episodes = rng.permutation(np.unique(self.episode_ids))
        if len(episodes) < 2:
            indices = np.arange(len(self.states))
            split_at = min(len(indices) - 1, max(1, int(len(indices) * (1 - validation_fraction))))
            return self.subset(indices[:split_at]), self.subset(indices[split_at:])
        validation_count = min(
            len(episodes) - 1,
            max(1, int(round(len(episodes) * validation_fraction))),
        )
        validation_episodes = episodes[:validation_count]
        validation_mask = np.isin(self.episode_ids, validation_episodes)
        return (
            self.subset(np.flatnonzero(~validation_mask)),
            self.subset(np.flatnonzero(validation_mask)),
        )

    def subset(self, indices):
        return StateDataset(*(
            field[indices]
            for field in (
                self.states,
                self.actions,
                self.next_states,
                self.rewards,
                self.policy_actions,
                self.value_targets,
                self.episode_ids,
            )
        ))


def heuristic_action(state):
    """A transparent stabilizing controller used only to create labels."""
    x, x_dot, theta, theta_dot = map(float, state)
    score = theta + 0.45 * theta_dot + 0.02 * x + 0.08 * x_dot
    return int(score > 0.0)


def survival_target(state, exact_dynamics, horizon):
    current = np.asarray(state, dtype=np.float32)
    survived = 0
    for _ in range(horizon):
        current, _, done = exact_dynamics.step(current, heuristic_action(current))
        survived += 1
        if done:
            break
    return survived / horizon


def collect_state_dataset(
    env_id="CartPole-v1",
    samples=5000,
    value_horizon=30,
    expert_probability=0.5,
    seed=0,
    include_value_targets=True,
):
    if samples <= 0 or value_horizon <= 0:
        raise ValueError("samples and value_horizon must be positive")
    env = gym.make(env_id)
    env.action_space.seed(seed)
    rng = np.random.default_rng(seed)
    exact = ExactCartPoleDynamics.from_env(env)
    state, _ = env.reset(seed=seed)
    records = []
    episode = 0
    try:
        while len(records) < samples:
            label = heuristic_action(state)
            action = label if rng.random() < expert_probability else int(rng.integers(2))
            next_state, reward, terminated, truncated, _ = env.step(action)
            records.append(
                (
                    np.asarray(state, dtype=np.float32),
                    action,
                    np.asarray(next_state, dtype=np.float32),
                    float(reward),
                    label,
                    (
                        survival_target(state, exact, value_horizon)
                        if include_value_targets
                        else 0.0
                    ),
                    episode,
                )
            )
            state = next_state
            if terminated or truncated:
                episode += 1
                state, _ = env.reset(seed=seed + episode)
    finally:
        env.close()

    states, actions, next_states, rewards, labels, values, episode_ids = zip(*records)
    return StateDataset(
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        next_states=np.asarray(next_states, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32).reshape(-1, 1),
        policy_actions=np.asarray(labels, dtype=np.int64),
        value_targets=np.asarray(values, dtype=np.float32).reshape(-1, 1),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
    )


def _batches(size, batch_size, rng):
    indices = rng.permutation(size)
    for start in range(0, size, batch_size):
        yield indices[start : start + batch_size]


def _tensor(array, device, dtype=torch.float32):
    return torch.as_tensor(array, dtype=dtype, device=device)


def dynamics_curriculum(max_horizon):
    if max_horizon <= 0:
        raise ValueError("max_horizon must be positive")
    return tuple(dict.fromkeys(min(max_horizon, horizon) for horizon in (5, 10, 20, max_horizon)))


def curriculum_horizon(epoch, epochs, max_horizon):
    stages = dynamics_curriculum(max_horizon)
    if epochs <= 1:
        return stages[-1]
    stage_index = min(len(stages) - 1, int(epoch * len(stages) / epochs))
    return stages[stage_index]


def rollout_arrays(dataset, starts, horizon):
    """Build padded contiguous action/target sequences with terminal masks."""
    actions = np.zeros((len(starts), horizon), dtype=np.int64)
    targets = np.zeros((len(starts), horizon, 4), dtype=np.float32)
    masks = np.zeros((len(starts), horizon), dtype=np.float32)
    for row, start in enumerate(starts):
        episode_id = dataset.episode_ids[start]
        for step in range(horizon):
            index = start + step
            if index >= len(dataset.states) or dataset.episode_ids[index] != episode_id:
                break
            actions[row, step] = dataset.actions[index]
            targets[row, step] = dataset.next_states[index]
            masks[row, step] = 1.0
    return actions, targets, masks


def multistep_state_loss(model, initial_states, actions, targets, masks, horizon):
    """One-step anchor plus progressively downweighted horizon bands."""
    predicted_state = initial_states
    losses = []
    for step in range(horizon):
        action_one_hot = F.one_hot(actions[:, step], 2).float()
        predicted_state, _ = model(predicted_state, action_one_hot)
        losses.append(
            F.smooth_l1_loss(
                predicted_state,
                targets[:, step],
                reduction="none",
            ).mean(dim=1)
        )

    total = torch.zeros((), dtype=initial_states.dtype, device=initial_states.device)
    groups = ((0, 1, 1.0), (1, 5, 0.5), (5, 10, 0.25), (10, 20, 0.125), (20, horizon, 0.0625))
    for start, stop, weight in groups:
        stop = min(stop, horizon)
        if start >= stop:
            continue
        group_masks = masks[:, start:stop]
        group_losses = torch.stack(losses[start:stop], dim=1)
        total = total + weight * (group_losses * group_masks).sum() / group_masks.sum().clamp_min(1.0)
    return total


@torch.no_grad()
def evaluate_dynamics_rollouts(model, dataset, horizons, batch_size, device):
    horizons = tuple(sorted(set(int(horizon) for horizon in horizons)))
    maximum = max(horizons)
    absolute_error_sums = {horizon: 0.0 for horizon in horizons}
    element_counts = {horizon: 0 for horizon in horizons}
    model.eval()
    for starts in _batches(len(dataset.states), batch_size, np.random.default_rng(0)):
        actions_np, targets_np, masks_np = rollout_arrays(dataset, starts, maximum)
        predicted_state = _tensor(dataset.states[starts], device)
        actions = _tensor(actions_np, device, torch.long)
        targets = _tensor(targets_np, device)
        masks = _tensor(masks_np, device)
        for step in range(maximum):
            predicted_state, _ = model(
                predicted_state,
                F.one_hot(actions[:, step], 2).float(),
            )
            horizon = step + 1
            if horizon not in absolute_error_sums:
                continue
            mask = masks[:, step].unsqueeze(1)
            absolute_error_sums[horizon] += (
                (predicted_state - targets[:, step]).abs() * mask
            ).sum().item()
            element_counts[horizon] += int(mask.sum().item()) * predicted_state.shape[1]
    return {
        f"rollout_mae_h{horizon}": absolute_error_sums[horizon] / element_counts[horizon]
        for horizon in horizons
        if element_counts[horizon] > 0
    }


def train_dynamics(
    dataset,
    hidden_dim=128,
    epochs=30,
    batch_size=128,
    lr=1e-3,
    seed=0,
    device=None,
    rollout_horizon=30,
    loss_history=None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    torch.manual_seed(seed)
    train, validation = dataset.split(seed=seed)
    model = DynamicsModel(4, 2, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        model.train()
        horizon = curriculum_horizon(epoch, epochs, rollout_horizon)
        epoch_loss_sum = 0.0
        epoch_examples = 0
        for starts in _batches(len(train.states), batch_size, rng):
            actions_np, targets_np, masks_np = rollout_arrays(train, starts, horizon)
            optimizer.zero_grad()
            loss = multistep_state_loss(
                model,
                _tensor(train.states[starts], device),
                _tensor(actions_np, device, torch.long),
                _tensor(targets_np, device),
                _tensor(masks_np, device),
                horizon,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss_sum += loss.item() * len(starts)
            epoch_examples += len(starts)
        if loss_history is not None:
            loss_history.append(
                {
                    "epoch": epoch + 1,
                    "training_loss": epoch_loss_sum / epoch_examples,
                    "rollout_horizon": horizon,
                }
            )
    diagnostic_horizons = tuple(
        dict.fromkeys(min(rollout_horizon, horizon) for horizon in (1, 5, 10, 20, rollout_horizon))
    )
    metrics = evaluate_dynamics_rollouts(
        model,
        validation,
        diagnostic_horizons,
        batch_size,
        device,
    )
    metrics["state_mae"] = metrics["rollout_mae_h1"]
    metrics["training_rollout_horizon"] = float(rollout_horizon)
    return model, metrics


def train_policy(
    dataset,
    hidden_dim=128,
    epochs=20,
    batch_size=128,
    lr=1e-3,
    seed=0,
    device=None,
    loss_history=None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    train, validation = dataset.split(seed=seed)
    model = PolicyNetwork(4, 2, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_examples = 0
        for indices in _batches(len(train.states), batch_size, rng):
            optimizer.zero_grad()
            probabilities = model(_tensor(train.states[indices], device))
            labels = _tensor(train.policy_actions[indices], device, torch.long)
            loss = F.nll_loss(torch.log(probabilities + 1e-8), labels)
            loss.backward()
            optimizer.step()
            epoch_loss_sum += loss.item() * len(indices)
            epoch_examples += len(indices)
        if loss_history is not None:
            loss_history.append(
                {
                    "epoch": epoch + 1,
                    "training_loss": epoch_loss_sum / epoch_examples,
                }
            )
    model.eval()
    with torch.no_grad():
        predictions = model(_tensor(validation.states, device)).argmax(dim=1)
        labels = _tensor(validation.policy_actions, device, torch.long)
        accuracy = (predictions == labels).float().mean().item()
    return model, {"accuracy": accuracy}


def train_value(
    dataset,
    hidden_dim=128,
    epochs=20,
    batch_size=128,
    lr=1e-3,
    seed=0,
    device=None,
    loss_history=None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    train, validation = dataset.split(seed=seed)
    model = ValueNetwork(4, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_examples = 0
        for indices in _batches(len(train.states), batch_size, rng):
            optimizer.zero_grad()
            predictions = model(_tensor(train.states[indices], device))
            loss = F.mse_loss(predictions, _tensor(train.value_targets[indices], device))
            loss.backward()
            optimizer.step()
            epoch_loss_sum += loss.item() * len(indices)
            epoch_examples += len(indices)
        if loss_history is not None:
            loss_history.append(
                {
                    "epoch": epoch + 1,
                    "training_loss": epoch_loss_sum / epoch_examples,
                }
            )
    model.eval()
    with torch.no_grad():
        predictions = model(_tensor(validation.states, device))
        targets = _tensor(validation.value_targets, device)
        mae = F.l1_loss(predictions, targets).item()
    return model, {"mae": mae}


def save_component(path, component, model, metrics, value_horizon, loss_history=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "component": component,
        "state_dict": model.state_dict(),
        "hidden_dim": model.net[0].out_features,
        "metrics": metrics,
        "loss_history": list(loss_history or []),
        "value_horizon": value_horizon,
        "training_horizon": value_horizon,
    }
    if component == "dynamics":
        checkpoint["dynamics_training_version"] = 2
        checkpoint["dynamics_training"] = {
            "rollout_horizon": value_horizon,
            "curriculum": dynamics_curriculum(value_horizon),
            "loss": "weighted multi-step Smooth L1 with one-step anchor",
            "reward": "hardcoded +1 at planning time",
            "validation_split": "whole episodes",
        }
    elif component == "value":
        checkpoint["value_target_horizon"] = value_horizon
    torch.save(checkpoint, path)


def load_component(path, component, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint["component"] != component:
        raise ValueError(f"expected {component} checkpoint, got {checkpoint['component']}")
    hidden = checkpoint["hidden_dim"]
    constructors = {
        "dynamics": lambda: DynamicsModel(4, 2, hidden),
        "policy": lambda: PolicyNetwork(4, 2, hidden),
        "value": lambda: ValueNetwork(4, hidden),
    }
    model = constructors[component]().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
