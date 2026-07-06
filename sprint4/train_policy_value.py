import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from buffers import LatentReplayBuffer
from image_observation import FrameStack, ScreenshotConfig, make_image_env
from mcts import ModelBasedMCTS, select_action, visit_count_policy
from models import DynamicsModel, ImageRepresentationNetwork, PolicyNetwork, ValueNetwork
from utils import ensure_dir


LATENT_TRAINING_VERSION = 10
COMPATIBLE_LATENT_TRAINING_VERSIONS = {8, 9, LATENT_TRAINING_VERSION}


def parse_simulations(value):
    if isinstance(value, int):
        target = initial = value
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        target, initial = value
    else:
        parts = [part.strip() for part in str(value).split(",")]
        if len(parts) == 1:
            target = initial = int(parts[0])
        elif len(parts) == 2:
            target, initial = map(int, parts)
        else:
            raise argparse.ArgumentTypeError(
                "simulations must be one integer or TARGET,INITIAL"
            )

    if initial <= 0 or target <= 0:
        raise argparse.ArgumentTypeError("simulation counts must be positive")
    if initial > target:
        raise argparse.ArgumentTypeError(
            "initial simulations cannot exceed target simulations"
        )
    return target, initial


def latent_terminal(_latent):
    # A latent state has no reliable analytic terminal test. The planned
    # vector-state pipeline supplies CartPole's real state boundary instead.
    return False


def discounted_return_scale(discount, max_steps):
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if np.isclose(discount, 1.0):
        return float(max_steps)
    return float((1.0 - discount ** max_steps) / (1.0 - discount))


def shaped_environment_reward(real_reward, terminated, terminal_penalty):
    if terminated:
        return float(terminal_penalty)
    return float(real_reward)


def training_temperature(episode, anneal_episodes, minimum=0.1, maximum=1.0):
    if anneal_episodes <= 1:
        return float(minimum)
    progress = min(max(episode, 0) / (anneal_episodes - 1), 1.0)
    return float(maximum + progress * (minimum - maximum))


def performance_gated_temperature(
    episode,
    anneal_episodes,
    minimum,
    hold_temperature,
    unlock_episode=None,
    unlock_temperature=None,
):
    if unlock_episode is None:
        return training_temperature(
            episode,
            anneal_episodes,
            minimum=hold_temperature,
        )

    return training_temperature(
        episode - unlock_episode,
        anneal_episodes,
        minimum=minimum,
        maximum=unlock_temperature,
    )


def make_frame_stack(stack_size, image_size):
    return FrameStack(
        stack_size=stack_size,
        observation_shape=(1, image_size, image_size),
    )


def make_latent_mcts(
    dynamics_model,
    policy_network,
    value_network,
    action_dim,
    simulations,
    discount,
    max_steps,
):
    return ModelBasedMCTS(
        dynamics_model=dynamics_model,
        policy_network=policy_network,
        value_network=value_network,
        action_dim=action_dim,
        terminal_fn=latent_terminal,
        simulations=simulations,
        discount=discount,
        exploration_c=1.4,
        reward_scale=1.0 / discounted_return_scale(discount, max_steps),
    )


def train_latent_step(
    representation_network,
    dynamics_model,
    policy_network,
    value_network,
    buffer,
    optimizer,
    batch_size,
    unroll_steps,
    action_dim,
    consistency_weight,
    device,
):
    (
        observations,
        actions,
        next_observations,
        target_policies,
        target_values,
        target_rewards,
        _target_terminals,
        masks,
    ) = buffer.sample(
        batch_size,
        unroll_steps=unroll_steps,
        action_dim=action_dim,
        device=device,
    )

    optimizer.zero_grad()
    latents = representation_network(observations)
    with torch.no_grad():
        flat_next_observations = next_observations.flatten(0, 1)
        target_next_latents = representation_network(flat_next_observations).view(
            batch_size,
            unroll_steps,
            -1,
        )

    policy_loss = torch.zeros((), device=device)
    value_loss = torch.zeros((), device=device)
    reward_loss = torch.zeros((), device=device)
    consistency_loss = torch.zeros((), device=device)
    valid_steps = masks.sum().clamp_min(1.0)

    for step in range(unroll_steps):
        mask = masks[:, step]
        policy_predictions = policy_network(latents)
        value_predictions = value_network(latents)
        predicted_next_latents, predicted_rewards = dynamics_model(
            latents,
            actions[:, step],
        )

        policy_loss += (
            -(
                target_policies[:, step]
                * torch.log(policy_predictions + 1e-8)
            ).sum(dim=1, keepdim=True)
            * mask
        ).sum()
        value_loss += (
            F.mse_loss(
                value_predictions,
                target_values[:, step],
                reduction="none",
            )
            * mask
        ).sum()
        reward_loss += (
            F.mse_loss(
                predicted_rewards,
                target_rewards[:, step],
                reduction="none",
            )
            * mask
        ).sum()
        consistency_loss += (
            F.smooth_l1_loss(
                predicted_next_latents,
                target_next_latents[:, step],
                reduction="none",
            ).mean(dim=1, keepdim=True)
            * mask
        ).sum()
        latents = predicted_next_latents

    policy_loss /= valid_steps
    value_loss /= valid_steps
    reward_loss /= valid_steps
    consistency_loss /= valid_steps
    total_loss = (
        policy_loss
        + value_loss
        + reward_loss
        + consistency_weight * consistency_loss
    )
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [
            *representation_network.parameters(),
            *dynamics_model.parameters(),
            *policy_network.parameters(),
            *value_network.parameters(),
        ],
        max_norm=5.0,
    )
    optimizer.step()

    return {
        "total_loss": total_loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "reward_loss": reward_loss.item(),
        "consistency_loss": consistency_loss.item(),
    }


def bootstrapped_value_targets(
    trajectory,
    discount,
    bootstrap_steps,
    max_steps,
):
    targets = []
    return_scale = discounted_return_scale(discount, max_steps)
    for start in range(len(trajectory)):
        value = 0.0
        discount_power = 1.0
        stop = min(start + bootstrap_steps, len(trajectory))

        for index in range(start, stop):
            value += (
                discount_power
                * trajectory[index].get(
                    "value_reward",
                    trajectory[index]["reward"],
                )
                / return_scale
            )
            discount_power *= discount

        if stop < len(trajectory):
            value += discount_power * trajectory[stop]["search_value"]

        targets.append(float(np.clip(value, -1.0, 1.0)))
    return targets


def full_episode_value_targets(trajectory, discount, max_steps):
    return_scale = discounted_return_scale(discount, max_steps)
    targets = [0.0] * len(trajectory)
    discounted_return = 0.0

    for index in range(len(trajectory) - 1, -1, -1):
        discounted_return = (
            trajectory[index].get(
                "value_reward",
                trajectory[index]["reward"],
            ) / return_scale
            + discount * discounted_return
        )
        targets[index] = float(np.clip(discounted_return, -1.0, 1.0))

    return targets


def evaluate_latent_agent(
    env_id,
    representation_network,
    mcts,
    episodes,
    max_steps,
    image_size,
    stack_size,
    seeds,
):
    if episodes <= 0:
        raise ValueError("Evaluation episodes must be positive.")

    screenshot_config = ScreenshotConfig(width=image_size, height=image_size)
    env = make_image_env(env_id, screenshot_config=screenshot_config)
    env.action_space.seed(seeds[0])
    rewards = []
    random_state = random.getstate()
    numpy_state = np.random.get_state()

    try:
        for evaluation_index in range(episodes):
            evaluation_seed = seeds[evaluation_index]
            random.seed(evaluation_seed)
            np.random.seed(evaluation_seed)
            observation, _ = env.reset(seed=evaluation_seed)
            frame_stack = make_frame_stack(stack_size, image_size)
            stacked_observation = frame_stack.reset(observation)
            done = False
            total_reward = 0.0
            steps = 0

            while not done and steps < max_steps:
                root_latent = representation_network.encode(stacked_observation)
                root = mcts.search(root_latent)
                action, _ = select_action(root, temperature=0.0)
                observation, reward, terminated, truncated, _ = env.step(action)
                stacked_observation = frame_stack.append(observation)
                done = terminated or truncated
                total_reward += reward
                steps += 1

            rewards.append(total_reward)
    finally:
        env.close()
        random.setstate(random_state)
        np.random.set_state(numpy_state)

    return float(np.mean(rewards)), rewards


def train_latent_muzero(
    env_id="CartPole-v1",
    episodes=300,
    max_steps=500,
    simulations=50,
    batch_size=64,
    updates_per_episode=10,
    buffer_capacity=20000,
    hidden_dim=128,
    learning_rate=3e-4,
    warmup_episodes=20,
    exploration_episodes=180,
    minimum_temperature=0.25,
    temperature_hold=0.5,
    temperature_unlock_reward_threshold=20.0,
    value_discount=0.997,
    bootstrap_steps=10,
    value_target_mode="full-episode",
    terminal_penalty=-25.0,
    image_size=32,
    stack_size=5,
    latent_dim=32,
    unroll_steps=5,
    consistency_weight=0.25,
    save_best_checkpoint=True,
    checkpoint_path="checkpoints/latent_muzero_cartpole.pt",
    checkpoint_eval_interval=20,
    checkpoint_eval_episodes=20,
    simulation_upgrade_reward_threshold=20.0,
    simulation_upgrade_window=20,
    resume_path=None,
    seed=0,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if image_size != 32:
        raise ValueError("The configured representation architecture requires 32x32 images.")
    if stack_size != 5:
        raise ValueError("The configured representation architecture requires five frames.")
    if unroll_steps <= 0:
        raise ValueError("unroll_steps must be positive.")
    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    if value_target_mode not in {"full-episode", "n-step"}:
        raise ValueError("value_target_mode must be 'full-episode' or 'n-step'.")
    if not 0.0 <= minimum_temperature <= temperature_hold <= 1.0:
        raise ValueError(
            "Temperatures must satisfy 0 <= minimum <= hold <= 1."
        )

    target_simulations, initial_simulations = parse_simulations(simulations)
    screenshot_config = ScreenshotConfig(width=image_size, height=image_size)
    env = make_image_env(env_id, screenshot_config=screenshot_config)
    env.action_space.seed(seed)
    action_dim = env.action_space.n

    resume_checkpoint = None
    if resume_path is not None:
        if not Path(resume_path).exists():
            raise FileNotFoundError(f"Cannot resume missing checkpoint: {resume_path}")
        (
            representation_network,
            dynamics_model,
            policy_network,
            value_network,
            resume_checkpoint,
        ) = load_latent_checkpoint(resume_path, device=device)
        if resume_checkpoint["image_size"] != image_size:
            raise ValueError("Resume checkpoint uses a different image size.")
        if resume_checkpoint["stack_size"] != stack_size:
            raise ValueError("Resume checkpoint uses a different frame-stack size.")
        if resume_checkpoint["action_dim"] != action_dim:
            raise ValueError("Resume checkpoint uses a different action space.")
    else:
        representation_network = ImageRepresentationNetwork(
            input_channels=stack_size,
            latent_dim=latent_dim,
        ).to(device)
        dynamics_model = DynamicsModel(
            latent_dim,
            action_dim,
            hidden_dim=hidden_dim,
        ).to(device)
        policy_network = PolicyNetwork(
            latent_dim,
            action_dim,
            hidden_dim=hidden_dim,
        ).to(device)
        value_network = ValueNetwork(
            latent_dim,
            hidden_dim=hidden_dim,
        ).to(device)

    optimizer = torch.optim.Adam(
        [
            *representation_network.parameters(),
            *dynamics_model.parameters(),
            *policy_network.parameters(),
            *value_network.parameters(),
        ],
        lr=learning_rate,
    )
    if (
        resume_checkpoint is not None
        and resume_checkpoint.get("training_version") == LATENT_TRAINING_VERSION
        and resume_checkpoint.get("optimizer_state_dict")
    ):
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        print("restored Adam optimizer state")
    elif resume_checkpoint is not None:
        print("older checkpoint uses a different optimizer state; continuing fresh")

    buffer = LatentReplayBuffer(buffer_capacity)
    current_simulations = initial_simulations
    fresh_history = {
        "rewards": [],
        "temperatures": [],
        "total_losses": [],
        "policy_losses": [],
        "value_losses": [],
        "reward_losses": [],
        "consistency_losses": [],
        "evaluations": [],
        "terminated_episodes": [],
        "temperature_unlocks": [],
        "resume_events": [],
        "simulation_history": [
            {"episode": 1, "simulations": current_simulations, "reason": "initial"}
        ],
    }
    if resume_checkpoint is None:
        history = fresh_history
    else:
        history = resume_checkpoint.get("history", {})
        for key, default in fresh_history.items():
            history.setdefault(key, default)
        if history["simulation_history"]:
            saved_simulations = history["simulation_history"][-1]["simulations"]
            current_simulations = min(target_simulations, saved_simulations)

    start_episode = len(history["rewards"])
    final_episode = start_episode + episodes
    if resume_checkpoint is not None:
        history["resume_events"].append(
            {
                "episode": start_episode,
                "update": len(history["total_losses"]),
                "checkpoint": str(Path(resume_path)),
                "previous_terminal_penalty": resume_checkpoint.get(
                    "terminal_penalty"
                ),
                "terminal_penalty": terminal_penalty,
            }
        )
    best_evaluation_reward = history.get("best_evaluation_reward", float("-inf"))
    if history["temperature_unlocks"]:
        latest_unlock = history["temperature_unlocks"][-1]
        temperature_unlock_episode = latest_unlock["episode"]
        temperature_at_unlock = latest_unlock["temperature"]
    else:
        temperature_unlock_episode = None
        temperature_at_unlock = None

    mcts = make_latent_mcts(
        dynamics_model,
        policy_network,
        value_network,
        action_dim,
        current_simulations,
        value_discount,
        max_steps,
    )
    if resume_checkpoint is not None:
        print(
            f"resuming checkpoint at episode {start_episode}; "
            f"training {episodes} additional episodes with "
            f"terminal_penalty={terminal_penalty:g}"
        )

    for episode in range(start_episode, final_episode):
        observation, _ = env.reset(seed=seed + episode)
        frame_stack = make_frame_stack(stack_size, image_size)
        stacked_observation = frame_stack.reset(observation)
        trajectory = []
        done = False
        total_reward = 0.0
        steps = 0
        episode_terminated = False
        temperature = performance_gated_temperature(
            episode,
            exploration_episodes,
            minimum_temperature,
            temperature_hold,
            unlock_episode=temperature_unlock_episode,
            unlock_temperature=temperature_at_unlock,
        )

        while not done and steps < max_steps:
            if resume_checkpoint is None and episode < warmup_episodes:
                action = env.action_space.sample()
                policy_target = np.ones(action_dim, dtype=np.float32) / action_dim
                search_value = 0.0
            else:
                root_latent = representation_network.encode(stacked_observation)
                root = mcts.search(root_latent)
                action, _ = select_action(root, temperature=temperature)
                policy_target = visit_count_policy(root, temperature=1.0)
                search_value = root.mean_value

            next_observation, reward, terminated, truncated, _ = env.step(action)
            value_reward = shaped_environment_reward(
                reward,
                terminated=terminated,
                terminal_penalty=terminal_penalty,
            )
            next_stacked_observation = frame_stack.append(next_observation)
            trajectory.append(
                {
                    "observation": stacked_observation.copy(),
                    "action": action,
                    "next_observation": next_stacked_observation.copy(),
                    "policy": np.asarray(policy_target, dtype=np.float32),
                    "search_value": float(search_value),
                    "reward": float(reward),
                    "value_reward": value_reward,
                    "terminated": bool(terminated),
                }
            )
            stacked_observation = next_stacked_observation
            done = terminated or truncated
            episode_terminated = bool(terminated)
            total_reward += reward
            steps += 1

        if value_target_mode == "full-episode":
            value_targets = full_episode_value_targets(
                trajectory,
                discount=value_discount,
                max_steps=max_steps,
            )
        else:
            value_targets = bootstrapped_value_targets(
                trajectory,
                discount=value_discount,
                bootstrap_steps=bootstrap_steps,
                max_steps=max_steps,
            )
        for transition, value_target in zip(trajectory, value_targets):
            transition["value"] = value_target
        buffer.add_episode(trajectory)

        history["rewards"].append(total_reward)
        history["temperatures"].append(temperature)
        history["terminated_episodes"].append(episode_terminated)

        if len(buffer) >= batch_size:
            for _ in range(updates_per_episode):
                losses = train_latent_step(
                    representation_network,
                    dynamics_model,
                    policy_network,
                    value_network,
                    buffer,
                    optimizer,
                    batch_size,
                    unroll_steps,
                    action_dim,
                    consistency_weight,
                    device,
                )
                history["total_losses"].append(losses["total_loss"])
                history["policy_losses"].append(losses["policy_loss"])
                history["value_losses"].append(losses["value_loss"])
                history["reward_losses"].append(losses["reward_loss"])
                history["consistency_losses"].append(losses["consistency_loss"])

        recent = float(np.mean(history["rewards"][-10:]))
        print(
            "latent MuZero "
            f"episode={episode + 1:4d}/{final_episode} "
            f"reward={total_reward:6.1f} "
            f"avg10={recent:6.1f} "
            f"temp={temperature:.2f} "
            f"sims={current_simulations:3d} "
            f"buffer={len(buffer):5d}"
        )

        if (
            current_simulations < target_simulations
            and len(history["rewards"]) >= simulation_upgrade_window
        ):
            upgrade_average = float(
                np.mean(history["rewards"][-simulation_upgrade_window:])
            )
            if upgrade_average >= simulation_upgrade_reward_threshold:
                current_simulations = target_simulations
                mcts.simulations = current_simulations
                history["simulation_history"].append(
                    {
                        "episode": episode + 1,
                        "simulations": current_simulations,
                        "reason": f"average reward reached {upgrade_average:.1f}",
                    }
                )

        should_evaluate = (
            save_best_checkpoint or temperature_unlock_episode is None
        ) and (
            (episode + 1) % checkpoint_eval_interval == 0
            or episode + 1 == final_episode
        )
        if should_evaluate:
            evaluation_seeds = [
                seed + 10_000 + index
                for index in range(checkpoint_eval_episodes)
            ]
            mean_reward, evaluation_rewards = evaluate_latent_agent(
                env_id,
                representation_network,
                mcts,
                checkpoint_eval_episodes,
                max_steps,
                image_size,
                stack_size,
                evaluation_seeds,
            )
            reward_std = (
                float(np.std(evaluation_rewards, ddof=1))
                if len(evaluation_rewards) > 1
                else 0.0
            )
            evaluation = {
                "episode": episode + 1,
                "mean_reward": mean_reward,
                "reward_std": reward_std,
                "rewards": evaluation_rewards,
                "simulations": current_simulations,
            }
            history["evaluations"].append(evaluation)
            print(
                "latent checkpoint evaluation "
                f"episode={episode + 1:4d} "
                f"mean_reward={mean_reward:6.1f} std={reward_std:5.1f}"
            )

            if (
                temperature_unlock_episode is None
                and mean_reward >= temperature_unlock_reward_threshold
            ):
                temperature_unlock_episode = episode + 1
                temperature_at_unlock = temperature
                history["temperature_unlocks"].append(
                    {
                        "episode": episode + 1,
                        "evaluation_reward": mean_reward,
                        "temperature": temperature,
                    }
                )
                print(
                    "temperature gate unlocked "
                    f"at evaluation reward={mean_reward:.1f}; "
                    f"annealing below {temperature_hold:.2f} is now enabled"
                )

            if save_best_checkpoint and mean_reward > best_evaluation_reward:
                best_evaluation_reward = mean_reward
                history["best_evaluation_reward"] = mean_reward
                history["best_evaluation_episode"] = episode + 1
                save_latent_checkpoint(
                    checkpoint_path,
                    representation_network,
                    dynamics_model,
                    policy_network,
                    value_network,
                    history,
                    image_size=image_size,
                    stack_size=stack_size,
                    value_discount=value_discount,
                    max_steps=max_steps,
                    terminal_penalty=terminal_penalty,
                    value_target_mode=value_target_mode,
                    unroll_steps=unroll_steps,
                    minimum_temperature=minimum_temperature,
                    temperature_hold=temperature_hold,
                    temperature_unlock_reward_threshold=(
                        temperature_unlock_reward_threshold
                    ),
                    optimizer_state_dict=optimizer.state_dict(),
                    consistency_weight=consistency_weight,
                )
                print(
                    f"saved best latent checkpoint: {checkpoint_path} "
                    f"| mean_reward={mean_reward:.1f}"
                )

    if not save_best_checkpoint:
        save_latent_checkpoint(
            checkpoint_path,
            representation_network,
            dynamics_model,
            policy_network,
            value_network,
            history,
            image_size=image_size,
            stack_size=stack_size,
            value_discount=value_discount,
            max_steps=max_steps,
            terminal_penalty=terminal_penalty,
            value_target_mode=value_target_mode,
            unroll_steps=unroll_steps,
            minimum_temperature=minimum_temperature,
            temperature_hold=temperature_hold,
            temperature_unlock_reward_threshold=(
                temperature_unlock_reward_threshold
            ),
            optimizer_state_dict=optimizer.state_dict(),
            consistency_weight=consistency_weight,
        )
        print("saved final latent checkpoint:", checkpoint_path)

    env.close()
    return (
        representation_network,
        dynamics_model,
        policy_network,
        value_network,
        history,
    )


def save_latent_checkpoint(
    path,
    representation_network,
    dynamics_model,
    policy_network,
    value_network,
    history,
    image_size=32,
    stack_size=5,
    value_discount=0.997,
    max_steps=500,
    terminal_penalty=-25.0,
    value_target_mode="full-episode",
    unroll_steps=5,
    minimum_temperature=0.25,
    temperature_hold=0.5,
    temperature_unlock_reward_threshold=20.0,
    optimizer_state_dict=None,
    consistency_weight=0.25,
):
    ensure_dir(Path(path).parent)
    torch.save(
        {
            "training_version": LATENT_TRAINING_VERSION,
            "representation_state_dict": representation_network.state_dict(),
            "dynamics_state_dict": dynamics_model.state_dict(),
            "policy_state_dict": policy_network.state_dict(),
            "value_state_dict": value_network.state_dict(),
            "image_size": image_size,
            "stack_size": stack_size,
            "latent_dim": representation_network.latent_dim,
            "action_dim": policy_network.action_dim,
            "hidden_dim": policy_network.net[0].out_features,
            "value_discount": value_discount,
            "max_steps": max_steps,
            "terminal_penalty": terminal_penalty,
            "value_target_mode": value_target_mode,
            "unroll_steps": unroll_steps,
            "minimum_temperature": minimum_temperature,
            "temperature_hold": temperature_hold,
            "temperature_unlock_reward_threshold": (
                temperature_unlock_reward_threshold
            ),
            "consistency_weight": consistency_weight,
            "history": history,
            "optimizer_state_dict": optimizer_state_dict,
        },
        path,
    )


def load_latent_checkpoint(path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    checkpoint_version = checkpoint.get("training_version")
    if checkpoint_version not in COMPATIBLE_LATENT_TRAINING_VERSIONS:
        raise ValueError(
            "Checkpoint is not compatible with the recurrent latent MuZero pipeline."
        )

    representation_network = ImageRepresentationNetwork(
        input_channels=checkpoint["stack_size"],
        latent_dim=checkpoint["latent_dim"],
    ).to(device)
    dynamics_model = DynamicsModel(
        checkpoint["latent_dim"],
        checkpoint["action_dim"],
        checkpoint["hidden_dim"],
    ).to(device)
    policy_network = PolicyNetwork(
        checkpoint["latent_dim"],
        checkpoint["action_dim"],
        checkpoint["hidden_dim"],
    ).to(device)
    value_network = ValueNetwork(
        checkpoint["latent_dim"],
        checkpoint["hidden_dim"],
    ).to(device)

    representation_network.load_state_dict(checkpoint["representation_state_dict"])
    dynamics_state_dict = {
        key: value
        for key, value in checkpoint["dynamics_state_dict"].items()
        if not key.startswith("terminal_head.")
    }
    dynamics_model.load_state_dict(dynamics_state_dict)
    policy_network.load_state_dict(checkpoint["policy_state_dict"])
    value_network.load_state_dict(checkpoint["value_state_dict"])
    for network in (
        representation_network,
        dynamics_model,
        policy_network,
        value_network,
    ):
        network.eval()
    return (
        representation_network,
        dynamics_model,
        policy_network,
        value_network,
        checkpoint,
    )


def save_loss_plot(history, output_path):
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    losses = {
        "policy": history.get("policy_losses", []),
        "value": history.get("value_losses", []),
        "reward": history.get("reward_losses", []),
        "latent consistency": history.get("consistency_losses", []),
    }
    if not any(losses.values()):
        print("no latent losses recorded, skipping loss plot")
        return

    _, ax = plt.subplots(figsize=(10, 5))
    for name, values in losses.items():
        if values:
            ax.plot(values, label=name)
    for index, event in enumerate(history.get("resume_events", [])):
        ax.axvline(
            event["update"],
            color="black",
            linestyle="--",
            alpha=0.7,
            label="resumed checkpoint" if index == 0 else None,
        )
    ax.set_xlabel("network update")
    ax.set_ylabel("loss")
    ax.set_title("Latent MuZero Training Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()
    print("saved latent loss plot:", output_path)


def save_training_progress_plot(history, output_path):
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    rewards = np.asarray(history.get("rewards", []), dtype=np.float32)
    evaluations = history.get("evaluations", [])
    total_losses = history.get("total_losses", [])
    resume_events = history.get("resume_events", [])
    fig, (reward_ax, loss_ax) = plt.subplots(2, 1, figsize=(10, 8))

    if rewards.size:
        episodes = np.arange(1, rewards.size + 1)
        reward_ax.plot(episodes, rewards, alpha=0.35, label="episode reward")
        if rewards.size >= 10:
            average = np.convolve(rewards, np.ones(10) / 10, mode="valid")
            reward_ax.plot(np.arange(10, rewards.size + 1), average, label="avg10")
    if evaluations:
        evaluation_stds = [
            item.get(
                "reward_std",
                float(np.std(item.get("rewards", []), ddof=1))
                if len(item.get("rewards", [])) > 1
                else 0.0,
            )
            for item in evaluations
        ]
        reward_ax.errorbar(
            [item["episode"] for item in evaluations],
            [item["mean_reward"] for item in evaluations],
            yerr=evaluation_stds,
            marker="o",
            capsize=3,
            label="checkpoint evaluation mean +/- std",
        )
        running_best = float("-inf")
        best_evaluations = []
        for evaluation in evaluations:
            if evaluation["mean_reward"] > running_best:
                running_best = evaluation["mean_reward"]
                best_evaluations.append(evaluation)
        reward_ax.scatter(
            [item["episode"] for item in best_evaluations],
            [item["mean_reward"] for item in best_evaluations],
            marker="*",
            s=110,
            color="gold",
            edgecolor="black",
            linewidth=0.5,
            zorder=4,
            label="saved best checkpoint",
        )
    for index, event in enumerate(resume_events):
        reward_ax.axvline(
            event["episode"],
            color="black",
            linestyle="--",
            alpha=0.7,
            label="resumed checkpoint" if index == 0 else None,
        )
    reward_ax.set_title("Latent MuZero Reward")
    reward_ax.set_xlabel("episode")
    reward_ax.set_ylabel("reward")
    reward_ax.grid(alpha=0.25)
    reward_ax.legend()

    if total_losses:
        loss_ax.plot(total_losses, label="total loss")
    for index, event in enumerate(resume_events):
        loss_ax.axvline(
            event["update"],
            color="black",
            linestyle="--",
            alpha=0.7,
            label="resumed checkpoint" if index == 0 else None,
        )
    if total_losses or resume_events:
        loss_ax.legend()
    loss_ax.set_title("Joint Network Loss")
    loss_ax.set_xlabel("network update")
    loss_ax.set_ylabel("loss")
    loss_ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print("saved latent training progress plot:", output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train five-frame image-latent MuZero on CartPole."
    )
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--save-path", default="checkpoints/latent_muzero_cartpole.pt")
    parser.add_argument("--loss-plot-path", default="artifacts/latent_muzero_loss.png")
    parser.add_argument(
        "--training-plot-path",
        default="artifacts/latent_muzero_training_progress.png",
    )
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--simulations",
        type=parse_simulations,
        default=parse_simulations("50"),
        metavar="TARGET[,INITIAL]",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--updates-per-episode", type=int, default=10)
    parser.add_argument("--buffer-capacity", type=int, default=20000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-episodes", type=int, default=20)
    parser.add_argument("--exploration-episodes", type=int, default=180)
    parser.add_argument("--minimum-temperature", type=float, default=0.25)
    parser.add_argument("--temperature-hold", type=float, default=0.5)
    parser.add_argument(
        "--temperature-unlock-reward-threshold",
        type=float,
        default=20.0,
    )
    parser.add_argument("--value-discount", type=float, default=0.997)
    parser.add_argument("--bootstrap-steps", type=int, default=10)
    parser.add_argument(
        "--value-target-mode",
        choices=("full-episode", "n-step"),
        default="full-episode",
    )
    parser.add_argument("--terminal-penalty", type=float, default=-25.0)
    parser.add_argument("--consistency-weight", type=float, default=0.25)
    parser.add_argument("--unroll-steps", type=int, default=5)
    parser.add_argument("--save-best-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-eval-interval", type=int, default=20)
    parser.add_argument("--checkpoint-eval-episodes", type=int, default=20)
    parser.add_argument(
        "--simulation-upgrade-reward-threshold",
        type=float,
        default=20.0,
    )
    parser.add_argument("--simulation-upgrade-window", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from --save-path; --episodes counts additional episodes.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    networks = train_latent_muzero(
        env_id=args.env,
        episodes=args.episodes,
        max_steps=args.max_steps,
        simulations=args.simulations,
        batch_size=args.batch_size,
        updates_per_episode=args.updates_per_episode,
        buffer_capacity=args.buffer_capacity,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        warmup_episodes=args.warmup_episodes,
        exploration_episodes=args.exploration_episodes,
        minimum_temperature=args.minimum_temperature,
        temperature_hold=args.temperature_hold,
        temperature_unlock_reward_threshold=(
            args.temperature_unlock_reward_threshold
        ),
        value_discount=args.value_discount,
        bootstrap_steps=args.bootstrap_steps,
        value_target_mode=args.value_target_mode,
        terminal_penalty=args.terminal_penalty,
        consistency_weight=args.consistency_weight,
        unroll_steps=args.unroll_steps,
        save_best_checkpoint=args.save_best_checkpoint,
        checkpoint_path=args.save_path,
        checkpoint_eval_interval=args.checkpoint_eval_interval,
        checkpoint_eval_episodes=args.checkpoint_eval_episodes,
        simulation_upgrade_reward_threshold=(
            args.simulation_upgrade_reward_threshold
        ),
        simulation_upgrade_window=args.simulation_upgrade_window,
        resume_path=args.save_path if args.resume else None,
        seed=args.seed,
        device=device,
    )
    _, _, _, _, history = networks
    if args.save_best_checkpoint:
        print("kept best latent checkpoint:", args.save_path)
    save_loss_plot(history, args.loss_plot_path)
    save_training_progress_plot(history, args.training_plot_path)


if __name__ == "__main__":
    main()
