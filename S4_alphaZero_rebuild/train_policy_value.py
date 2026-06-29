import argparse
import random
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from buffers import PolicyValueReplayBuffer
from mcts import ModelBasedMCTS, select_action
from models import PolicyNetwork, ValueNetwork
from train_dynamics import load_dynamics
from utils import cartpole_terminal, ensure_dir, normalized_return


POLICY_VALUE_TRAINING_VERSION = 2


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


def train_policy_value_step(policy_network, value_network, buffer, optimizer_p, optimizer_v, batch_size, device):
    states, target_policies, target_values = buffer.sample(batch_size, device=device)

    optimizer_p.zero_grad()
    policy_preds = policy_network(states)
    policy_loss = -(target_policies * torch.log(policy_preds + 1e-8)).sum(dim=1).mean()
    policy_loss.backward()
    optimizer_p.step()

    optimizer_v.zero_grad()
    value_preds = value_network(states)
    value_loss = F.mse_loss(value_preds, target_values)
    value_loss.backward()
    optimizer_v.step()

    return policy_loss.item(), value_loss.item()


def evaluate_policy_value(env_id, mcts, episodes, max_steps, seeds):
    if episodes <= 0:
        raise ValueError("Evaluation episodes must be positive.")

    env = gym.make(env_id)
    rewards = []
    random_state = random.getstate()
    numpy_state = np.random.get_state()

    try:
        for evaluation_index in range(episodes):
            evaluation_seed = seeds[evaluation_index]
            random.seed(evaluation_seed)
            np.random.seed(evaluation_seed)
            state, _ = env.reset(seed=evaluation_seed)
            done = False
            total_reward = 0.0
            steps = 0

            while not done and steps < max_steps:
                root = mcts.search(state)
                action, _ = select_action(root, temperature=0.0)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1

            rewards.append(total_reward)
    finally:
        env.close()
        random.setstate(random_state)
        np.random.set_state(numpy_state)

    return float(np.mean(rewards)), rewards


def train_policy_value(
    dynamics_model,
    env_id="CartPole-v1",
    episodes=80,
    max_steps=500,
    simulations=50,
    batch_size=32,
    updates_per_episode=2,
    buffer_capacity=10000,
    hidden_dim=128,
    learning_rate=1e-3,
    exploration_episodes=10,
    value_discount=0.99,
    save_best_checkpoint=False,
    checkpoint_path="checkpoints/policy_value_cartpole.pt",
    checkpoint_eval_interval=5,
    checkpoint_eval_episodes=5,
    simulation_upgrade_reward_threshold=50.0,
    simulation_upgrade_window=10,
    seed=0,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_simulations, initial_simulations = parse_simulations(simulations)
    if initial_simulations != target_simulations and not save_best_checkpoint:
        raise ValueError(
            "A simulation schedule requires --save-best-checkpoint."
        )
    if save_best_checkpoint:
        if checkpoint_eval_interval <= 0:
            raise ValueError("Checkpoint evaluation interval must be positive.")
        if checkpoint_eval_episodes <= 0:
            raise ValueError("Checkpoint evaluation episodes must be positive.")
        if not checkpoint_path:
            raise ValueError("A checkpoint path is required when best saving is enabled.")
        if simulation_upgrade_window <= 0:
            raise ValueError("Simulation upgrade window must be positive.")

    env = gym.make(env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy_network = PolicyNetwork(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
    value_network = ValueNetwork(state_dim, hidden_dim=hidden_dim).to(device)

    optimizer_p = torch.optim.Adam(policy_network.parameters(), lr=learning_rate)
    optimizer_v = torch.optim.Adam(value_network.parameters(), lr=learning_rate)
    buffer = PolicyValueReplayBuffer(buffer_capacity)

    reward_history = []
    policy_losses = []
    value_losses = []
    evaluation_history = []
    best_evaluation_reward = float("-inf")
    current_simulations = initial_simulations
    simulation_history = [
        {
            "episode": 1,
            "simulations": current_simulations,
            "reason": "initial",
        }
    ]

    mcts = ModelBasedMCTS(
        dynamics_model=dynamics_model,
        policy_network=policy_network,
        value_network=value_network,
        action_dim=action_dim,
        terminal_fn=cartpole_terminal,
        simulations=current_simulations,
        discount=value_discount,
        reward_scale=1.0 / max_steps,
    )
    if current_simulations != target_simulations:
        print(
            "simulation schedule "
            f"initial={current_simulations} "
            f"target={target_simulations} "
            f"upgrade_avg{simulation_upgrade_window}="
            f"{simulation_upgrade_reward_threshold:.1f}"
        )

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        trajectory = []
        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < max_steps:
            root = mcts.search(state)
            temperature = 1.0 if episode < exploration_episodes else 0.0
            action, policy_target = select_action(root, temperature=temperature)
            current_state = state
            next_state, reward, terminated, truncated, _ = env.step(action)
            trajectory.append((current_state, policy_target, reward))
            done = terminated or truncated
            state = next_state
            total_reward += reward
            steps += 1

        discounted_returns = []
        running_return = 0.0
        for _, _, reward in reversed(trajectory):
            running_return = reward + value_discount * running_return
            discounted_returns.append(running_return)
        discounted_returns.reverse()

        for (obs, policy_target, _), return_value in zip(trajectory, discounted_returns):
            buffer.add(
                obs,
                policy_target,
                normalized_return(return_value, max_reward=max_steps),
            )

        reward_history.append(total_reward)

        if len(buffer) >= batch_size:
            for _ in range(updates_per_episode):
                p_loss, v_loss = train_policy_value_step(
                    policy_network,
                    value_network,
                    buffer,
                    optimizer_p,
                    optimizer_v,
                    batch_size,
                    device,
                )
                policy_losses.append(p_loss)
                value_losses.append(v_loss)

        recent = np.mean(reward_history[-10:])
        print(
            "policy/value "
            f"episode={episode + 1:4d}/{episodes} "
            f"reward={total_reward:6.1f} "
            f"avg10={recent:6.1f} "
            f"buffer={len(buffer):5d}"
        )

        if (
            current_simulations < target_simulations
            and len(reward_history) >= simulation_upgrade_window
        ):
            upgrade_average = float(
                np.mean(reward_history[-simulation_upgrade_window:])
            )
            if upgrade_average >= simulation_upgrade_reward_threshold:
                current_simulations = target_simulations
                mcts.simulations = current_simulations
                simulation_history.append(
                    {
                        "episode": episode + 1,
                        "simulations": current_simulations,
                        "reason": (
                            f"average reward reached {upgrade_average:.1f}"
                        ),
                    }
                )
                print(
                    "increased MCTS simulations "
                    f"to={current_simulations} "
                    f"after episode={episode + 1} "
                    f"avg{simulation_upgrade_window}={upgrade_average:.1f}"
                )

        should_evaluate = (
            save_best_checkpoint
            and (
                (episode + 1) % checkpoint_eval_interval == 0
                or episode + 1 == episodes
            )
        )
        if should_evaluate:
            evaluation_seeds = [
                seed + 10_000 + evaluation_index
                for evaluation_index in range(checkpoint_eval_episodes)
            ]
            mean_reward, evaluation_rewards = evaluate_policy_value(
                env_id,
                mcts,
                checkpoint_eval_episodes,
                max_steps,
                evaluation_seeds,
            )
            evaluation_history.append(
                {
                    "episode": episode + 1,
                    "mean_reward": mean_reward,
                    "rewards": evaluation_rewards,
                    "simulations": current_simulations,
                }
            )
            print(
                "checkpoint evaluation "
                f"episode={episode + 1:4d} "
                f"mean_reward={mean_reward:6.1f}"
            )

            if mean_reward > best_evaluation_reward:
                best_evaluation_reward = mean_reward
                checkpoint_history = {
                    "rewards": reward_history.copy(),
                    "policy_losses": policy_losses.copy(),
                    "value_losses": value_losses.copy(),
                    "evaluations": evaluation_history.copy(),
                    "simulation_history": simulation_history.copy(),
                    "best_evaluation_reward": best_evaluation_reward,
                    "best_evaluation_episode": episode + 1,
                }
                save_policy_value(
                    checkpoint_path,
                    policy_network,
                    value_network,
                    checkpoint_history,
                )
                print(
                    "saved best policy/value: "
                    f"{checkpoint_path} | mean_reward={mean_reward:.1f}"
                )

    env.close()
    history = {
        "rewards": reward_history,
        "policy_losses": policy_losses,
        "value_losses": value_losses,
        "evaluations": evaluation_history,
        "simulation_history": simulation_history,
    }
    if evaluation_history:
        best_evaluation = max(
            evaluation_history,
            key=lambda evaluation: evaluation["mean_reward"],
        )
        history["best_evaluation_reward"] = best_evaluation["mean_reward"]
        history["best_evaluation_episode"] = best_evaluation["episode"]
        history["checkpoint_path"] = str(checkpoint_path)

    return policy_network, value_network, history


def save_policy_value(path, policy_network, value_network, history):
    ensure_dir(Path(path).parent)
    torch.save(
        {
            "policy_state_dict": policy_network.state_dict(),
            "value_state_dict": value_network.state_dict(),
            "state_dim": policy_network.input_dim,
            "action_dim": policy_network.action_dim,
            "hidden_dim": policy_network.net[0].out_features,
            "training_version": POLICY_VALUE_TRAINING_VERSION,
            "history": history,
        },
        path,
    )


def save_loss_plot(history, output_path):
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    if output_path.name == "policy_value_loss.png":
        for stale_path in output_path.parent.glob("policy_value_loss_*.png"):
            stale_path.unlink()

    policy_losses = history.get("policy_losses", [])
    value_losses = history.get("value_losses", [])

    if not policy_losses and not value_losses:
        print("no policy/value losses recorded, skipping loss plot")
        return

    plt.figure(figsize=(9, 5))
    if policy_losses:
        plt.plot(policy_losses, label="policy loss")
    if value_losses:
        plt.plot(value_losses, label="value loss")
    plt.xlabel("network update")
    plt.ylabel("loss")
    plt.title("Policy/Value Training Loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()
    print("saved policy/value loss plot:", output_path)


def save_training_progress_plot(history, output_path):
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    rewards = np.asarray(history.get("rewards", []), dtype=np.float32)
    policy_losses = history.get("policy_losses", [])
    value_losses = history.get("value_losses", [])
    evaluations = history.get("evaluations", [])

    if rewards.size == 0 and not policy_losses and not value_losses:
        print("no policy/value history recorded, skipping training progress plot")
        return

    fig, (reward_ax, loss_ax) = plt.subplots(2, 1, figsize=(10, 8))

    if rewards.size:
        episodes = np.arange(1, rewards.size + 1)
        reward_ax.plot(episodes, rewards, alpha=0.4, label="episode reward")
        if rewards.size >= 10:
            average = np.convolve(rewards, np.ones(10) / 10, mode="valid")
            reward_ax.plot(
                np.arange(10, rewards.size + 1),
                average,
                linewidth=2,
                label="10-episode average",
            )
    if evaluations:
        reward_ax.plot(
            [evaluation["episode"] for evaluation in evaluations],
            [evaluation["mean_reward"] for evaluation in evaluations],
            marker="o",
            linewidth=2,
            label="checkpoint evaluation",
        )
    reward_ax.set_xlabel("episode")
    reward_ax.set_ylabel("reward")
    reward_ax.set_title("Training Reward")
    reward_ax.grid(alpha=0.25)
    if rewards.size or evaluations:
        reward_ax.legend()

    if policy_losses:
        loss_ax.plot(policy_losses, label="policy loss")
    if value_losses:
        loss_ax.plot(value_losses, label="value loss")
    loss_ax.set_xlabel("network update")
    loss_ax.set_ylabel("loss")
    loss_ax.set_title("Policy/Value Training Loss")
    loss_ax.grid(alpha=0.25)
    if policy_losses or value_losses:
        loss_ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print("saved policy/value training progress plot:", output_path)


def load_policy_value(path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    training_version = checkpoint.get("training_version")
    if training_version != POLICY_VALUE_TRAINING_VERSION:
        raise ValueError(
            "Policy/value checkpoint was trained with an incompatible MCTS "
            "policy usage and must be retrained."
        )
    policy = PolicyNetwork(
        checkpoint["state_dim"],
        checkpoint["action_dim"],
        checkpoint["hidden_dim"],
    ).to(device)
    value = ValueNetwork(
        checkpoint["state_dim"],
        checkpoint["hidden_dim"],
    ).to(device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    value.load_state_dict(checkpoint["value_state_dict"])
    policy.eval()
    value.eval()
    return policy, value


def parse_args():
    parser = argparse.ArgumentParser(description="Train CartPole policy/value networks from model-based MCTS targets.")
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--dynamics-path", default="checkpoints/dynamics_cartpole.pt")
    parser.add_argument("--save-path", default="checkpoints/policy_value_cartpole.pt")
    parser.add_argument("--loss-plot-path", default="artifacts/policy_value_loss.png")
    parser.add_argument(
        "--training-plot-path",
        default="artifacts/policy_value_training_progress.png",
    )
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--simulations",
        type=parse_simulations,
        default=parse_simulations("50"),
        metavar="TARGET[,INITIAL]",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--updates-per-episode", type=int, default=2)
    parser.add_argument("--buffer-capacity", type=int, default=10000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--exploration-episodes", type=int, default=10)
    parser.add_argument("--value-discount", type=float, default=0.99)
    parser.add_argument("--save-best-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-eval-interval", type=int, default=5)
    parser.add_argument("--checkpoint-eval-episodes", type=int, default=5)
    parser.add_argument(
        "--simulation-upgrade-reward-threshold",
        type=float,
        default=50.0,
    )
    parser.add_argument("--simulation-upgrade-window", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dynamics_model = load_dynamics(args.dynamics_path, device=device)
    policy_network, value_network, history = train_policy_value(
        dynamics_model=dynamics_model,
        env_id=args.env,
        episodes=args.episodes,
        max_steps=args.max_steps,
        simulations=args.simulations,
        batch_size=args.batch_size,
        updates_per_episode=args.updates_per_episode,
        buffer_capacity=args.buffer_capacity,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        exploration_episodes=args.exploration_episodes,
        value_discount=args.value_discount,
        save_best_checkpoint=args.save_best_checkpoint,
        checkpoint_path=args.save_path,
        checkpoint_eval_interval=args.checkpoint_eval_interval,
        checkpoint_eval_episodes=args.checkpoint_eval_episodes,
        simulation_upgrade_reward_threshold=(
            args.simulation_upgrade_reward_threshold
        ),
        simulation_upgrade_window=args.simulation_upgrade_window,
        seed=args.seed,
        device=device,
    )
    if not args.save_best_checkpoint:
        save_policy_value(args.save_path, policy_network, value_network, history)
        print("saved final policy/value:", args.save_path)
    else:
        print("kept best policy/value checkpoint:", args.save_path)
    save_loss_plot(history, args.loss_plot_path)
    save_training_progress_plot(history, args.training_plot_path)
