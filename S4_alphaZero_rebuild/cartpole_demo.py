import argparse
import random
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from image_observation import ScreenshotConfig, ScreenshotPreprocessor
from mcts import ModelBasedMCTS, select_action
from train_dynamics import train_or_load_dynamics
from train_policy_value import (
    load_policy_value,
    save_loss_plot,
    save_policy_value,
    save_training_progress_plot,
    parse_simulations,
    train_policy_value,
)
from utils import cartpole_terminal, ensure_dir


def save_observation_sheet(observations, output_path):
    if not observations:
        raise ValueError("Cannot save observation sheet without observations.")

    cols = 4
    rows = int(np.ceil(len(observations) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.asarray(axes).reshape(-1)
    for index, ax in enumerate(axes):
        ax.axis("off")
        if index < len(observations):
            ax.imshow(observations[index], cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_title(f"t={index}", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def save_playthrough_gif(frames, output_path, fps=20):
    if not frames:
        raise ValueError("Cannot save a GIF without frames.")
    if fps <= 0:
        raise ValueError("GIF FPS must be positive.")

    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    imageio.mimsave(
        output_path,
        [np.asarray(frame, dtype=np.uint8) for frame in frames],
        format="GIF",
        duration=1000 / fps,
        loop=0,
    )


def collect_learned_playthrough_observations(
    env_id,
    mcts,
    seed=123,
    max_steps=500,
    frames_to_show=12,
):
    random.seed(seed)
    np.random.seed(seed)
    env = gym.make(env_id, render_mode="rgb_array")
    preprocessor = ScreenshotPreprocessor(ScreenshotConfig(width=84, height=84, binary=False))
    state, _ = env.reset(seed=seed)
    observations = []
    frames = []
    done = False
    total_reward = 0.0
    steps = 0

    while not done and steps < max_steps:
        frame = env.render()
        frames.append(frame)
        if len(observations) < frames_to_show:
            observations.append(preprocessor.process(frame)[0])

        root = mcts.search(state)
        action, _ = select_action(root, temperature=0.0)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

    frames.append(env.render())
    env.close()
    return observations, frames, total_reward


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ensure_dir(args.artifact_dir)
    ensure_dir(args.checkpoint_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dyn_args = argparse.Namespace(
        env=args.env,
        collect_episodes=args.dynamics_collect_episodes,
        max_steps=args.max_steps,
        capacity=100000,
        batch_size=args.batch_size,
        train_steps=args.dynamics_train_steps,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        epsilon=1.0,
        seed=args.seed,
        log_every=max(1, args.dynamics_train_steps // 5),
        save_path=str(Path(args.checkpoint_dir) / "dynamics_cartpole.pt"),
        load_path=str(Path(args.checkpoint_dir) / "dynamics_cartpole.pt") if args.reuse_checkpoints else "",
    )
    dynamics_model = train_or_load_dynamics(dyn_args, device)

    pv_path = Path(args.checkpoint_dir) / "policy_value_cartpole.pt"
    policy_network = None
    value_network = None
    if args.reuse_checkpoints and pv_path.exists():
        print("loading policy/value:", pv_path)
        try:
            policy_network, value_network = load_policy_value(pv_path, device=device)
        except ValueError as error:
            print(f"cannot reuse policy/value checkpoint: {error}")

    if policy_network is None:
        policy_network, value_network, history = train_policy_value(
            dynamics_model=dynamics_model,
            env_id=args.env,
            episodes=args.policy_value_episodes,
            max_steps=args.max_steps,
            simulations=args.simulations,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            exploration_episodes=args.exploration_episodes,
            value_discount=args.value_discount,
            save_best_checkpoint=args.save_best_checkpoint,
            checkpoint_path=pv_path,
            checkpoint_eval_interval=args.checkpoint_eval_interval,
            checkpoint_eval_episodes=args.checkpoint_eval_episodes,
            simulation_upgrade_reward_threshold=(
                args.simulation_upgrade_reward_threshold
            ),
            simulation_upgrade_window=args.simulation_upgrade_window,
            seed=args.seed,
            device=device,
        )
        if args.save_best_checkpoint:
            policy_network, value_network = load_policy_value(pv_path, device=device)
            print("using best policy/value for evaluation:", pv_path)
        else:
            save_policy_value(pv_path, policy_network, value_network, history)
            print("saved final policy/value:", pv_path)
        save_loss_plot(history, Path(args.artifact_dir) / "policy_value_loss.png")
        save_training_progress_plot(
            history,
            Path(args.artifact_dir) / "policy_value_training_progress.png",
        )

    eval_mcts = ModelBasedMCTS(
        dynamics_model=dynamics_model,
        policy_network=policy_network,
        value_network=value_network,
        action_dim=2,
        terminal_fn=cartpole_terminal,
        simulations=args.eval_simulations,
        discount=args.value_discount,
        reward_scale=1.0 / args.max_steps,
    )
    sheet_path = Path(args.artifact_dir) / "downscaled_grayscale_observations.png"
    gif_path = Path(args.artifact_dir) / "cartpole_playthrough.gif"
    observations, frames, reward = collect_learned_playthrough_observations(
        args.env,
        eval_mcts,
        seed=args.playthrough_seed,
        max_steps=args.max_steps,
        frames_to_show=args.observation_frames,
    )
    save_observation_sheet(observations, sheet_path)
    save_playthrough_gif(frames, gif_path, fps=args.gif_fps)
    print(
        "saved learned-playthrough observation sheet: "
        f"{sheet_path} | seed={args.playthrough_seed} reward={reward:.1f}"
    )
    print(
        "saved learned-playthrough GIF: "
        f"{gif_path} | frames={len(frames)}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dynamics-collect-episodes", type=int, default=250)
    parser.add_argument("--dynamics-train-steps", type=int, default=1500)
    parser.add_argument("--policy-value-episodes", type=int, default=60)
    parser.add_argument("--exploration-episodes", type=int, default=10)
    parser.add_argument("--value-discount", type=float, default=0.99)
    parser.add_argument(
        "--simulations",
        type=parse_simulations,
        default=parse_simulations("50"),
        metavar="TARGET[,INITIAL]",
    )
    parser.add_argument("--eval-simulations", type=int, default=90)
    parser.add_argument("--save-best-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-eval-interval", type=int, default=5)
    parser.add_argument("--checkpoint-eval-episodes", type=int, default=5)
    parser.add_argument(
        "--simulation-upgrade-reward-threshold",
        type=float,
        default=50.0,
    )
    parser.add_argument("--simulation-upgrade-window", type=int, default=10)
    parser.add_argument("--playthrough-seed", type=int, default=234)
    parser.add_argument("--observation-frames", type=int, default=12)
    parser.add_argument("--gif-fps", type=int, default=20)
    parser.add_argument("--reuse-checkpoints", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
