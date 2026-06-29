import argparse
import random
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from image_observation import FrameStack, ScreenshotConfig, ScreenshotPreprocessor
from mcts import select_action, visit_count_policy
from train_policy_value import (
    load_latent_checkpoint,
    make_latent_mcts,
    parse_simulations,
    save_latent_checkpoint,
    save_loss_plot,
    save_training_progress_plot,
    train_latent_muzero,
)
from utils import ensure_dir


def display_font(size=18):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def compose_side_by_side_frame(
    rgb_frame,
    grayscale_observation,
    step,
    history_count,
    value,
    policy,
    visits,
    action,
    cumulative_reward,
):
    rgb_image = Image.fromarray(np.asarray(rgb_frame, dtype=np.uint8)).convert("RGB")
    width, height = rgb_image.size
    grayscale = np.asarray(grayscale_observation, dtype=np.float32).squeeze(0)
    grayscale = np.clip(grayscale * 255.0, 0, 255).astype(np.uint8)
    grayscale_image = Image.fromarray(grayscale, mode="L").convert("RGB")
    grayscale_image = grayscale_image.resize((width, height), Image.Resampling.NEAREST)

    header_height = 72
    canvas = Image.new("RGB", (width * 2, height + header_height), "#111418")
    canvas.paste(grayscale_image, (0, header_height))
    canvas.paste(rgb_image, (width, header_height))
    draw = ImageDraw.Draw(canvas)
    font = display_font(18)
    small_font = display_font(16)
    action_name = "LEFT" if action == 0 else "RIGHT"

    draw.text(
        (16, 10),
        f"GRAYSCALE 32x32 | frame history {history_count}/5",
        fill="white",
        font=font,
    )
    draw.text(
        (width + 16, 10),
        f"GYM RGB | step {step} | action {action_name} | reward {cumulative_reward:.0f}",
        fill="white",
        font=font,
    )
    metrics = (
        f"V(s) {value:+.3f}    "
        f"policy L/R {policy[0]:.2f}/{policy[1]:.2f}    "
        f"MCTS visits L/R {visits[0]:.2f}/{visits[1]:.2f}"
    )
    draw.text((16, 42), metrics, fill="#9fdbff", font=small_font)
    return np.asarray(canvas)


def save_playthrough_gif(frames, output_path, fps=20):
    if not frames:
        raise ValueError("Cannot save a GIF without frames.")
    if fps <= 0:
        raise ValueError("GIF FPS must be positive.")
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    imageio.mimsave(
        output_path,
        frames,
        format="GIF",
        duration=1000 / fps,
        loop=0,
    )


def collect_annotated_playthrough(
    env_id,
    representation_network,
    policy_network,
    value_network,
    mcts,
    seed,
    max_steps,
    image_size,
    stack_size,
):
    random.seed(seed)
    np.random.seed(seed)
    env = gym.make(env_id, render_mode="rgb_array")
    preprocessor = ScreenshotPreprocessor(
        ScreenshotConfig(width=image_size, height=image_size)
    )
    _, _ = env.reset(seed=seed)
    rgb_frame = env.render()
    grayscale_observation = preprocessor.process(rgb_frame)
    frame_stack = FrameStack(
        stack_size=stack_size,
        observation_shape=(1, image_size, image_size),
    )
    stacked_observation = frame_stack.reset(grayscale_observation)
    frames = []
    done = False
    total_reward = 0.0
    steps = 0

    while not done and steps < max_steps:
        latent = representation_network.encode(stacked_observation)
        policy = policy_network.action_probs(latent)
        value = value_network.value(latent)
        root = mcts.search(latent)
        visits = visit_count_policy(root, temperature=1.0)
        action, _ = select_action(root, temperature=0.0)
        frames.append(
            compose_side_by_side_frame(
                rgb_frame,
                grayscale_observation,
                step=steps,
                history_count=frame_stack.real_frame_count,
                value=value,
                policy=policy,
                visits=visits,
                action=action,
                cumulative_reward=total_reward,
            )
        )

        _, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
        rgb_frame = env.render()
        grayscale_observation = preprocessor.process(rgb_frame)
        stacked_observation = frame_stack.append(grayscale_observation)

    env.close()
    return frames, total_reward


def train_from_args(args, checkpoint_path, device):
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
        save_best_checkpoint=args.save_best_checkpoint,
        checkpoint_path=checkpoint_path,
        checkpoint_eval_interval=args.checkpoint_eval_interval,
        checkpoint_eval_episodes=args.checkpoint_eval_episodes,
        simulation_upgrade_reward_threshold=(
            args.simulation_upgrade_reward_threshold
        ),
        simulation_upgrade_window=args.simulation_upgrade_window,
        seed=args.seed,
        device=device,
    )
    representation, dynamics, policy, value, history = networks
    if args.save_best_checkpoint:
        representation, dynamics, policy, value, checkpoint = load_latent_checkpoint(
            checkpoint_path,
            device=device,
        )
        print("using best latent checkpoint:", checkpoint_path)
    else:
        save_latent_checkpoint(
            checkpoint_path,
            representation,
            dynamics,
            policy,
            value,
            history,
            value_discount=args.value_discount,
            max_steps=args.max_steps,
            terminal_penalty=args.terminal_penalty,
            value_target_mode=args.value_target_mode,
            minimum_temperature=args.minimum_temperature,
            temperature_hold=args.temperature_hold,
            temperature_unlock_reward_threshold=(
                args.temperature_unlock_reward_threshold
            ),
        )
        checkpoint = {
            "image_size": 32,
            "stack_size": 5,
            "value_discount": args.value_discount,
            "max_steps": args.max_steps,
            "terminal_penalty": args.terminal_penalty,
            "value_target_mode": args.value_target_mode,
            "minimum_temperature": args.minimum_temperature,
            "temperature_hold": args.temperature_hold,
            "temperature_unlock_reward_threshold": (
                args.temperature_unlock_reward_threshold
            ),
            "history": history,
        }
        print("saved final latent checkpoint:", checkpoint_path)

    save_loss_plot(history, Path(args.artifact_dir) / "latent_muzero_loss.png")
    save_training_progress_plot(
        history,
        Path(args.artifact_dir) / "latent_muzero_training_progress.png",
    )
    return representation, dynamics, policy, value, checkpoint


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ensure_dir(args.artifact_dir)
    ensure_dir(args.checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint_dir) / "latent_muzero_cartpole.pt"

    networks = None
    if checkpoint_path.exists() and not args.train:
        try:
            networks = load_latent_checkpoint(checkpoint_path, device=device)
        except ValueError as error:
            raise SystemExit(
                f"cannot load latent checkpoint: {error}\n"
                "Run with --train to create a new compatible checkpoint."
            ) from error

        checkpoint = networks[4]
        history = checkpoint.get("history", {})
        print("loaded latent checkpoint:", checkpoint_path.resolve())
        print(
            "checkpoint metadata "
            f"episodes={len(history.get('rewards', []))} "
            f"best_episode={history.get('best_evaluation_episode', 'n/a')} "
            f"best_reward={history.get('best_evaluation_reward', 'n/a')} "
            f"value_target={checkpoint.get('value_target_mode', 'n/a')}"
        )

    if networks is None:
        if args.train:
            print("explicit --train requested; training a new latent model")
        else:
            print("no latent checkpoint found; training a new latent model")
        networks = train_from_args(args, checkpoint_path, device)

    representation, dynamics, policy, value, checkpoint = networks
    image_size = checkpoint.get("image_size", 32)
    stack_size = checkpoint.get("stack_size", 5)
    search_discount = checkpoint.get("value_discount", args.value_discount)
    reward_horizon = checkpoint.get("max_steps", args.max_steps)
    eval_mcts = make_latent_mcts(
        dynamics,
        policy,
        value,
        action_dim=2,
        simulations=args.eval_simulations,
        discount=search_discount,
        max_steps=reward_horizon,
    )
    frames, reward = collect_annotated_playthrough(
        args.env,
        representation,
        policy,
        value,
        eval_mcts,
        seed=args.playthrough_seed,
        max_steps=args.max_steps,
        image_size=image_size,
        stack_size=stack_size,
    )
    gif_path = Path(args.artifact_dir) / "latent_cartpole_side_by_side.gif"
    save_playthrough_gif(frames, gif_path, fps=args.gif_fps)
    print(
        f"saved latent side-by-side GIF: {gif_path} "
        f"| seed={args.playthrough_seed} reward={reward:.1f} frames={len(frames)}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or run five-frame image-latent MuZero CartPole."
    )
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--updates-per-episode", type=int, default=10)
    parser.add_argument("--buffer-capacity", type=int, default=20000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-episodes", type=int, default=10)
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
    parser.add_argument("--terminal-penalty", type=float, default=-10.0)
    parser.add_argument("--consistency-weight", type=float, default=0.25)
    parser.add_argument(
        "--simulations",
        type=parse_simulations,
        default=parse_simulations("50"),
        metavar="TARGET[,INITIAL]",
    )
    parser.add_argument("--eval-simulations", type=int, default=90)
    parser.add_argument(
        "--save-best-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--checkpoint-eval-interval", type=int, default=20)
    parser.add_argument("--checkpoint-eval-episodes", type=int, default=20)
    parser.add_argument(
        "--simulation-upgrade-reward-threshold",
        type=float,
        default=20.0,
    )
    parser.add_argument("--simulation-upgrade-window", type=int, default=20)
    parser.add_argument("--playthrough-seed", type=int, default=234)
    parser.add_argument("--gif-fps", type=int, default=20)
    checkpoint_mode = parser.add_mutually_exclusive_group()
    checkpoint_mode.add_argument(
        "--train",
        action="store_true",
        help="Train a new model even when a checkpoint already exists.",
    )
    checkpoint_mode.add_argument(
        "--reuse-checkpoints",
        action="store_false",
        dest="train",
        help="Deprecated alias; existing checkpoints are reused by default.",
    )
    parser.set_defaults(train=False)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
