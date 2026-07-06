"""Read-only fixed-seed ablations for MCTS termination and policy-only play."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from image_observation import ScreenshotConfig, make_image_env
from train_policy_value import (
    evaluate_latent_agent,
    load_latent_checkpoint,
    make_frame_stack,
    make_latent_mcts,
)
from utils import ensure_dir


def summarize(rewards):
    rewards = np.asarray(rewards, dtype=np.float64)
    return {
        "mean": float(rewards.mean()),
        "sample_std": float(rewards.std(ddof=1)) if len(rewards) > 1 else 0.0,
        "median": float(np.median(rewards)),
        "minimum": float(rewards.min()),
        "maximum": float(rewards.max()),
        "rewards": rewards.tolist(),
    }


def evaluate_policy_only(
    checkpoint,
    representation,
    policy,
    seeds,
):
    image_size = checkpoint["image_size"]
    stack_size = checkpoint["stack_size"]
    max_steps = checkpoint["max_steps"]
    env = make_image_env(
        "CartPole-v1",
        screenshot_config=ScreenshotConfig(width=image_size, height=image_size),
    )
    rewards = []
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            frame_stack = make_frame_stack(stack_size, image_size)
            stacked_observation = frame_stack.reset(observation)
            total_reward = 0.0
            for _ in range(max_steps):
                latent = representation.encode(stacked_observation)
                action = int(np.argmax(policy.action_probs(latent)))
                observation, reward, terminated, truncated, _ = env.step(action)
                stacked_observation = frame_stack.append(observation)
                total_reward += reward
                if terminated or truncated:
                    break
            rewards.append(total_reward)
    finally:
        env.close()
    return summarize(rewards)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/latent_muzero_terminal_v9.pt",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default="artifacts/search_ablations.json",
    )
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    representation, dynamics, policy, value, checkpoint = load_latent_checkpoint(
        args.checkpoint,
        device=device,
    )
    seeds = [args.seed + 10_000 + index for index in range(args.episodes)]
    results = {}

    for name, threshold, enabled in (
        ("learned_termination_0_50", 0.50, True),
        ("learned_termination_0_80", 0.80, True),
        ("learned_termination_0_95", 0.95, True),
        ("learned_termination_disabled", 0.50, False),
    ):
        mcts = make_latent_mcts(
            dynamics,
            policy,
            value,
            action_dim=checkpoint["action_dim"],
            simulations=args.simulations,
            discount=checkpoint["value_discount"],
            max_steps=checkpoint["max_steps"],
            terminal_penalty=checkpoint.get("terminal_penalty", -25.0),
            termination_threshold=threshold,
            use_learned_termination=enabled,
        )
        _, rewards = evaluate_latent_agent(
            "CartPole-v1",
            representation,
            mcts,
            args.episodes,
            checkpoint["max_steps"],
            checkpoint["image_size"],
            checkpoint["stack_size"],
            seeds,
        )
        results[name] = summarize(rewards)
        print(
            f"{name}: mean={results[name]['mean']:.1f} "
            f"std={results[name]['sample_std']:.1f}"
        )

    results["policy_argmax_without_mcts"] = evaluate_policy_only(
        checkpoint,
        representation,
        policy,
        seeds,
    )
    print(
        "policy_argmax_without_mcts: "
        f"mean={results['policy_argmax_without_mcts']['mean']:.1f} "
        f"std={results['policy_argmax_without_mcts']['sample_std']:.1f}"
    )
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_episode": len(checkpoint.get("history", {}).get("rewards", [])),
        "fixed_seeds": seeds,
        "simulations": args.simulations,
        "results": results,
    }
    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print("saved search ablations:", output_path.resolve())


if __name__ == "__main__":
    main()
