"""Diagnostics and real-environment GIF recording for native-state MuZero."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .environment import make_state_env
from .buffers import LatentReplayBuffer
from .mcts import visit_count_policy
from .train import evaluate, load_latent_checkpoint, make_latent_mcts


def entropy(probabilities):
    probabilities = np.asarray(probabilities)
    return float(
        -(
            probabilities * np.log(probabilities + 1e-12)
        ).sum(-1).mean()
    )


@torch.no_grad()
def diagnose_batch(
    representation,
    dynamics,
    prediction,
    replay,
    unroll_steps,
    action_dim,
    device,
):
    observations, actions, target_policies, target_values, target_rewards, policy_masks, value_masks, _ = replay.sample(
        min(64, len(replay)), unroll_steps, action_dim, device
    )
    del target_policies, policy_masks
    state = representation(observations.float())
    report = {"depth": []}
    for depth in range(unroll_steps + 1):
        _, value = prediction(state)
        valid = value_masks[:, depth, 0].bool()
        predicted_value = value[:, 0][valid].cpu().numpy()
        target_value = target_values[:, depth, 0][valid].cpu().numpy()
        item = {
            "depth": depth,
            "latent_norm": float(state.flatten(1).norm(dim=1).mean()),
            "value_mae": (
                float(np.mean(np.abs(predicted_value - target_value)))
                if len(predicted_value)
                else None
            ),
            "value_correlation": (
                float(np.corrcoef(predicted_value, target_value)[0, 1])
                if len(predicted_value) > 1
                and np.std(predicted_value) > 0
                and np.std(target_value) > 0
                else None
            ),
        }
        if depth < unroll_steps:
            state, reward = dynamics(state, actions[:, depth])
            item["reward_mae"] = float(
                (reward - target_rewards[:, depth]).abs().mean()
            )
        report["depth"].append(item)
    return report


def policy_search_metrics(raw_policy, visit_policy):
    raw_policy = np.asarray(raw_policy)
    visit_policy = np.asarray(visit_policy)
    return {
        "raw_policy_entropy": entropy(raw_policy),
        "mcts_visit_entropy": entropy(visit_policy),
        "mean_l1_policy_change": float(
            np.abs(raw_policy - visit_policy).sum(-1).mean()
        ),
        "argmax_agreement": float(
            (raw_policy.argmax(-1) == visit_policy.argmax(-1)).mean()
        ),
    }


def _rgb_image(frame, output_width):
    """Convert an environment render to a consistently sized RGB image."""
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(
            "GIF recording requires env.render() to return an RGB or RGBA frame"
        )
    if np.issubdtype(array.dtype, np.floating):
        maximum = float(array.max()) if array.size else 0.0
        if maximum <= 1.0:
            array = array * 255.0
    image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)[..., :3], "RGB")
    if image.width != output_width:
        output_height = max(1, round(image.height * output_width / image.width))
        image = image.resize(
            (output_width, output_height),
            Image.Resampling.BILINEAR,
        )
    return image


def _annotate_frame(frame, output_width, lines, observation):
    """Show real play and the exact four numbers supplied to h."""
    image = _rgb_image(frame, output_width)
    panel = Image.new("RGB", (250, image.height), (20, 28, 36))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    state = np.asarray(observation, dtype=np.float32)
    if state.shape != (4,) or not np.isfinite(state).all():
        raise ValueError("GIF model input must contain four finite state variables")
    text = ["MODEL INPUT (native state)"]
    text.extend(f"{name}: {value:+.5f}" for name, value in zip(
        ("position", "velocity", "pole angle", "angular velocity"), state,
    ))
    text.extend(["", *lines])
    # The panel stays legible even if a small gameplay width was requested.
    height = max(220, image.height, 12 + 14 * len(text))
    if panel.height < height:
        panel = Image.new("RGB", (250, height), (20, 28, 36))
        draw = ImageDraw.Draw(panel)
    for index, line in enumerate(text):
        draw.text((8, 8 + index * 14), str(line), fill="white", font=font)
    composite = Image.new("RGB", (image.width + panel.width, height), (20, 28, 36))
    composite.paste(image, (0, 0))
    composite.paste(panel, (image.width, 0))
    return composite


def _format_policy(probabilities):
    return "[" + ", ".join(f"{value:.3f}" for value in probabilities) + "]"


@torch.no_grad()
def record_episode_gif(
    output_path,
    env_id,
    representation,
    prediction,
    mcts,
    mode="mcts",
    seed=0,
    max_steps=500,
    fps=20,
    output_width=480,
):
    """Record real RGB play; screenshots never enter the networks."""
    if mode not in {"raw", "mcts"}:
        raise ValueError("GIF mode must be 'raw' or 'mcts'")
    if int(max_steps) <= 0:
        raise ValueError("GIF max_steps must be positive")
    if int(fps) <= 0:
        raise ValueError("GIF fps must be positive")
    if int(output_width) <= 0:
        raise ValueError("GIF output_width must be positive")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    env = make_state_env(env_id, render_mode="rgb_array")
    numpy_random_state = np.random.get_state()
    np.random.seed(int(seed))
    frames = []
    score = 0.0
    steps = 0
    status = "max_steps"
    try:
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(int(seed))
        observation, _ = env.reset(seed=int(seed))

        for step in range(int(max_steps)):
            latent = representation.encode(observation)
            raw_policy, predicted_value = prediction.predict(latent)
            visit_policy = None
            if mode == "mcts":
                root = mcts.search(latent, add_exploration_noise=False)
                visit_policy = visit_count_policy(root, temperature=1.0)
                action = int(np.argmax(visit_policy))
                search_value = root.mean_value
            else:
                action = int(np.argmax(raw_policy))
                search_value = None

            lines = [
                f"mode={mode} seed={seed} step={step}",
                f"score={score:.1f} action={action}",
                f"raw policy={_format_policy(raw_policy)} value={predicted_value:.3f}",
            ]
            if visit_policy is not None:
                lines.append(
                    f"MCTS visits={_format_policy(visit_policy)} value={search_value:.3f}"
                )
            frames.append(
                _annotate_frame(
                    env.render(),
                    output_width,
                    lines,
                    observation=observation,
                )
            )

            observation, reward, terminated, truncated, _ = env.step(action)
            score += float(reward)
            steps = step + 1
            if terminated or truncated:
                status = "terminated" if terminated else "truncated"
                break

        frames.append(
            _annotate_frame(
                env.render(),
                output_width,
                [
                    f"mode={mode} seed={seed} steps={steps}",
                    f"final score={score:.1f} status={status}",
                ],
                observation=observation,
            )
        )
    finally:
        np.random.set_state(numpy_random_state)
        env.close()

    frame_duration_ms = max(1, round(1000 / int(fps)))
    frames[0].save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=False,
    )
    return {
        "path": str(output_path),
        "mode": mode,
        "seed": int(seed),
        "score": score,
        "steps": steps,
        "frames": len(frames),
        "gameplay_width": int(output_width),
        "gif_width": int(frames[0].width),
        "observation_mode": "cartpole_state_4d",
    }


def build_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--replay", help="optional replay file; otherwise use checkpoint replay")
    parser.add_argument("--output-json", type=Path, help="save evaluation and diagnostic results")
    parser.add_argument("--simulations", type=int, help="defaults to checkpoint search budget")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument(
        "--record-gif",
        type=Path,
        help="write one real-environment episode to this GIF path",
    )
    parser.add_argument("--gif-mode", choices=("mcts", "raw"), default="mcts")
    parser.add_argument(
        "--gif-seed",
        type=int,
        help="episode seed; defaults to --seed",
    )
    parser.add_argument(
        "--gif-max-steps",
        type=int,
        help="recording limit; defaults to the checkpoint training limit",
    )
    parser.add_argument("--gif-fps", type=int, default=20)
    parser.add_argument(
        "--gif-width",
        type=int,
        default=480,
        help="RGB gameplay-pane width; the model-input panel is added beside it",
    )
    return parser


def main():
    args = build_argument_parser().parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    representation, dynamics, prediction, metadata = load_latent_checkpoint(
        args.checkpoint,
        device,
    )
    action_dim = metadata["action_dim"]
    training_config = metadata.get("training_config", {})
    env_id = training_config.get("env_id", "CartPole-v1")
    max_steps = int(training_config.get("max_steps", 500))
    mcts = make_latent_mcts(
        dynamics,
        prediction,
        action_dim,
        simulations=(training_config.get("simulations", 25)
                     if args.simulations is None else args.simulations),
        discount=metadata.get("discount", 0.997),
        pb_c_base=training_config.get("pb_c_base", 19652),
        pb_c_init=training_config.get("pb_c_init", 1.25),
        root_dirichlet_alpha=training_config.get("root_dirichlet_alpha", 0.25),
        root_exploration_fraction=training_config.get(
            "root_exploration_fraction", 0.25
        ),
    )
    raw_policies = []
    visit_policies = []

    replay_data = (torch.load(args.replay, map_location="cpu", weights_only=False)
                   if args.replay else metadata.get("replay_state"))
    replay = None
    if isinstance(replay_data, LatentReplayBuffer):
        replay = replay_data
    elif replay_data is not None:
        replay = LatentReplayBuffer()
        replay.load_state_dict(replay_data.get("replay_state", replay_data))
    roots = [] if replay is None else [
        item["observation"] for ep in replay.episodes for item in ep
    ][:64]
    root_source = "checkpoint/replay transitions"
    if not roots:
        env = make_state_env(env_id)
        try:
            roots = [env.reset(seed=args.seed + i)[0] for i in range(args.episodes)]
        finally:
            env.close()
        root_source = "fresh environment resets"
    for observation in roots:
        latent = representation.encode(observation)
        raw_policies.append(prediction.predict(latent)[0])
        visit_policies.append(
            visit_count_policy(
                mcts.search(latent, add_exploration_noise=False),
                temperature=1.0,
            )
        )

    report = policy_search_metrics(raw_policies, visit_policies)
    report["policy_metric_roots"] = {"source": root_source, "count": len(roots)}
    if replay is not None and len(replay):
        report.update(
            diagnose_batch(
                representation,
                dynamics,
                prediction,
                replay,
                training_config.get("unroll_steps", 5),
                action_dim,
                device,
            )
        )

    seeds = [args.seed + episode for episode in range(args.episodes)]
    evaluations = {}
    for mode in ("random", "raw", "mcts"):
        mean, scores = evaluate(
            env_id, representation, prediction, mcts, args.episodes,
            max_steps, seeds, use_mcts=mode == "mcts", random_actions=mode == "random",
        )
        evaluations[mode] = {"mean": mean, "scores": scores}
    report["checkpoint"] = str(args.checkpoint)
    report["observation_mode"] = metadata["observation_mode"]
    report["evaluation_seeds"] = seeds
    report["simulations"] = mcts.simulations
    report["max_steps"] = max_steps
    report["evaluation"] = evaluations

    if args.record_gif is not None:
        report["recorded_gif"] = record_episode_gif(
            args.record_gif,
            env_id,
            representation,
            prediction,
            mcts,
            mode=args.gif_mode,
            seed=args.seed if args.gif_seed is None else args.gif_seed,
            max_steps=max_steps if args.gif_max_steps is None else args.gif_max_steps,
            fps=args.gif_fps,
            output_width=args.gif_width,
        )
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
