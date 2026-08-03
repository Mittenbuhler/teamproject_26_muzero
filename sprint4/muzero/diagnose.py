"""Diagnostics and synchronized native-MinAtar GIF recording for MuZero."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .buffers import EpisodeReplayBuffer
from .environment import ObservationHistory, make_minatar_env, minatar_env_id
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


def _observation_planes_panel(
    observation,
    output_height,
    base_channels,
    history_length,
    real_frame_count=None,
):
    """Render every exact native feature plane consumed by representation h."""
    history = np.asarray(observation)
    if history.ndim != 3 or history.shape[0] <= 0:
        raise ValueError(
            "GIF observation must have shape [channels, height, width]"
        )
    if not np.isfinite(history).all():
        raise ValueError("GIF observation history must contain only finite values")

    output_height = int(output_height)
    if output_height <= 0:
        raise ValueError("GIF observation panel height must be positive")
    base_channels = int(base_channels)
    history_length = int(history_length)
    if base_channels <= 0 or history_length <= 0:
        raise ValueError("base_channels and history_length must be positive")
    plane_count = base_channels * history_length
    if history.shape[0] != plane_count:
        raise ValueError(
            f"expected {plane_count} input planes, got {history.shape[0]}"
        )
    if real_frame_count is None:
        real_frame_count = history_length
    real_frame_count = int(real_frame_count)
    if not 0 <= real_frame_count <= history_length:
        raise ValueError("real_frame_count must be between zero and history length")

    margin = 6
    header_height = 20
    gap = 5
    columns = max(1, int(np.ceil(np.sqrt(plane_count))))
    rows = int(np.ceil(plane_count / columns))
    panel_width = output_height
    tile_size = max(
        1,
        min(
            (panel_width - 2 * margin - gap * (columns - 1)) // columns,
            (output_height - header_height - 2 * margin - gap * (rows - 1)) // rows,
        ),
    )
    panel = Image.new("RGB", (panel_width, output_height), (18, 18, 18))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    draw.text(
        (margin, 4),
        "exact model input: native feature planes",
        fill=(255, 255, 255),
        font=font,
    )

    padding_count = history_length - real_frame_count
    for index, frame in enumerate(history):
        pixels = np.rint(np.clip(frame.astype(np.float32), 0.0, 1.0) * 255.0)
        tile = Image.fromarray(pixels.astype(np.uint8), "L").convert("RGB")
        tile = tile.resize((tile_size, tile_size), Image.Resampling.NEAREST)
        row, column = divmod(index, columns)
        x = margin + column * (tile_size + gap)
        y = header_height + margin + row * (tile_size + gap)
        history_index, channel_index = divmod(index, base_channels)
        offset = history_length - 1 - history_index
        border = (106, 214, 136) if offset == 0 else (95, 95, 95)
        draw.rectangle(
            (x - 1, y - 1, x + tile_size, y + tile_size),
            outline=border,
        )
        panel.paste(tile, (x, y))

        time_label = "t" if offset == 0 else f"t-{offset}"
        label = (
            f"pad/c{channel_index}"
            if history_index < padding_count
            else f"{time_label}/c{channel_index}"
        )
        label_y = max(header_height, y + tile_size - 11)
        draw.rectangle((x, label_y, x + tile_size, y + tile_size), fill=(0, 0, 0))
        draw.text(
            (x + 2, label_y),
            label,
            fill=(210, 210, 210),
            font=font,
        )
    return panel


def _annotate_frame(
    frame,
    output_width,
    lines,
    observation=None,
    base_channels=None,
    history_length=1,
    real_frame_count=None,
):
    image = _rgb_image(frame, output_width)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    line_height = 12
    panel_height = 7 + line_height * len(lines)
    draw.rectangle((0, 0, image.width, panel_height), fill=(0, 0, 0))
    for line_index, line in enumerate(lines):
        draw.text(
            (5, 4 + line_index * line_height),
            str(line),
            fill=(255, 255, 255),
            font=font,
        )
    if observation is None:
        return image

    history_panel = _observation_planes_panel(
        observation,
        image.height,
        base_channels=base_channels,
        history_length=history_length,
        real_frame_count=real_frame_count,
    )
    composite = Image.new(
        "RGB",
        (image.width + history_panel.width, image.height),
        (0, 0, 0),
    )
    composite.paste(image, (0, 0))
    composite.paste(history_panel, (image.width, 0))
    return composite


def _format_policy(probabilities):
    return "[" + ", ".join(f"{value:.3f}" for value in probabilities) + "]"


@torch.no_grad()
def record_episode_gif(
    output_path,
    game,
    representation,
    prediction,
    mcts,
    base_observation_shape,
    history_length,
    mode="mcts",
    seed=0,
    max_steps=2500,
    fps=20,
    output_width=480,
    sticky_action_prob=0.1,
    difficulty_ramping=True,
):
    """Record rendered play beside every exact live native input plane."""
    if mode not in {"raw", "mcts"}:
        raise ValueError("GIF mode must be 'raw' or 'mcts'")
    if int(max_steps) <= 0:
        raise ValueError("GIF max_steps must be positive")
    if int(fps) <= 0:
        raise ValueError("GIF fps must be positive")
    if int(output_width) <= 0:
        raise ValueError("GIF output_width must be positive")
    base_observation_shape = tuple(int(size) for size in base_observation_shape)
    if len(base_observation_shape) != 3 or any(
        size <= 0 for size in base_observation_shape
    ):
        raise ValueError(
            "base_observation_shape must be positive (channels, height, width)"
        )
    if int(history_length) <= 0:
        raise ValueError("history_length must be positive")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_channels = base_observation_shape[0]
    env = make_minatar_env(
        game,
        sticky_action_prob=sticky_action_prob,
        difficulty_ramping=difficulty_ramping,
    )
    numpy_random_state = np.random.get_state()
    np.random.seed(int(seed))
    frames = []
    score = 0.0
    steps = 0
    status = "max_steps"
    try:
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(int(seed))
        frame, _ = env.reset(seed=int(seed))
        history_stack = ObservationHistory(
            int(history_length),
            base_observation_shape,
        )
        observation = history_stack.reset(frame)

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
                    base_channels=base_channels,
                    history_length=history_length,
                    real_frame_count=history_stack.real_frame_count,
                )
            )

            frame, reward, terminated, truncated, _ = env.step(action)
            observation = history_stack.append(frame)
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
                base_channels=base_channels,
                history_length=history_length,
                real_frame_count=history_stack.real_frame_count,
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
        "native_feature_planes": int(base_channels),
        "history_length": int(history_length),
    }


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate and visualize a native-MinAtar MuZero checkpoint."
    )
    parser.add_argument("checkpoint_positional", nargs="?")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--game",
        help="override/check the checkpoint game (for example breakout)",
    )
    parser.add_argument("--replay", help="optional torch-saved EpisodeReplayBuffer")
    parser.add_argument("--simulations", type=int, default=25)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
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
    parser.add_argument(
        "--output-json",
        type=Path,
        help="optionally save the printed diagnostic report as JSON",
    )
    return parser


def main():
    args = build_argument_parser().parse_args()
    checkpoint_path = args.checkpoint or args.checkpoint_positional
    if checkpoint_path is None:
        raise SystemExit("provide a checkpoint path or --checkpoint PATH")
    device = torch.device("cpu")
    representation, dynamics, prediction, metadata = load_latent_checkpoint(
        checkpoint_path,
        device,
    )
    action_dim = metadata["action_dim"]
    training_config = metadata.get("training_config", {})
    game = args.game or metadata.get("game") or metadata.get("env_id")
    if game is None:
        raise ValueError("checkpoint does not contain a MinAtar game id")
    checkpoint_env_id = metadata.get("env_id")
    if checkpoint_env_id is not None and minatar_env_id(game) != checkpoint_env_id:
        raise ValueError(
            f"requested game {minatar_env_id(game)} does not match checkpoint "
            f"environment {checkpoint_env_id}"
        )
    max_steps = int(training_config.get("max_steps", 2500))
    history_length = int(metadata.get("history_length", 1))
    base_observation_shape = tuple(metadata.get("base_observation_shape", ()))
    if len(base_observation_shape) != 3:
        raise ValueError("checkpoint lacks a valid native base_observation_shape")
    sticky_action_prob = float(metadata.get("sticky_action_prob", 0.1))
    difficulty_ramping = bool(metadata.get("difficulty_ramping", True))
    mcts = make_latent_mcts(
        dynamics,
        prediction,
        action_dim,
        simulations=args.simulations,
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

    # Policy-change statistics use encoded replay roots when available and a
    # deterministic blank encoded root otherwise.
    replay = torch.load(args.replay, weights_only=False) if args.replay else None
    if replay is None and "replay_state" in metadata:
        replay = EpisodeReplayBuffer()
        replay.load_state_dict(metadata["replay_state"])
    roots = (
        []
        if replay is None
        else [episode[0]["observation"] for episode in list(replay.episodes)[:64]]
    )
    if not roots:
        roots = [
            np.zeros(
                (metadata["input_channels"], *metadata["observation_shape"]),
                np.float32,
            )
        ]
    for observation in roots:
        latent = representation.encode(observation)
        raw_policies.append(prediction.predict(latent)[0])
        visit_policies.append(
            visit_count_policy(
                mcts.search(latent, add_exploration_noise=False),
                temperature=1.0,
            )
        )

    report = {
        "checkpoint": str(checkpoint_path),
        "game": game,
        "env_id": checkpoint_env_id,
        "base_observation_shape": base_observation_shape,
        "history_length": history_length,
        "input_channels": metadata["input_channels"],
        "action_dim": action_dim,
        "simulations": args.simulations,
        **policy_search_metrics(raw_policies, visit_policies),
    }
    if replay is not None:
        report.update(
            diagnose_batch(
                representation,
                dynamics,
                prediction,
                replay,
                int(training_config.get("unroll_steps", 5)),
                action_dim,
                device,
            )
        )

    seeds = [args.seed + episode for episode in range(args.episodes)]
    raw_score, _ = evaluate(
        game,
        representation,
        prediction,
        mcts,
        args.episodes,
        max_steps,
        history_length,
        seeds,
        False,
        sticky_action_prob=sticky_action_prob,
        difficulty_ramping=difficulty_ramping,
    )
    mcts_score, _ = evaluate(
        game,
        representation,
        prediction,
        mcts,
        args.episodes,
        max_steps,
        history_length,
        seeds,
        True,
        sticky_action_prob=sticky_action_prob,
        difficulty_ramping=difficulty_ramping,
    )
    report["evaluation_raw_argmax"] = raw_score
    report["evaluation_mcts_identical_seeds"] = mcts_score

    if args.record_gif is not None:
        report["recorded_gif"] = record_episode_gif(
            args.record_gif,
            game,
            representation,
            prediction,
            mcts,
            base_observation_shape,
            history_length,
            mode=args.gif_mode,
            seed=args.seed if args.gif_seed is None else args.gif_seed,
            max_steps=max_steps if args.gif_max_steps is None else args.gif_max_steps,
            fps=args.gif_fps,
            output_width=args.gif_width,
            sticky_action_prob=sticky_action_prob,
            difficulty_ramping=difficulty_ramping,
        )
    report_json = json.dumps(report, indent=2)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(report_json + "\n", encoding="utf-8")
    print(report_json)


if __name__ == "__main__":
    main()
