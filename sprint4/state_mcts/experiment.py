"""Train optional vector-state models and evaluate their MCTS substitutions.

Examples:
    python -m state_mcts.experiment --models none
    python -m state_mcts.experiment --models value
    python -m state_mcts.experiment --models policy,dynamics
    python -m state_mcts.experiment --models all --ablation
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape as xml_escape
from itertools import combinations
import json
from pathlib import Path
import random

import gymnasium as gym
import numpy as np
import torch

from .search import (
    ExactCartPoleDynamics,
    LearnedCartPoleDynamics,
    ModularMCTS,
    NetworkPrior,
    NetworkValueEvaluator,
    RandomRolloutEvaluator,
    UniformPrior,
    select_mcts_action,
)
from .training import (
    collect_state_dataset,
    dynamics_curriculum,
    load_component,
    save_component,
    train_dynamics,
    train_policy,
    train_value,
)


COMPONENTS = ("dynamics", "policy", "value")
REPORT_SCHEMA_VERSION = 2
LOSS_NAMES = {
    "dynamics": "weighted multi-step Smooth L1 state loss",
    "policy": "negative log-likelihood classification loss",
    "value": "mean squared error on normalized survival targets",
}


def utc_timestamp():
    return datetime.now(timezone.utc).isoformat()


def load_report_history(path):
    """Load v2 history or losslessly wrap a legacy single-run report."""
    path = Path(path)
    if not path.exists():
        return {"schema_version": REPORT_SCHEMA_VERSION, "runs": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") == REPORT_SCHEMA_VERSION:
        if not isinstance(payload.get("runs"), list):
            raise ValueError("state MCTS report has an invalid runs field")
        return payload
    if not all(key in payload for key in ("selected_models", "evaluations", "settings")):
        raise ValueError("state MCTS report has an unknown schema")
    legacy_run = dict(payload)
    legacy_run.update(
        {
            "run_id": "legacy-imported-run",
            "status": "completed",
            "migrated_from_schema": 1,
        }
    )
    return {"schema_version": REPORT_SCHEMA_VERSION, "runs": [legacy_run]}


def write_report_history(path, history):
    """Atomically persist all runs so interruption cannot erase prior data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The dashboard can edit presentation metadata while an experiment is
    # running. Merge that live metadata before each experiment checkpoint so a
    # stale in-memory report cannot undo names or folders.
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = {}
        if isinstance(current.get("dashboard_folders"), list):
            history["dashboard_folders"] = current["dashboard_folders"]
        if isinstance(current.get("dashboard_run_folders"), dict):
            history["dashboard_run_folders"] = current["dashboard_run_folders"]
        current_runs = {
            run.get("run_id"): run
            for run in current.get("runs", [])
            if isinstance(run, dict) and run.get("run_id")
        }
        for run in history.get("runs", []):
            current_run = current_runs.get(run.get("run_id"))
            if current_run is None:
                continue
            if "display_name" in current_run:
                run["display_name"] = current_run["display_name"]
            else:
                run.pop("display_name", None)
    history["updated_at"] = utc_timestamp()
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def write_loss_graph(
    loss_dir,
    run_id,
    component,
    loss_history,
    validation_metrics,
    source,
    settings,
):
    """Write one component's epoch-loss curve as a self-contained SVG."""
    output_path = Path(loss_dir) / run_id / f"{component}.svg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 900, 500
    left, right, top, bottom = 78, 34, 92, 62
    plot_width = width - left - right
    plot_height = height - top - bottom
    values = [float(entry["training_loss"]) for entry in loss_history]
    epochs = [int(entry["epoch"]) for entry in loss_history]
    maximum = max(values, default=1.0)
    y_max = maximum * 1.08 if maximum > 0 else 1.0

    def x_position(epoch):
        if len(epochs) <= 1 or epochs[-1] == epochs[0]:
            return left + plot_width / 2
        return left + (epoch - epochs[0]) / (epochs[-1] - epochs[0]) * plot_width

    def y_position(value):
        return top + plot_height - value / y_max * plot_height

    grid = []
    for index in range(6):
        value = y_max * index / 5
        y = y_position(value)
        grid.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" '
            'stroke="#29394d" stroke-width="1"/>'
            f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" '
            f'fill="#91a2b7" font-size="12">{value:.4g}</text>'
        )
    points = " ".join(
        f"{x_position(epoch):.2f},{y_position(value):.2f}"
        for epoch, value in zip(epochs, values)
    )
    markers = "".join(
        f'<circle cx="{x_position(epoch):.2f}" cy="{y_position(value):.2f}" r="3.5" '
        f'fill="#62b0ff"><title>Epoch {epoch}: {value:.7g}'
        + (
            f' (rollout horizon {entry["rollout_horizon"]})'
            if "rollout_horizon" in entry
            else ""
        )
        + "</title></circle>"
        for entry, epoch, value in zip(loss_history, epochs, values)
    )
    x_labels = ""
    if epochs:
        label_epochs = sorted(set(epochs[index] for index in np.linspace(0, len(epochs) - 1, min(6, len(epochs)), dtype=int)))
        x_labels = "".join(
            f'<text x="{x_position(epoch):.2f}" y="{height-bottom+25}" text-anchor="middle" '
            f'fill="#91a2b7" font-size="12">{epoch}</text>'
            for epoch in label_epochs
        )
    metric_text = " · ".join(
        f"{key}={value:.5g}" if isinstance(value, (int, float)) else f"{key}={value}"
        for key, value in validation_metrics.items()
    )
    if values:
        plot = (
            f'<polyline points="{points}" fill="none" stroke="#62b0ff" '
            f'stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>{markers}'
        )
    else:
        plot = (
            f'<text x="{left + plot_width/2:.2f}" y="{top + plot_height/2:.2f}" '
            'text-anchor="middle" fill="#f6c85f" font-size="17">'
            "No per-epoch loss history stored in this checkpoint</text>"
        )
    setting_text = " · ".join(
        f"{key}={value}" for key, value in settings.items() if value is not None
    )
    subtitle = (
        f"{LOSS_NAMES[component]} · source={source} · epochs={len(values)}"
        + (f" · {setting_text}" if setting_text else "")
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#0b1016"/>
<text x="{left}" y="36" fill="#e7edf5" font-family="system-ui,sans-serif" font-size="24" font-weight="700">{xml_escape(component.title())} training loss</text>
<text x="{left}" y="60" fill="#91a2b7" font-family="system-ui,sans-serif" font-size="13">{xml_escape(run_id)}</text>
<text x="{left}" y="80" fill="#91a2b7" font-family="system-ui,sans-serif" font-size="12">{xml_escape(subtitle)}</text>
<g font-family="system-ui,sans-serif">{''.join(grid)}{x_labels}{plot}</g>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#91a2b7"/>
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#91a2b7"/>
<text x="{left + plot_width/2:.2f}" y="{height-14}" text-anchor="middle" fill="#91a2b7" font-family="system-ui,sans-serif" font-size="13">Training epoch</text>
<text x="18" y="{top + plot_height/2:.2f}" text-anchor="middle" fill="#91a2b7" font-family="system-ui,sans-serif" font-size="13" transform="rotate(-90 18 {top + plot_height/2:.2f})">Mean training loss</text>
<text x="{width-right}" y="36" text-anchor="end" fill="#55d69e" font-family="system-ui,sans-serif" font-size="12">{xml_escape(metric_text)}</text>
</svg>'''
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(svg, encoding="utf-8")
    temporary_path.replace(output_path)
    return output_path


def new_report_run(args, selected):
    model_label = "+".join(name for name in COMPONENTS if name in selected) or "baseline"
    started_at = utc_timestamp()
    return {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ-") + model_label,
        "status": "running",
        "stage": "training",
        "started_at": started_at,
        "selected_models": sorted(selected),
        "training_metrics": {},
        "evaluations": {},
        "settings": {
            "env": args.env,
            "ablation": args.ablation,
            "train_samples": args.train_samples,
            "train_epochs": args.train_epochs,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "learning_rate": args.learning_rate,
            "expert_probability": args.expert_probability,
            "simulations": args.simulations,
            "search_depth": args.search_depth,
            "bootstrap_after": args.bootstrap_after,
            "search_depth_controls": [
                "MCTS tree lookahead",
                "random leaf-rollout length",
                "dynamics multi-step training horizon when dynamics is trained",
                "minimum depth before learned value bootstrap when value is active",
            ],
            "value_target_horizon": args.max_steps,
            "eval_episodes": args.eval_episodes,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "eval_seed": args.eval_seed,
            "reuse_checkpoints": args.reuse_checkpoints,
            "checkpoint_dir": str(args.checkpoint_dir),
            "loss_dir": str(args.loss_dir),
            "dynamics_training": {
                "maximum_rollout_horizon": args.search_depth,
                "horizon_source": "search_depth",
                "curriculum": list(dynamics_curriculum(args.search_depth)),
                "loss": "one-step anchor plus weighted multi-step Smooth L1 bands",
                "band_weights": {
                    "step_1": 1.0,
                    "steps_2_5": 0.5,
                    "steps_6_10": 0.25,
                    "steps_11_20": 0.125,
                    "steps_21_max": 0.0625,
                },
                "reward": "hardcoded +1 during MCTS; reward head is not trained",
                "terminal_handling": "mask sequence after Gym episode boundary",
                "validation_split": "whole episodes",
            },
        },
    }


def parse_models(value):
    value = value.strip().lower()
    if value == "all":
        return frozenset(COMPONENTS)
    if value == "none" or not value:
        return frozenset()
    selected = frozenset(part.strip() for part in value.split(","))
    unknown = selected.difference(COMPONENTS)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown models: {', '.join(sorted(unknown))}")
    return selected


def component_configurations(selected, ablation):
    if not ablation:
        return [selected]
    ordered = [name for name in COMPONENTS if name in selected]
    return [
        frozenset(configuration)
        for size in range(len(ordered) + 1)
        for configuration in combinations(ordered, size)
    ]


def build_mcts(
    enabled,
    models,
    exact_dynamics,
    simulations,
    search_depth,
    seed,
    value_horizon=500,
    bootstrap_after=0,
):
    transitions = (
        LearnedCartPoleDynamics(models["dynamics"], exact_dynamics.is_terminal)
        if "dynamics" in enabled
        else exact_dynamics
    )
    priors = NetworkPrior(models["policy"]) if "policy" in enabled else UniformPrior()
    evaluator = (
        NetworkValueEvaluator(models["value"], value_horizon)
        if "value" in enabled
        else RandomRolloutEvaluator(transitions, search_depth, seed=seed)
    )
    return ModularMCTS(
        transitions,
        priors,
        evaluator,
        simulations=simulations,
        search_depth=search_depth,
        bootstrap_after=bootstrap_after,
        seed=seed,
    )


def evaluate_agent(env_id, mcts, episodes=10, max_steps=500, seed=1000):
    env = gym.make(env_id)
    rewards = []
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            total = 0.0
            for _ in range(max_steps):
                action = select_mcts_action(mcts.search(state))
                state, reward, terminated, truncated, _ = env.step(action)
                total += reward
                if terminated or truncated:
                    break
            rewards.append(total)
    finally:
        env.close()
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "episode_rewards": rewards,
    }


def train_selected(args, selected, device, run_id):
    if not selected:
        return {}, {}, {}
    dataset = None
    models = {}
    metrics = {}
    loss_graphs = {}
    trainers = {
        "dynamics": train_dynamics,
        "policy": train_policy,
        "value": train_value,
    }
    for offset, component in enumerate(COMPONENTS):
        if component not in selected:
            continue
        checkpoint_path = Path(args.checkpoint_dir) / f"state_{component}.pt"
        if args.reuse_checkpoints and checkpoint_path.exists():
            model, checkpoint = load_component(checkpoint_path, component, device=device)
            if component == "dynamics" and (
                checkpoint.get("dynamics_training_version") != 2
                or checkpoint.get("training_horizon") != args.search_depth
            ):
                raise ValueError(
                    "Dynamics checkpoint does not use the current multi-step "
                    f"training scheme at search depth {args.search_depth}. "
                    "Run without --reuse-checkpoints to retrain it."
                )
            if component == "value" and (
                checkpoint.get("value_target_horizon") != args.max_steps
            ):
                raise ValueError(
                    "Value checkpoint uses a different episode target horizon. "
                    "Run without --reuse-checkpoints to retrain it."
                )
            models[component] = model
            metrics[component] = checkpoint.get("metrics", {})
            loss_path = write_loss_graph(
                args.loss_dir,
                run_id,
                component,
                checkpoint.get("loss_history", []),
                metrics[component],
                "checkpoint",
                {
                    "train_epochs": args.train_epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "search_depth": args.search_depth,
                },
            )
            loss_graphs[component] = str(loss_path)
            print(f"loaded {component}: {checkpoint_path}")
            print(f"saved {component} losses: {loss_path}")
            continue
        if dataset is None:
            print(f"collecting {args.train_samples} real Gym state transitions")
            dataset = collect_state_dataset(
                env_id=args.env,
                samples=args.train_samples,
                value_horizon=args.max_steps,
                expert_probability=args.expert_probability,
                seed=args.seed,
                include_value_targets="value" in selected,
            )
        print(f"training {component} independently")
        trainer_kwargs = {}
        loss_history = []
        if component == "dynamics":
            trainer_kwargs["rollout_horizon"] = args.search_depth
        model, component_metrics = trainers[component](
            dataset,
            hidden_dim=args.hidden_dim,
            epochs=args.train_epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            seed=args.seed + offset,
            device=device,
            loss_history=loss_history,
            **trainer_kwargs,
        )
        models[component] = model
        metrics[component] = component_metrics
        component_horizon = (
            args.search_depth
            if component == "dynamics"
            else args.max_steps
            if component == "value"
            else None
        )
        save_component(
            checkpoint_path,
            component,
            model,
            component_metrics,
            value_horizon=component_horizon,
            loss_history=loss_history,
        )
        loss_path = write_loss_graph(
            args.loss_dir,
            run_id,
            component,
            loss_history,
            component_metrics,
            "trained",
            {
                "train_epochs": args.train_epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "search_depth": args.search_depth,
                "training_rollout_horizon": (
                    args.search_depth if component == "dynamics" else None
                ),
            },
        )
        loss_graphs[component] = str(loss_path)
        print(f"{component} validation: {component_metrics}")
        print(f"saved {component} losses: {loss_path}")
    return models, metrics, loss_graphs


def run(args):
    if args.bootstrap_after < 0 or args.bootstrap_after > args.search_depth:
        raise ValueError("--bootstrap-after must be between 0 and --search-depth")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = args.models
    output_path = Path(args.report_path)
    report_history = load_report_history(output_path)
    report = new_report_run(args, selected)
    report_history["runs"].append(report)
    write_report_history(output_path, report_history)
    print(f"initialized report run {report['run_id']}: {output_path}")

    try:
        models, training_metrics, loss_graphs = train_selected(
            args, selected, device, report["run_id"]
        )
        report["training_metrics"] = training_metrics
        report["loss_graphs"] = loss_graphs
        report["stage"] = "evaluation"
        write_report_history(output_path, report_history)

        calibration_env = gym.make(args.env)
        exact_dynamics = ExactCartPoleDynamics.from_env(calibration_env)
        calibration_env.close()
        for index, enabled in enumerate(component_configurations(selected, args.ablation)):
            label = "+".join(name for name in COMPONENTS if name in enabled) or "baseline"
            print(f"evaluating {label}")
            mcts = build_mcts(
                enabled,
                models,
                exact_dynamics,
                simulations=args.simulations,
                search_depth=args.search_depth,
                bootstrap_after=args.bootstrap_after,
                seed=args.seed + 10_000 * index,
                value_horizon=args.max_steps,
            )
            report["evaluations"][label] = evaluate_agent(
                args.env,
                mcts,
                episodes=args.eval_episodes,
                max_steps=args.max_steps,
                seed=args.eval_seed,
            )
            summary = report["evaluations"][label]
            write_report_history(output_path, report_history)
            print(
                f"{label}: mean={summary['mean_reward']:.1f} "
                f"std={summary['std_reward']:.1f} "
                f"range={summary['min_reward']:.0f}-{summary['max_reward']:.0f}"
            )

        report["status"] = "completed"
        report["stage"] = "completed"
        report["completed_at"] = utc_timestamp()
        write_report_history(output_path, report_history)
        print(f"saved report history: {output_path}")
        return report
    except BaseException as error:
        report["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        report["finished_at"] = utc_timestamp()
        write_report_history(output_path, report_history)
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Modular real-state MCTS ablations for CartPole-v1."
    )
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--models", type=parse_models, default=parse_models("all"))
    parser.add_argument(
        "--ablation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate every subset of the selected trained models.",
    )
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--train-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--expert-probability", type=float, default=0.5)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument(
        "--search-depth",
        type=int,
        default=30,
        help=(
            "Global MCTS lookahead depth. Also sets dynamics multi-step training "
            "horizon when dynamics is trained; does not change policy or value targets."
        ),
    )
    parser.add_argument(
        "--bootstrap-after",
        type=int,
        default=0,
        help=(
            "Minimum simulated tree depth before a learned value leaf evaluator "
            "may bootstrap. Default 0 preserves the current immediate-bootstrap behavior."
        ),
    )
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--checkpoint-dir", default="checkpoints/state_mcts")
    parser.add_argument("--loss-dir", default="artifacts/state_mcts/losses")
    parser.add_argument(
        "--report-path", default="artifacts/state_mcts/state_mcts_report.json"
    )
    parser.add_argument("--reuse-checkpoints", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
