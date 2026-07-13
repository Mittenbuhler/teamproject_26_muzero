"""Compare a distilled value checkpoint with vanilla-MCTS action choices.

This is a diagnostic, not a trainer. It regenerates states from teacher-MCTS
episodes, runs a fresh vanilla-MCTS root search at each state, and compares the
teacher's root-visit action with the one implied by the value network:

    argmax_a reward(s, a) / search_depth + V(next_state(s, a))

The output answers whether the scalar value surface has the local action
ordering needed by value-only MCTS.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from .mcts_distillation import make_teacher_mcts, selected_visit_fraction, visit_distribution
from .search import ExactCartPoleDynamics, select_mcts_action
from .training import load_component


REPORT_SCHEMA_VERSION = 1


def utc_timestamp():
    return datetime.now(timezone.utc).isoformat()


def value_action_scores(value_model, exact_dynamics, state, search_depth):
    """Score CartPole's two actions using immediate reward plus V(next_state)."""
    scores = []
    next_values = []
    dones = []
    for action in range(2):
        next_state, reward, done = exact_dynamics.step(state, action)
        future_value = 0.0 if done else float(np.clip(value_model.value(next_state), 0.0, 1.0))
        scores.append(float(reward / search_depth + future_value))
        next_values.append(float(future_value))
        dones.append(bool(done))
    return np.asarray(scores, dtype=np.float32), next_values, dones


def value_implied_action(value_model, exact_dynamics, state, search_depth):
    scores, next_values, dones = value_action_scores(
        value_model,
        exact_dynamics,
        state,
        search_depth,
    )
    action = int(np.flatnonzero(scores == scores.max())[0])
    return action, scores, next_values, dones


def summarize_records(records, confidence_thresholds=(0.55, 0.6, 0.7)):
    agreements = np.asarray([record["agreement"] for record in records], dtype=np.float32)
    value_margins = np.asarray([record["value_margin"] for record in records], dtype=np.float32)
    visit_fractions = np.asarray(
        [record["mcts_selected_visit_fraction"] for record in records],
        dtype=np.float32,
    )
    mcts_actions = np.asarray([record["mcts_action"] for record in records], dtype=np.int64)
    value_actions = np.asarray([record["value_action"] for record in records], dtype=np.int64)
    wrong_margins = value_margins[agreements == 0]
    summary = {
        "samples": len(records),
        "agreement": float(agreements.mean()) if len(records) else 0.0,
        "mcts_action_fraction_right": float(mcts_actions.mean()) if len(records) else 0.0,
        "value_action_fraction_right": float(value_actions.mean()) if len(records) else 0.0,
        "mcts_selected_visit_fraction_mean": (
            float(visit_fractions.mean()) if len(records) else 0.0
        ),
        "value_margin_mean": float(value_margins.mean()) if len(records) else 0.0,
        "value_margin_median": float(np.median(value_margins)) if len(records) else 0.0,
        "wrong_action_value_margin_mean": (
            float(wrong_margins.mean()) if len(wrong_margins) else 0.0
        ),
        "wrong_action_count": int((agreements == 0).sum()) if len(records) else 0,
    }
    for threshold in confidence_thresholds:
        mask = visit_fractions >= threshold
        summary[f"agreement_when_mcts_visit_fraction_at_least_{threshold:.2f}"] = (
            float(agreements[mask].mean()) if mask.any() else None
        )
        summary[f"samples_when_mcts_visit_fraction_at_least_{threshold:.2f}"] = int(mask.sum())
    return summary


def select_records_to_store(
    records,
    limit,
    mode="first",
    confidence_threshold=0.7,
):
    """Choose which raw examples to keep without changing summary metrics."""
    if limit <= 0:
        return []
    if mode == "first":
        selected = records
    elif mode == "high-confidence":
        selected = [
            record
            for record in records
            if record["mcts_selected_visit_fraction"] >= confidence_threshold
        ]
    elif mode == "disagreements":
        selected = [record for record in records if not record["agreement"]]
    elif mode == "high-confidence-disagreements":
        selected = [
            record
            for record in records
            if (
                record["mcts_selected_visit_fraction"] >= confidence_threshold
                and not record["agreement"]
            )
        ]
    else:
        raise ValueError(f"unknown store record mode: {mode}")
    return selected[:limit]


def collect_action_comparison_records(
    value_model,
    env_id="CartPole-v1",
    samples=500,
    teacher_episodes=30,
    teacher_simulations=64,
    search_depth=30,
    max_steps=500,
    seed=0,
):
    if samples <= 0 or teacher_episodes <= 0:
        raise ValueError("samples and teacher_episodes must be positive")
    env = gym.make(env_id)
    env.action_space.seed(seed)
    records = []
    episode_rewards = []
    try:
        exact = ExactCartPoleDynamics.from_env(env)
        for episode in range(teacher_episodes):
            state, _ = env.reset(seed=seed + episode)
            teacher = make_teacher_mcts(
                exact,
                simulations=teacher_simulations,
                search_depth=search_depth,
                seed=seed + 10_000 * episode,
            )
            episode_reward = 0.0
            for step in range(max_steps):
                root = teacher.search(state)
                mcts_action = select_mcts_action(root)
                value_action, scores, next_values, dones = value_implied_action(
                    value_model,
                    exact,
                    state,
                    search_depth,
                )
                visits = [int(root.children[action].visits) for action in range(2)]
                policy_target = visit_distribution(root).astype(float).tolist()
                record = {
                    "episode": int(episode),
                    "step": int(step),
                    "state": np.asarray(state, dtype=np.float32).astype(float).tolist(),
                    "mcts_action": int(mcts_action),
                    "value_action": int(value_action),
                    "agreement": bool(mcts_action == value_action),
                    "mcts_child_visits": visits,
                    "mcts_policy_target": policy_target,
                    "mcts_selected_visit_fraction": selected_visit_fraction(root),
                    "value_action_scores": scores.astype(float).tolist(),
                    "value_next_state_values": next_values,
                    "value_child_done": dones,
                    "value_margin": float(abs(scores[1] - scores[0])),
                }
                records.append(record)
                state, reward, terminated, truncated, _ = env.step(mcts_action)
                episode_reward += reward
                if len(records) >= samples or terminated or truncated:
                    break
            episode_rewards.append(float(episode_reward))
            if len(records) >= samples:
                break
    finally:
        env.close()
    collection = {
        "episodes_collected": len(episode_rewards),
        "samples_collected": len(records),
        "teacher_episode_reward_mean": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "teacher_episode_reward_std": float(np.std(episode_rewards)) if episode_rewards else 0.0,
        "teacher_episode_rewards": episode_rewards,
    }
    return records, collection


def load_report_history(path):
    path = Path(path)
    if not path.exists():
        return {"schema_version": REPORT_SCHEMA_VERSION, "runs": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("value/MCTS diagnostic report has an unknown schema")
    if not isinstance(payload.get("runs"), list):
        raise ValueError("value/MCTS diagnostic report has an invalid runs field")
    return payload


def write_report_history(path, history):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    history["updated_at"] = utc_timestamp()
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def run(args):
    if args.search_depth <= 0 or args.teacher_simulations <= 0:
        raise ValueError("--search-depth and --teacher-simulations must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    value_model, checkpoint = load_component(args.value_checkpoint_path, "value", device=device)
    run_id = utc_timestamp().replace(":", "").replace("+00:00", "Z") + "-value-mcts-diagnostic"
    history = load_report_history(args.report_path)
    report = {
        "run_id": run_id,
        "status": "running",
        "created_at": utc_timestamp(),
        "purpose": (
            "Compare vanilla-MCTS root visit action with the action implied by "
            "a value checkpoint's one-step lookahead."
        ),
        "settings": vars(args),
        "checkpoint": {
            "path": str(args.value_checkpoint_path),
            "metrics": checkpoint.get("metrics", {}),
            "value_target_horizon": checkpoint.get("value_target_horizon"),
            "loss_history_epochs": len(checkpoint.get("loss_history", [])),
        },
        "teacher_collection": {},
        "summary": {},
        "records": [],
    }
    history["runs"].append(report)
    write_report_history(args.report_path, history)
    print(f"initialized value/MCTS diagnostic {run_id}: {args.report_path}")
    try:
        records, collection = collect_action_comparison_records(
            value_model,
            env_id=args.env,
            samples=args.samples,
            teacher_episodes=args.teacher_episodes,
            teacher_simulations=args.teacher_simulations,
            search_depth=args.search_depth,
            max_steps=args.max_steps,
            seed=args.seed,
        )
        report["teacher_collection"] = collection
        report["summary"] = summarize_records(records)
        report["record_storage"] = {
            "mode": args.store_record_mode,
            "limit": args.store_records,
            "confidence_threshold": args.store_confidence_threshold,
        }
        report["records"] = select_records_to_store(
            records,
            args.store_records,
            mode=args.store_record_mode,
            confidence_threshold=args.store_confidence_threshold,
        )
        report["stored_records"] = len(report["records"])
        report["status"] = "completed"
        report["completed_at"] = utc_timestamp()
        write_report_history(args.report_path, history)
        print(
            "value/MCTS action agreement: "
            f"{report['summary']['agreement']:.3f} over {report['summary']['samples']} states"
        )
        print(f"saved diagnostic report: {args.report_path}")
        return report
    except BaseException as error:
        report["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        report["finished_at"] = utc_timestamp()
        write_report_history(args.report_path, history)
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare a value checkpoint's implied actions against vanilla MCTS."
    )
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--teacher-episodes", type=int, default=30)
    parser.add_argument("--teacher-simulations", type=int, default=64)
    parser.add_argument("--search-depth", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--value-checkpoint-path",
        default="checkpoints/state_mcts/state_value_mcts_teacher.pt",
    )
    parser.add_argument(
        "--report-path",
        default="artifacts/state_mcts/value_mcts_action_diagnostic.json",
    )
    parser.add_argument(
        "--store-records",
        type=int,
        default=200,
        help="Number of per-state comparison records to keep in the JSON report.",
    )
    parser.add_argument(
        "--store-record-mode",
        choices=("first", "high-confidence", "disagreements", "high-confidence-disagreements"),
        default="first",
        help=(
            "Which raw records to keep. Summary metrics are always computed "
            "from all sampled states."
        ),
    )
    parser.add_argument(
        "--store-confidence-threshold",
        type=float,
        default=0.7,
        help="MCTS selected-visit-fraction threshold for high-confidence record storage.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
