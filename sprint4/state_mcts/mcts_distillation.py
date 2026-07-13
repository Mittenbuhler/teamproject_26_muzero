"""Distill vanilla MCTS search statistics into toggleable policy/value nets.

This is the non-heuristic teacher-data path. The teacher is ordinary
exact-dynamics MCTS with uniform priors and random leaf rollouts.

Targets:

* policy = root child visit distribution
* value = normalized discounted teacher episode return-to-go

Use ``--models policy``, ``--models value``, or ``--models policy,value`` to
train and evaluate each distilled component independently.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from .experiment import evaluate_agent, write_loss_graph
from .models import PolicyNetwork
from .search import (
    ExactCartPoleDynamics,
    ModularMCTS,
    NetworkPrior,
    NetworkValueEvaluator,
    RandomRolloutEvaluator,
    UniformPrior,
    select_mcts_action,
)
from .training import StateDataset, _batches, _tensor, save_component, train_value


COMPONENTS = ("policy", "value")
REPORT_SCHEMA_VERSION = 2


@dataclass
class MCTSDistillationDataset:
    states: np.ndarray
    policy_targets: np.ndarray
    value_targets: np.ndarray
    selected_actions: np.ndarray
    episode_ids: np.ndarray
    selected_visit_fractions: np.ndarray

    def split(self, validation_fraction=0.2, seed=0):
        """Split whole teacher episodes to avoid neighboring-state leakage."""
        rng = np.random.default_rng(seed)
        episodes = rng.permutation(np.unique(self.episode_ids))
        if len(episodes) < 2:
            indices = np.arange(len(self.states))
            split_at = min(
                len(indices) - 1,
                max(1, int(len(indices) * (1 - validation_fraction))),
            )
            return self.subset(indices[:split_at]), self.subset(indices[split_at:])
        validation_count = min(
            len(episodes) - 1,
            max(1, int(round(len(episodes) * validation_fraction))),
        )
        validation_mask = np.isin(self.episode_ids, episodes[:validation_count])
        return (
            self.subset(np.flatnonzero(~validation_mask)),
            self.subset(np.flatnonzero(validation_mask)),
        )

    def subset(self, indices):
        return MCTSDistillationDataset(
            states=self.states[indices],
            policy_targets=self.policy_targets[indices],
            value_targets=self.value_targets[indices],
            selected_actions=self.selected_actions[indices],
            episode_ids=self.episode_ids[indices],
            selected_visit_fractions=self.selected_visit_fractions[indices],
        )

    def as_value_dataset(self) -> StateDataset:
        count = len(self.states)
        return StateDataset(
            states=self.states.astype(np.float32),
            actions=np.zeros(count, dtype=np.int64),
            next_states=self.states.astype(np.float32),
            rewards=np.zeros((count, 1), dtype=np.float32),
            policy_actions=self.selected_actions.astype(np.int64),
            value_targets=self.value_targets.astype(np.float32).reshape(-1, 1),
            episode_ids=self.episode_ids.astype(np.int64),
        )


def utc_timestamp():
    return datetime.now(timezone.utc).isoformat()


def parse_models(value):
    if value in {"all", "both", "policy,value", "value,policy"}:
        return frozenset(COMPONENTS)
    if value in {"none", ""}:
        return frozenset()
    selected = frozenset(part.strip() for part in value.split(",") if part.strip())
    unknown = selected.difference(COMPONENTS)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown models: {', '.join(sorted(unknown))}")
    return selected


def make_teacher_mcts(exact_dynamics, simulations, search_depth, seed):
    return ModularMCTS(
        exact_dynamics,
        UniformPrior(),
        RandomRolloutEvaluator(exact_dynamics, search_depth, seed=seed),
        simulations=simulations,
        search_depth=search_depth,
        seed=seed,
    )


def discounted_return_normalizer(discount: float, horizon: int) -> float:
    """Maximum discounted return for one unit reward over ``horizon`` steps."""
    if horizon <= 0:
        raise ValueError("discounted return horizon must be positive")
    if not 0.0 <= discount <= 1.0:
        raise ValueError("discount must be between 0 and 1")
    if discount == 1.0:
        return float(horizon)
    return float((1.0 - discount**horizon) / (1.0 - discount))


def discounted_return_targets(
    rewards,
    discount: float = 1.0,
    horizon: int = 500,
):
    """Return normalized discounted return-to-go targets for an episode."""
    normalizer = discounted_return_normalizer(discount, horizon)
    targets = []
    running_return = 0.0
    for reward in reversed(rewards):
        running_return = float(reward) + discount * running_return
        targets.append(float(np.clip(running_return / normalizer, 0.0, 1.0)))
    return list(reversed(targets))


def value_target_description(discount: float, value_horizon: int) -> str:
    return (
        "Normalized discounted return-to-go from the completed "
        "MCTS-teacher episode: sum_t gamma^t reward_t / "
        f"sum_t gamma^t over value_horizon={value_horizon}, gamma={discount}."
    )


def visit_distribution(root, action_dim=2):
    """Return normalized root child visits as a policy target."""
    visits = np.asarray(
        [root.children[action].visits for action in range(action_dim)],
        dtype=np.float32,
    )
    total = float(visits.sum())
    if total <= 0.0:
        return np.full(action_dim, 1.0 / action_dim, dtype=np.float32)
    return visits / total


def selected_visit_fraction(root) -> float:
    visits = [child.visits for child in root.children.values()]
    total = sum(visits)
    return float(max(visits) / total) if total else 0.0


def collect_mcts_distillation_dataset(
    env_id="CartPole-v1",
    samples=5000,
    teacher_episodes=20,
    simulations=64,
    search_depth=30,
    discount=1.0,
    value_horizon=None,
    max_steps=500,
    seed=0,
):
    """Play with vanilla MCTS and record policy plus return-to-go value targets."""
    if samples <= 0 or teacher_episodes <= 0:
        raise ValueError("samples and teacher_episodes must be positive")
    value_horizon = value_horizon or max_steps
    discounted_return_normalizer(discount, value_horizon)
    env = gym.make(env_id)
    env.action_space.seed(seed)
    states = []
    policy_targets = []
    value_targets = []
    selected_actions = []
    episode_ids = []
    visit_fractions = []
    episode_rewards = []
    try:
        exact = ExactCartPoleDynamics.from_env(env)
        for episode in range(teacher_episodes):
            state, _ = env.reset(seed=seed + episode)
            teacher = make_teacher_mcts(
                exact,
                simulations=simulations,
                search_depth=search_depth,
                seed=seed + 10_000 * episode,
            )
            total_reward = 0.0
            episode_states = []
            episode_policy_targets = []
            episode_selected_actions = []
            episode_visit_fractions = []
            episode_step_rewards = []
            for _step in range(max_steps):
                root = teacher.search(state)
                action = select_mcts_action(root)
                episode_states.append(np.asarray(state, dtype=np.float32))
                episode_policy_targets.append(visit_distribution(root))
                episode_selected_actions.append(action)
                episode_visit_fractions.append(selected_visit_fraction(root))
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                episode_step_rewards.append(float(reward))
                if terminated or truncated:
                    break
            episode_rewards.append(float(total_reward))
            episode_value_targets = discounted_return_targets(
                episode_step_rewards,
                discount=discount,
                horizon=value_horizon,
            )

            remaining = samples - len(states)
            take = min(remaining, len(episode_states))
            if take > 0:
                states.extend(episode_states[:take])
                policy_targets.extend(episode_policy_targets[:take])
                value_targets.extend(episode_value_targets[:take])
                selected_actions.extend(episode_selected_actions[:take])
                episode_ids.extend([episode] * take)
                visit_fractions.extend(episode_visit_fractions[:take])
            if len(states) >= samples:
                break
    finally:
        env.close()

    return MCTSDistillationDataset(
        states=np.asarray(states, dtype=np.float32),
        policy_targets=np.asarray(policy_targets, dtype=np.float32),
        value_targets=np.asarray(value_targets, dtype=np.float32),
        selected_actions=np.asarray(selected_actions, dtype=np.int64),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        selected_visit_fractions=np.asarray(visit_fractions, dtype=np.float32),
    ), {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "episode_rewards": episode_rewards,
        "episodes_collected": len(episode_rewards),
        "samples_collected": len(states),
    }


def train_policy_from_visit_targets(
    dataset,
    hidden_dim=128,
    epochs=30,
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
            targets = _tensor(train.policy_targets[indices], device)
            loss = -(targets * torch.log(probabilities + 1e-8)).sum(dim=1).mean()
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
        probabilities = model(_tensor(validation.states, device))
        targets = _tensor(validation.policy_targets, device)
        cross_entropy = -(targets * torch.log(probabilities + 1e-8)).sum(dim=1).mean().item()
        teacher_actions = _tensor(validation.policy_targets.argmax(axis=1), device, torch.long)
        accuracy = (probabilities.argmax(dim=1) == teacher_actions).float().mean().item()
        kl = F.kl_div(torch.log(probabilities + 1e-8), targets, reduction="batchmean").item()
    return model, {
        "cross_entropy": cross_entropy,
        "visit_argmax_accuracy": accuracy,
        "kl_divergence": kl,
    }


def evaluate_teacher(env_id, simulations, search_depth, episodes, max_steps, seed):
    env = gym.make(env_id)
    try:
        exact = ExactCartPoleDynamics.from_env(env)
    finally:
        env.close()
    teacher = make_teacher_mcts(exact, simulations, search_depth, seed)
    return evaluate_agent(env_id, teacher, episodes=episodes, max_steps=max_steps, seed=seed)


def evaluate_hybrid(
    env_id,
    enabled,
    models,
    simulations,
    search_depth,
    bootstrap_after,
    episodes,
    max_steps,
    seed,
):
    env = gym.make(env_id)
    try:
        exact = ExactCartPoleDynamics.from_env(env)
    finally:
        env.close()
    priors = NetworkPrior(models["policy"]) if "policy" in enabled else UniformPrior()
    evaluator = (
        NetworkValueEvaluator(models["value"], value_horizon=search_depth)
        if "value" in enabled
        else RandomRolloutEvaluator(exact, search_depth, seed=seed)
    )
    hybrid = ModularMCTS(
        exact,
        priors,
        evaluator,
        simulations=simulations,
        search_depth=search_depth,
        bootstrap_after=bootstrap_after if "value" in enabled else 0,
        seed=seed,
    )
    return evaluate_agent(env_id, hybrid, episodes=episodes, max_steps=max_steps, seed=seed)


def target_summary(dataset):
    policy_entropy = -np.sum(
        dataset.policy_targets * np.log(dataset.policy_targets + 1e-8),
        axis=1,
    )
    return {
        "value_target_mean": float(np.mean(dataset.value_targets)),
        "value_target_std": float(np.std(dataset.value_targets)),
        "value_target_min": float(np.min(dataset.value_targets)),
        "value_target_max": float(np.max(dataset.value_targets)),
        "policy_entropy_mean": float(np.mean(policy_entropy)),
        "selected_visit_fraction_mean": float(np.mean(dataset.selected_visit_fractions)),
        "selected_action_fraction_right": float(np.mean(dataset.selected_actions)),
    }


def load_report_history(path):
    """Load append-only history or migrate a legacy single-run report."""
    path = Path(path)
    if not path.exists():
        return {"schema_version": REPORT_SCHEMA_VERSION, "runs": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") == REPORT_SCHEMA_VERSION:
        if not isinstance(payload.get("runs"), list):
            raise ValueError("MCTS distillation report has an invalid runs field")
        return payload
    if not isinstance(payload, dict) or "run_id" not in payload:
        raise ValueError("MCTS distillation report has an unknown schema")
    legacy_run = dict(payload)
    legacy_run["migrated_from_schema"] = payload.get("schema_version", 1)
    return {"schema_version": REPORT_SCHEMA_VERSION, "runs": [legacy_run]}


def write_report_history(path, history):
    """Atomically persist all distillation runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    history["updated_at"] = utc_timestamp()
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def train_selected(args, selected, dataset, device, run_id):
    models = {}
    training_metrics = {}
    loss_graphs = {}
    if "policy" in selected:
        print("training policy from MCTS visit distributions")
        loss_history = []
        model, metrics = train_policy_from_visit_targets(
            dataset,
            hidden_dim=args.hidden_dim,
            epochs=args.train_epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            seed=args.seed,
            device=device,
            loss_history=loss_history,
        )
        save_component(
            args.policy_checkpoint_path,
            "policy",
            model,
            metrics,
            value_horizon=None,
            loss_history=loss_history,
        )
        loss_path = write_loss_graph(
            args.loss_dir,
            run_id,
            "policy",
            loss_history,
            metrics,
            "vanilla_mcts_visit_distribution_teacher",
            {
                "teacher_simulations": args.teacher_simulations,
                "search_depth": args.search_depth,
                "train_samples": len(dataset.states),
            },
        )
        models["policy"] = model
        training_metrics["policy"] = metrics
        loss_graphs["policy"] = str(loss_path)
        print(f"saved distilled policy checkpoint: {args.policy_checkpoint_path}")
    if "value" in selected:
        value_source = "vanilla_mcts_discounted_teacher_return_to_go"
        print("training value from discounted MCTS-teacher episode returns")
        loss_history = []
        model, metrics = train_value(
            dataset.as_value_dataset(),
            hidden_dim=args.hidden_dim,
            epochs=args.train_epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            seed=args.seed,
            device=device,
            loss_history=loss_history,
        )
        save_component(
            args.value_checkpoint_path,
            "value",
            model,
            metrics,
            value_horizon=args.value_horizon,
            loss_history=loss_history,
        )
        loss_path = write_loss_graph(
            args.loss_dir,
            run_id,
            "value",
            loss_history,
            metrics,
            value_source,
            {
                "teacher_simulations": args.teacher_simulations,
                "search_depth": args.search_depth,
                "value_target": "discounted_return_to_go",
                "discount": args.discount,
                "value_horizon": args.value_horizon,
                "train_samples": len(dataset.states),
            },
        )
        models["value"] = model
        training_metrics["value"] = metrics
        loss_graphs["value"] = str(loss_path)
        print(f"saved distilled value checkpoint: {args.value_checkpoint_path}")
    return models, training_metrics, loss_graphs


def run(args):
    if args.bootstrap_after < 0 or args.bootstrap_after > args.search_depth:
        raise ValueError("--bootstrap-after must be between 0 and --search-depth")
    if args.value_horizon is None:
        args.value_horizon = args.max_steps
    discounted_return_normalizer(args.discount, args.value_horizon)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = args.models
    if not selected:
        raise ValueError("select at least one model with --models policy,value")
    model_label = "+".join(component for component in COMPONENTS if component in selected)
    run_id = utc_timestamp().replace(":", "").replace("+00:00", "Z") + f"-mcts-{model_label}"
    report_path = Path(args.report_path)
    report_history = load_report_history(report_path)
    report = {
        "run_id": run_id,
        "status": "running",
        "stage": "collection",
        "created_at": utc_timestamp(),
        "selected_models": list(component for component in COMPONENTS if component in selected),
        "purpose": (
            "Train selected PolicyNetwork/ValueNetwork components from vanilla "
            "MCTS teacher episodes."
        ),
        "target_definition": {
            "policy": "Root child visit distribution from exact-dynamics vanilla MCTS.",
            "value": value_target_description(args.discount, args.value_horizon),
        },
        "settings": {
            **vars(args),
            "models": ",".join(component for component in COMPONENTS if component in selected),
            "bootstrap_after": args.bootstrap_after,
            "value_target": "discounted_return_to_go",
        },
        "teacher_collection": {},
        "target_summary": {},
        "training_metrics": {},
        "loss_graphs": {},
        "checkpoints": {},
        "evaluations": {},
    }
    report_history["runs"].append(report)
    write_report_history(report_path, report_history)
    print(f"initialized distillation report run {run_id}: {report_path}")

    try:
        print("collecting vanilla-MCTS teacher policy/value targets")
        dataset, teacher_collection = collect_mcts_distillation_dataset(
            env_id=args.env,
            samples=args.train_samples,
            teacher_episodes=args.teacher_episodes,
            simulations=args.teacher_simulations,
            search_depth=args.search_depth,
            discount=args.discount,
            value_horizon=args.value_horizon,
            max_steps=args.max_steps,
            seed=args.seed,
        )
        report["teacher_collection"] = teacher_collection
        report["target_summary"] = target_summary(dataset)
        report["stage"] = "training"
        write_report_history(report_path, report_history)
        print(
            "teacher collection: "
            f"mean={teacher_collection['mean_reward']:.1f}, "
            f"samples={teacher_collection['samples_collected']}"
        )
        if teacher_collection["mean_reward"] < args.min_teacher_mean:
            print(
                "warning: teacher mean reward is below "
                f"{args.min_teacher_mean:.1f}; networks will imitate a weak teacher"
            )

        models, training_metrics, loss_graphs = train_selected(
            args, selected, dataset, device, run_id
        )
        report["training_metrics"] = training_metrics
        report["loss_graphs"] = loss_graphs
        report["checkpoints"] = {
            key: str(path)
            for key, path in {
                "policy": args.policy_checkpoint_path if "policy" in selected else None,
                "value": args.value_checkpoint_path if "value" in selected else None,
            }.items()
            if path is not None
        }
        report["stage"] = "evaluation"
        write_report_history(report_path, report_history)

        print("evaluating exact-dynamics vanilla MCTS teacher")
        teacher_eval = evaluate_teacher(
            args.env,
            simulations=args.teacher_simulations,
            search_depth=args.search_depth,
            episodes=args.eval_episodes,
            max_steps=args.max_steps,
            seed=args.eval_seed,
        )
        report["evaluations"]["vanilla_mcts_teacher"] = teacher_eval
        write_report_history(report_path, report_history)
        label = "+".join(component for component in COMPONENTS if component in selected)
        print(f"evaluating exact-dynamics hybrid MCTS with distilled {label} NN(s)")
        hybrid_eval = evaluate_hybrid(
            args.env,
            selected,
            models,
            simulations=args.hybrid_simulations,
            search_depth=args.search_depth,
            bootstrap_after=args.bootstrap_after,
            episodes=args.eval_episodes,
            max_steps=args.max_steps,
            seed=args.eval_seed,
        )
        report["evaluations"][f"hybrid_mcts_{label}_distilled"] = hybrid_eval
        report["status"] = "completed"
        report["stage"] = "completed"
        report["completed_at"] = utc_timestamp()
        write_report_history(report_path, report_history)
        print(f"saved report history: {report_path}")
        return report
    except BaseException as error:
        report["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        report["finished_at"] = utc_timestamp()
        write_report_history(report_path, report_history)
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train policy and/or value NNs from vanilla MCTS root statistics."
    )
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument(
        "--models",
        type=parse_models,
        default=parse_models("policy,value"),
        help="Comma-separated subset: policy, value, policy,value, or all.",
    )
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--teacher-episodes", type=int, default=30)
    parser.add_argument("--teacher-simulations", type=int, default=64)
    parser.add_argument("--hybrid-simulations", type=int, default=64)
    parser.add_argument("--search-depth", type=int, default=30)
    parser.add_argument(
        "--discount",
        type=float,
        default=1.0,
        help="Gamma for discounted-return value targets. 1.0 is undiscounted.",
    )
    parser.add_argument(
        "--value-horizon",
        type=int,
        default=None,
        help=(
            "Normalizer horizon for discounted return-to-go value targets. "
            "Defaults to max-steps."
        ),
    )
    parser.add_argument(
        "--bootstrap-after",
        type=int,
        default=0,
        help=(
            "Minimum simulated tree depth before a distilled value evaluator "
            "may bootstrap. Default 0 preserves immediate value bootstrapping."
        ),
    )
    parser.add_argument("--min-teacher-mean", type=float, default=400.0)
    parser.add_argument("--train-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument(
        "--policy-checkpoint-path",
        default="checkpoints/state_mcts/state_policy_mcts_teacher.pt",
    )
    parser.add_argument(
        "--value-checkpoint-path",
        default="checkpoints/state_mcts/state_value_mcts_teacher.pt",
    )
    parser.add_argument(
        "--report-path",
        default="artifacts/state_mcts/mcts_distillation_report.json",
    )
    parser.add_argument(
        "--loss-dir",
        default="artifacts/state_mcts/losses",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
