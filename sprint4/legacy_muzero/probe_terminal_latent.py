"""Test whether frozen latent state + action can predict CartPole termination."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .diagnose_latent_muzero import collect_diagnostic_episodes, finite_float
from .train_policy_value import load_latent_checkpoint
from .utils import ensure_dir


class TerminalProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(1)


def episode_arrays(trajectories, episode_indices, latent_mean, latent_std):
    features = []
    labels = []
    for episode_index in episode_indices:
        for transition in trajectories[episode_index]:
            latent = (transition["latent"] - latent_mean) / latent_std
            action = np.zeros(2, dtype=np.float32)
            action[transition["action"]] = 1.0
            features.append(np.concatenate([latent, action]))
            labels.append(float(transition["terminated"]))
    return (
        torch.as_tensor(np.asarray(features), dtype=torch.float32),
        torch.as_tensor(np.asarray(labels), dtype=torch.float32),
    )


def roc_auc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positive = scores[labels]
    negative = scores[~labels]
    if not len(positive) or not len(negative):
        return None
    comparisons = positive[:, None] - negative[None, :]
    return finite_float(
        ((comparisons > 0).mean() + 0.5 * (comparisons == 0).mean())
    )


def average_precision(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if not labels.any():
        return None
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels)
    precision = true_positives / np.arange(1, len(labels) + 1)
    return finite_float(precision[sorted_labels].mean())


def threshold_metrics(labels, probabilities, threshold):
    labels = np.asarray(labels, dtype=bool)
    predicted = np.asarray(probabilities) >= threshold
    true_positive = int((predicted & labels).sum())
    false_positive = int((predicted & ~labels).sum())
    true_negative = int((~predicted & ~labels).sum())
    false_negative = int((~predicted & labels).sum())
    recall = true_positive / max(true_positive + false_negative, 1)
    false_positive_rate = false_positive / max(false_positive + true_negative, 1)
    precision = true_positive / max(true_positive + false_positive, 1)
    return {
        "threshold": finite_float(threshold),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "recall": finite_float(recall),
        "false_positive_rate": finite_float(false_positive_rate),
        "precision": finite_float(precision),
        "balanced_accuracy": finite_float(
            0.5 * (recall + true_negative / max(true_negative + false_positive, 1))
        ),
    }


def operating_point_for_recall(labels, probabilities, target_recall=0.8):
    labels = np.asarray(labels, dtype=bool)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    positive_scores = np.sort(probabilities[labels])[::-1]
    if not len(positive_scores):
        return None
    required = max(1, int(np.ceil(target_recall * len(positive_scores))))
    threshold = positive_scores[required - 1]
    return threshold_metrics(labels, probabilities, threshold)


def train_fold(
    trajectories,
    train_episodes,
    validation_episodes,
    steps,
    seed,
    device,
):
    train_latents = np.asarray(
        [
            transition["latent"]
            for episode in train_episodes
            for transition in trajectories[episode]
        ],
        dtype=np.float32,
    )
    latent_mean = train_latents.mean(axis=0)
    latent_std = np.maximum(train_latents.std(axis=0), 1e-4)
    train_features, train_labels = episode_arrays(
        trajectories,
        train_episodes,
        latent_mean,
        latent_std,
    )
    validation_features, validation_labels = episode_arrays(
        trajectories,
        validation_episodes,
        latent_mean,
        latent_std,
    )
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    validation_features = validation_features.to(device)
    positive_indices = torch.nonzero(train_labels > 0.5, as_tuple=False).flatten()
    negative_indices = torch.nonzero(train_labels < 0.5, as_tuple=False).flatten()
    if not len(positive_indices):
        raise ValueError("Training fold contains no terminal transitions.")

    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    probe = TerminalProbe(train_features.shape[1]).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    losses = []

    for _ in range(steps):
        positive_sample = positive_indices[
            torch.randint(
                len(positive_indices),
                (64,),
                generator=generator,
                device=positive_indices.device,
            )
        ]
        negative_sample = negative_indices[
            torch.randint(
                len(negative_indices),
                (64,),
                generator=generator,
                device=negative_indices.device,
            )
        ]
        indices = torch.cat([positive_sample, negative_sample])
        optimizer.zero_grad()
        logits = probe(train_features[indices])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            train_labels[indices],
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    probe.eval()
    with torch.no_grad():
        validation_probabilities = torch.sigmoid(
            probe(validation_features)
        ).cpu().numpy()
    labels = validation_labels.cpu().numpy()
    return {
        "train_episodes": len(train_episodes),
        "validation_episodes": len(validation_episodes),
        "train_transitions": int(len(train_labels)),
        "train_terminal_transitions": int(train_labels.sum().item()),
        "validation_transitions": int(len(validation_labels)),
        "validation_terminal_transitions": int(validation_labels.sum().item()),
        "initial_loss_mean_20": finite_float(np.mean(losses[:20])),
        "final_loss_mean_20": finite_float(np.mean(losses[-20:])),
        "roc_auc": roc_auc(labels, validation_probabilities),
        "average_precision": average_precision(labels, validation_probabilities),
        "threshold_0_5": threshold_metrics(labels, validation_probabilities, 0.5),
        "operating_point_at_80_percent_recall": operating_point_for_recall(
            labels,
            validation_probabilities,
            target_recall=0.8,
        ),
        "ordinary_probability": {
            "mean": finite_float(validation_probabilities[labels < 0.5].mean()),
            "max": finite_float(validation_probabilities[labels < 0.5].max()),
        },
        "terminal_probability": {
            "mean": finite_float(validation_probabilities[labels > 0.5].mean()),
            "min": finite_float(validation_probabilities[labels > 0.5].min()),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe whether frozen MuZero latents predict termination."
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/legacy_muzero/latent_muzero_cartpole.pt",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--train-steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default="artifacts/legacy_muzero/diagnostics/terminal_probe.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    representation, dynamics, policy, value, checkpoint = load_latent_checkpoint(
        args.checkpoint,
        device=device,
    )
    trajectories = collect_diagnostic_episodes(
        checkpoint,
        representation,
        dynamics,
        policy,
        value,
        episodes=args.episodes,
        simulations=args.simulations,
        seed=args.seed,
        terminal_penalty=checkpoint.get("terminal_penalty", -25.0),
        device=device,
    )
    episode_indices = np.arange(len(trajectories))
    np.random.default_rng(args.seed).shuffle(episode_indices)
    folds = np.array_split(episode_indices, args.folds)
    results = []
    for fold_index, validation_episodes in enumerate(folds):
        train_episodes = np.concatenate(
            [fold for index, fold in enumerate(folds) if index != fold_index]
        )
        result = train_fold(
            trajectories,
            train_episodes.tolist(),
            validation_episodes.tolist(),
            steps=args.train_steps,
            seed=args.seed + fold_index,
            device=device,
        )
        result["fold"] = fold_index + 1
        results.append(result)
        print(
            f"terminal probe fold={fold_index + 1}/{args.folds} "
            f"auc={result['roc_auc']:.4f} "
            f"ap={result['average_precision']:.4f} "
            f"fpr@80recall="
            f"{result['operating_point_at_80_percent_recall']['false_positive_rate']:.4f}"
        )

    aucs = [result["roc_auc"] for result in results]
    false_positive_rates = [
        result["operating_point_at_80_percent_recall"]["false_positive_rate"]
        for result in results
    ]
    passed = bool(
        np.mean(aucs) >= 0.9
        and np.min(aucs) >= 0.8
        and np.mean(false_positive_rates) <= 0.1
    )
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "episodes": args.episodes,
        "simulations": args.simulations,
        "folds": results,
        "summary": {
            "mean_roc_auc": finite_float(np.mean(aucs)),
            "minimum_roc_auc": finite_float(np.min(aucs)),
            "mean_false_positive_rate_at_80_percent_recall": finite_float(
                np.mean(false_positive_rates)
            ),
            "pass_criteria": {
                "mean_roc_auc_at_least": 0.9,
                "minimum_fold_roc_auc_at_least": 0.8,
                "mean_fpr_at_80_percent_recall_at_most": 0.1,
            },
            "passed": passed,
        },
    }
    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print("saved terminal probe:", output_path.resolve())


if __name__ == "__main__":
    main()
