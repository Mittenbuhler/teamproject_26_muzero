import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from buffers import DynamicsReplayBuffer
from models import DynamicsModel
from utils import ensure_dir, one_hot


def collect_transitions(env, buffer, episodes=200, max_steps=500, epsilon=1.0, seed=0):
    env.action_space.seed(seed)
    total = 0
    rewards = []

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        done = False
        episode_reward = 0.0
        steps = 0

        while not done and steps < max_steps:
            action = env.action_space.sample() if np.random.random() < epsilon else 0
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.add(state, one_hot(action, env.action_space.n), next_state, reward)
            state = next_state
            episode_reward += reward
            steps += 1
            total += 1

        rewards.append(episode_reward)

    return total, rewards


def train_dynamics_model(model, buffer, optimizer, batch_size=64, train_steps=1000, device=None, log_every=100):
    if device is None:
        device = next(model.parameters()).device

    history = []
    for step in range(1, train_steps + 1):
        states, actions, next_states, rewards = buffer.sample(batch_size, device=device)
        optimizer.zero_grad()
        total_loss, state_loss, reward_loss = model.loss_details(states, actions, next_states, rewards)
        total_loss.backward()
        optimizer.step()

        history.append((total_loss.item(), state_loss.item(), reward_loss.item()))
        if log_every and step % log_every == 0:
            recent = np.asarray(history[-log_every:])
            print(
                "dynamics "
                f"step={step:5d} "
                f"loss={recent[:, 0].mean():.6f} "
                f"state={recent[:, 1].mean():.6f} "
                f"reward={recent[:, 2].mean():.6f}"
            )

    return history


@torch.no_grad()
def evaluate_dynamics(model, buffer, batch_size=256, device=None):
    if device is None:
        device = next(model.parameters()).device
    batch_size = min(batch_size, len(buffer))
    states, actions, next_states, rewards = buffer.sample(batch_size, device=device)
    model.eval()
    total_loss, state_loss, reward_loss = model.loss_details(states, actions, next_states, rewards)
    return {
        "loss": total_loss.item(),
        "state_loss": state_loss.item(),
        "reward_loss": reward_loss.item(),
    }


def save_dynamics(path, model, metrics):
    ensure_dir(Path(path).parent)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "state_dim": model.state_dim,
            "action_dim": model.action_dim,
            "hidden_dim": model.hidden_dim,
            "metrics": metrics,
        },
        path,
    )


def load_dynamics(path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = DynamicsModel(
        checkpoint["state_dim"],
        checkpoint["action_dim"],
        checkpoint["hidden_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def train_or_load_dynamics(args, device):
    if args.load_path and Path(args.load_path).exists():
        print("loading dynamics:", args.load_path)
        return load_dynamics(args.load_path, device=device)

    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    buffer = DynamicsReplayBuffer(args.capacity)

    collected, rewards = collect_transitions(
        env,
        buffer,
        episodes=args.collect_episodes,
        max_steps=args.max_steps,
        epsilon=args.epsilon,
        seed=args.seed,
    )
    env.close()
    print(
        f"collected dynamics transitions={collected} "
        f"avg_reward={np.mean(rewards):.2f}"
    )

    model = DynamicsModel(state_dim, action_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train_dynamics_model(
        model,
        buffer,
        optimizer,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        device=device,
        log_every=args.log_every,
    )
    metrics = evaluate_dynamics(model, buffer, batch_size=args.batch_size, device=device)
    print("dynamics final:", metrics)

    if args.save_path:
        save_dynamics(args.save_path, model, metrics)
        print("saved dynamics:", args.save_path)

    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--collect-episodes", type=int, default=250)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--capacity", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=150)
    parser.add_argument("--save-path", default="checkpoints/dynamics_cartpole.pt")
    parser.add_argument("--load-path", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_or_load_dynamics(args, device)
