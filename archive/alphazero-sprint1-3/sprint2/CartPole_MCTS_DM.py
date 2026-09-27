import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from dynamics_model import DynamicsModel
from mcts_agent import MCTSAgent
from replay_buffer import ReplayBuffer
from train_dynamics import (
    collect_transitions,
    evaluate_dynamics_model,
    load_dynamics_model,
    print_prediction_examples,
    save_checkpoint,
    train_dynamics_model
)


def train_model(args, device):
    env = gym.make(args.env)
    env.action_space.seed(args.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    input_dim = state_dim + action_dim
    output_dim = state_dim + 1

    replay_buffer = ReplayBuffer(capacity=args.capacity)

    print(
        "training dynamics model | "
        f"env={args.env} | "
        f"device={device} | "
        f"input_dim={input_dim} | "
        f"output_dim={output_dim} | "
        f"episodes={args.episodes} | "
        f"train_steps={args.train_steps}"
    )

    collected = collect_transitions(
        env,
        replay_buffer,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        log_every=args.collect_log_every
    )

    print("collected transitions:", collected)

    model = DynamicsModel(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=args.hidden_dim,
        state_dim=state_dim,
        action_dim=action_dim
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate
    )

    train_dynamics_model(
        model,
        replay_buffer,
        optimizer,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        device=device,
        log_every=args.log_every
    )

    metrics = evaluate_dynamics_model(
        model,
        replay_buffer,
        batch_size=args.batch_size,
        device=device
    )

    print(
        "final training metrics | "
        f"loss={metrics['loss']:.6f} | "
        f"state={metrics['state_loss']:.6f} | "
        f"reward={metrics['reward_loss']:.6f}"
    )

    print_prediction_examples(
        model,
        replay_buffer,
        n_examples=args.prediction_examples,
        device=device
    )

    args.input_dim = input_dim
    args.output_dim = output_dim
    args.state_dim = state_dim
    args.action_dim = action_dim

    save_checkpoint(
        args.model_path,
        model,
        optimizer,
        args,
        metrics
    )

    env.close()

    print("saved model:", args.model_path)

    return model


def run_cartpole(args, model):
    env = gym.make(args.env, render_mode="human")

    agent = MCTSAgent(
        game_name=args.env,
        explore_iterations=args.explore_iterations,
        dynamics_model=model
    )

    observation, _ = env.reset()
    done = False
    total_reward = 0.0
    step = 0

    while not done:
        action = agent.get_action(env, observation, done)
        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1

        print(
            f"mcts step={step:4d} | "
            f"action={action} | "
            f"reward={reward:5.2f} | "
            f"total_reward={total_reward:7.2f}"
        )

    env.close()

    print("episode finished | total_reward:", total_reward)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/load a CartPole dynamics model and use it in MCTS."
    )

    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--model-path", default="dynamics_model_cartpole.pt")
    parser.add_argument("--train-first", action="store_true")
    parser.add_argument("--force-train", action="store_true")

    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--capacity", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--collect-log-every", type=int, default=25)
    parser.add_argument("--prediction-examples", type=int, default=3)

    parser.add_argument("--explore-iterations", type=int, default=100)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(args.model_path)

    should_train = args.force_train or args.train_first or not model_path.exists()

    if should_train:
        model = train_model(args, device)
    else:
        print("loading existing model:", args.model_path)
        model = load_dynamics_model(args.model_path, device=device)

    run_cartpole(args, model)
