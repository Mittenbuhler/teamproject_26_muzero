import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from dynamics_model import DynamicsModel
from mcts_agent import MCTSAgent
from replay_buffer import ReplayBuffer
from train_dynamics import (
    evaluate_dynamics_model,
    load_dynamics_model,
    save_checkpoint,
    train_dynamics_model
)


ACTION_NAMES = {
    0: "LEFT",
    1: "DOWN",
    2: "RIGHT",
    3: "UP"
}


def make_env(args, render_mode=None):
    return gym.make(
        "FrozenLake-v1",
        map_name=args.map_name,
        is_slippery=args.slippery,
        render_mode=render_mode
    )


def one_hot(index, size):
    vector = np.zeros(size, dtype=np.float32)
    vector[int(index)] = 1.0

    return vector


def is_terminal_state(env, state):
    desc = env.unwrapped.desc
    n_cols = desc.shape[1]
    row = int(state) // n_cols
    col = int(state) % n_cols

    return desc[row, col] in (b"H", b"G")


def collect_frozenlake_transitions(env, replay_buffer, repeats=200, log_every=25):
    """
    Collect one-step FrozenLake transitions from the environment transition table.

    For deterministic FrozenLake this exactly describes the environment. For
    slippery FrozenLake, repeated entries approximate the transition
    probabilities while the model still predicts one most likely next state.
    """

    state_dim = env.observation_space.n
    action_dim = env.action_space.n
    collected = 0

    for repeat in range(1, repeats + 1):
        for state in range(state_dim):
            for action in range(action_dim):
                transitions = env.unwrapped.P[state][action]

                for probability, next_state, reward, _ in transitions:
                    copies = max(1, round(probability * 10))

                    for _ in range(copies):
                        replay_buffer.add(
                            one_hot(state, state_dim),
                            one_hot(action, action_dim),
                            one_hot(next_state, state_dim),
                            reward
                        )
                        collected += 1

        if log_every and repeat % log_every == 0:
            print(
                "collect "
                f"pass={repeat:4d}/{repeats} | "
                f"transitions={collected:7d}"
            )

    return collected


def print_prediction_table(model, env, device):
    state_dim = env.observation_space.n
    action_dim = env.action_space.n

    print("prediction table:")

    with torch.no_grad():
        for state in range(state_dim):
            for action in range(action_dim):
                true_transitions = env.unwrapped.P[state][action]
                best_true = max(true_transitions, key=lambda item: item[0])
                _, true_next_state, true_reward, _ = best_true

                pred_next_state, pred_reward = model.predict(
                    state,
                    action,
                    device=device
                )

                print(
                    f"  s={state:2d} a={ACTION_NAMES[action]:5s} | "
                    f"true_next={true_next_state:2d} | "
                    f"pred_next={pred_next_state:2d} | "
                    f"true_reward={true_reward:.1f} | "
                    f"pred_reward={pred_reward:.3f}"
                )


def train_model(args, device):
    env = make_env(args)
    state_dim = env.observation_space.n
    action_dim = env.action_space.n
    input_dim = state_dim + action_dim
    output_dim = state_dim + 1

    replay_buffer = ReplayBuffer(capacity=args.capacity)

    print(
        "training FrozenLake dynamics model | "
        f"map={args.map_name} | "
        f"slippery={args.slippery} | "
        f"device={device} | "
        f"input_dim={input_dim} | "
        f"output_dim={output_dim}"
    )

    collected = collect_frozenlake_transitions(
        env,
        replay_buffer,
        repeats=args.collect_repeats,
        log_every=args.collect_log_every
    )

    print("collected transitions:", collected)

    model = DynamicsModel(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=args.hidden_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        discrete_state=True
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

    if args.print_predictions:
        print_prediction_table(model, env, device)

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


def run_episode(args, model, episode):
    render_mode = "human" if args.render else None
    env = make_env(args, render_mode=render_mode)
    env.action_space.seed(args.seed + episode)

    clone_factory = lambda: make_env(args, render_mode=None)
    terminal_fn = lambda state: is_terminal_state(env, state)

    agent = MCTSAgent(
        game_name="FrozenLake-v1",
        env_factory=clone_factory,
        explore_iterations=args.explore_iterations,
        c=args.exploration,
        dynamics_model=model,
        terminal_fn=terminal_fn,
        discount=args.discount
    )

    observation, _ = env.reset(seed=args.seed + episode)
    done = False
    total_reward = 0.0
    step = 0

    while not done and step < args.max_steps:
        action = agent.get_action(env, observation, done)
        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1

        print(
            f"episode={episode + 1:3d} | "
            f"step={step:3d} | "
            f"state={observation:2d} | "
            f"action={ACTION_NAMES[action]:5s} | "
            f"reward={reward:.1f} | "
            f"done={done}"
        )

        if args.render:
            time.sleep(args.render_delay)

    env.close()

    success = total_reward > 0
    print(
        f"episode={episode + 1:3d} finished | "
        f"steps={step:3d} | "
        f"total_reward={total_reward:.1f} | "
        f"success={success}"
    )

    return success, total_reward, step


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/load a FrozenLake dynamics model and use it in MCTS."
    )

    parser.add_argument("--model-path", default="dynamics_model_frozenlake.pt")
    parser.add_argument("--train-first", action="store_true")
    parser.add_argument("--force-train", action="store_true")

    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--map-name", default="4x4")
    parser.add_argument("--slippery", action="store_true")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-delay", type=float, default=0.25)

    parser.add_argument("--capacity", type=int, default=100000)
    parser.add_argument("--collect-repeats", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--collect-log-every", type=int, default=25)
    parser.add_argument("--print-predictions", action="store_true")

    parser.add_argument("--explore-iterations", type=int, default=500)
    parser.add_argument("--exploration", type=float, default=1.0)
    parser.add_argument("--discount", type=float, default=0.95)

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

    successes = 0
    rewards = []
    steps = []

    for episode in range(args.episodes):
        success, reward, step_count = run_episode(args, model, episode)
        successes += int(success)
        rewards.append(reward)
        steps.append(step_count)

    print(
        "summary | "
        f"episodes={args.episodes} | "
        f"successes={successes} | "
        f"success_rate={successes / args.episodes:.2f} | "
        f"avg_reward={sum(rewards) / len(rewards):.2f} | "
        f"avg_steps={sum(steps) / len(steps):.2f}"
    )
