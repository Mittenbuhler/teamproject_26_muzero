import argparse

import gymnasium as gym
import numpy as np
import torch

from dynamics_model import DynamicsModel
from replay_buffer import ReplayBuffer


def one_hot_action(action, action_dim):
    """
    Convert integer action into one-hot vector.

    Example:
        action=0 -> [1,0]
        action=1 -> [0,1]
    """

    vec = np.zeros(action_dim, dtype=np.float32)
    vec[action] = 1.0

    return vec


def collect_transitions(
    env,
    replay_buffer,
    n_episodes=100,
    max_steps=500,
    epsilon=1.0,
    log_every=25
):
    """
    Collect transitions from the environment and store them in the replay buffer.

    epsilon=1.0 gives fully random data. Lower values bias toward action 0 for a
    tiny deterministic baseline, but for CartPole random exploration is enough
    to learn one-step dynamics.
    """

    action_dim = env.action_space.n
    collected = 0
    episode_rewards = []
    episode_lengths = []

    for episode in range(1, n_episodes + 1):
        state, _ = env.reset()

        done = False
        step = 0
        episode_reward = 0.0

        while not done and step < max_steps:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = 0

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay_buffer.add(
                state,
                one_hot_action(action, action_dim),
                next_state,
                reward
            )

            state = next_state
            step += 1
            collected += 1
            episode_reward += reward

        episode_rewards.append(episode_reward)
        episode_lengths.append(step)

        if log_every and episode % log_every == 0:
            recent_rewards = episode_rewards[-log_every:]
            recent_lengths = episode_lengths[-log_every:]
            print(
                "collect "
                f"episode={episode:4d}/{n_episodes} | "
                f"transitions={collected:6d} | "
                f"avg_reward={np.mean(recent_rewards):7.2f} | "
                f"avg_length={np.mean(recent_lengths):7.2f}"
            )

    return collected


def train_dynamics_model(
    model,
    replay_buffer,
    optimizer,
    batch_size=64,
    train_steps=1000,
    device=None,
    log_every=100
):
    """
    Train the dynamics model on replayed one-step transitions.
    """

    if device is None:
        device = next(model.parameters()).device

    if len(replay_buffer) < batch_size:
        raise ValueError(
            f"Need at least batch_size={batch_size} transitions, got {len(replay_buffer)}"
        )

    model.train()

    running_total = 0.0
    running_state = 0.0
    running_reward = 0.0

    for step in range(1, train_steps + 1):
        states, actions, next_states, rewards = replay_buffer.sample(
            batch_size,
            device=device
        )

        optimizer.zero_grad()

        total_loss, state_loss, reward_loss = model.loss_details(
            states,
            actions,
            next_states,
            rewards
        )

        total_loss.backward()
        optimizer.step()

        running_total += total_loss.item()
        running_state += state_loss.item()
        running_reward += reward_loss.item()

        if log_every and step % log_every == 0:
            divisor = float(log_every)
            print(
                "step "
                f"{step:5d} | "
                f"loss={running_total / divisor:.6f} | "
                f"state={running_state / divisor:.6f} | "
                f"reward={running_reward / divisor:.6f}"
            )
            running_total = 0.0
            running_state = 0.0
            running_reward = 0.0


@torch.no_grad()
def evaluate_dynamics_model(
    model,
    replay_buffer,
    batch_size=256,
    device=None
):
    """
    Return average one-step prediction losses on a random replay batch.
    """

    if device is None:
        device = next(model.parameters()).device

    batch_size = min(batch_size, len(replay_buffer))
    states, actions, next_states, rewards = replay_buffer.sample(
        batch_size,
        device=device
    )

    model.eval()

    total_loss, state_loss, reward_loss = model.loss_details(
        states,
        actions,
        next_states,
        rewards
    )

    return {
        "loss": total_loss.item(),
        "state_loss": state_loss.item(),
        "reward_loss": reward_loss.item()
    }


@torch.no_grad()
def print_prediction_examples(
    model,
    replay_buffer,
    n_examples=3,
    device=None
):
    """
    Print a few one-step predictions next to the true transition targets.
    """

    if device is None:
        device = next(model.parameters()).device

    n_examples = min(n_examples, len(replay_buffer))
    states, actions, next_states, rewards = replay_buffer.sample(
        n_examples,
        device=device
    )

    model.eval()
    pred_next_states, pred_rewards = model(states, actions)

    print("prediction examples:")

    for index in range(n_examples):
        action = int(torch.argmax(actions[index]).item())
        true_state = next_states[index].detach().cpu().numpy()
        pred_state = pred_next_states[index].detach().cpu().numpy()
        true_reward = rewards[index].item()
        pred_reward = pred_rewards[index].item()

        print(
            f"  action={action} | "
            f"true_reward={true_reward:6.3f} | "
            f"pred_reward={pred_reward:6.3f}"
        )
        print(
            "    true_next_state="
            f"{np.array2string(true_state, precision=3, suppress_small=True)}"
        )
        print(
            "    pred_next_state="
            f"{np.array2string(pred_state, precision=3, suppress_small=True)}"
        )


def save_checkpoint(path, model, optimizer, args, metrics):
    """
    Save enough training state to reuse the learned dynamics model in MCTS.
    """

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "input_dim": model.input_dim,
            "output_dim": model.output_dim,
            "state_dim": args.state_dim,
            "action_dim": args.action_dim,
            "hidden_dim": args.hidden_dim,
            "metrics": metrics
        },
        path
    )


def load_dynamics_model(path, device=None):
    """
    Load a trained DynamicsModel checkpoint.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False
    )
    state_dim = checkpoint["state_dim"]
    action_dim = checkpoint["action_dim"]
    input_dim = checkpoint.get("input_dim", state_dim + action_dim)
    output_dim = checkpoint.get("output_dim", state_dim + 1)

    model = DynamicsModel(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=checkpoint["hidden_dim"],
        state_dim=state_dim,
        action_dim=action_dim
    ).to(device)

    model_state_dict = checkpoint["model_state_dict"]

    if "state_head.weight" in model_state_dict:
        model_state_dict = {
            "layer1.weight": model_state_dict["layer1.weight"],
            "layer1.bias": model_state_dict["layer1.bias"],
            "layer2.weight": model_state_dict["layer2.weight"],
            "layer2.bias": model_state_dict["layer2.bias"],
            "output_layer.weight": torch.cat(
                [
                    model_state_dict["state_head.weight"],
                    model_state_dict["reward_head.weight"]
                ],
                dim=0
            ),
            "output_layer.bias": torch.cat(
                [
                    model_state_dict["state_head.bias"],
                    model_state_dict["reward_head.bias"]
                ],
                dim=0
            )
        }

    model.load_state_dict(model_state_dict)
    model.eval()

    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a one-step CartPole dynamics model for MCTS."
    )

    parser.add_argument("--env", default="CartPole-v1")
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
    parser.add_argument("--save-path", default="dynamics_model_cartpole.pt")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(args.env)
    env.action_space.seed(args.seed)
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.n
    input_dim = args.state_dim + args.action_dim
    output_dim = args.state_dim + 1

    replay_buffer = ReplayBuffer(capacity=args.capacity)

    print(
        "training dynamics model | "
        f"env={args.env} | "
        f"device={device} | "
        f"input_dim={input_dim} | "
        f"output_dim={output_dim} | "
        f"episodes={args.episodes} | "
        f"train_steps={args.train_steps} | "
        f"batch_size={args.batch_size} | "
        f"lr={args.learning_rate}"
    )

    collected = collect_transitions(
        env,
        replay_buffer,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        log_every=args.collect_log_every
    )

    print("Collected transitions:", collected)

    model = DynamicsModel(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=args.hidden_dim,
        state_dim=args.state_dim,
        action_dim=args.action_dim
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
        "final | "
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

    save_checkpoint(
        args.save_path,
        model,
        optimizer,
        args,
        metrics
    )

    print("Saved checkpoint:", args.save_path)
