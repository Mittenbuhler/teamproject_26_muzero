import gymnasium as gym
import numpy as np
from training import train


def make_env():
    """Create CartPole-v1 environment."""
    return gym.make("CartPole-v1")


def main():
    """Build config and call train()."""

    config = {
        "game_name": "CartPole-v1",
        "input_dim": 4,
        "action_dim": 2,
        "max_reward": 500,
        "episodes": 300,
        "make_env": make_env,
    }

    train(config)


if __name__ == "__main__":
    main()