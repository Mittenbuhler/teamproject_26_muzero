import gymnasium as gym
import numpy as np
from training import train


class OneHotWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        one_hot = np.zeros(16, dtype=np.float32)
        one_hot[obs] = 1.0
        return one_hot


def make_env():
    """Create FrozenLake-v1 environment (non-slippery) and wrap it."""
    env = gym.make("FrozenLake-v1", is_slippery=False)
    return OneHotWrapper(env)


def main():
    """Build config and call train()."""

    config = {
        "game_name": "FrozenLake-v1",
        "input_dim": 16,
        "action_dim": 4,
        "max_reward": 1,
        "episodes": 100,
        "make_env": make_env,
    }

    train(config)


if __name__ == "__main__":
    main()