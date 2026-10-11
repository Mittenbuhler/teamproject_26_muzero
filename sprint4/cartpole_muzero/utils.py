from pathlib import Path

import numpy as np


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def one_hot(action, action_dim):
    vec = np.zeros(action_dim, dtype=np.float32)
    vec[int(action)] = 1.0
    return vec


def cartpole_terminal(state):
    x, _, theta, _ = np.asarray(state, dtype=np.float32)
    x_threshold = 2.4
    theta_threshold = 12 * 2 * np.pi / 360
    return bool(
        x < -x_threshold
        or x > x_threshold
        or theta < -theta_threshold
        or theta > theta_threshold
    )


def normalized_return(total_reward, max_reward=500.0):
    return float(np.clip(total_reward / max_reward, -1.0, 1.0))
