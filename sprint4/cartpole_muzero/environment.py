"""Native CartPole observations; rendering is used only for optional GIFs."""

import os

import gymnasium as gym
import numpy as np


class StateObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Box) or env.observation_space.shape != (4,):
            raise ValueError("State MuZero requires a four-dimensional observation space")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("State MuZero requires a discrete action space")

    def observation(self, observation):
        state = np.asarray(observation, dtype=np.float32)
        if state.shape != (4,) or not np.isfinite(state).all():
            raise ValueError("Expected four finite CartPole state variables")
        return state.copy()


def make_state_env(env_id="CartPole-v1", render_mode=None):
    if render_mode is not None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    env = gym.make(env_id, render_mode=render_mode)
    try:
        return StateObservationWrapper(env)
    except Exception:
        env.close()
        raise
