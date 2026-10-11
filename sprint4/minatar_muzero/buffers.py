"""Complete-episode replay with correctly aligned vanilla MuZero targets."""

from collections import deque
import random

import numpy as np
import torch


class EpisodeReplayBuffer:
    def __init__(self, capacity=20000):
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        self.episodes = deque()
        self.size = 0

    def add_episode(self, transitions):
        episode = []
        for item in transitions:
            episode.append(
                {
                    "observation": np.asarray(item["observation"], np.float32),
                    "action": int(item["action"]),
                    "reward": float(item["reward"]),
                    "policy": np.asarray(item["policy"], np.float32),
                    "value": float(item["value"]),
                    "policy_valid": bool(item.get("policy_valid", True)),
                    "terminated": bool(item.get("terminated", False)),
                    "truncated": bool(item.get("truncated", False)),
                }
            )
        if not episode:
            return
        self.episodes.append(tuple(episode))
        self.size += len(episode)
        self._trim()

    def _trim(self):
        # Keep at least one complete episode, even if that single episode is
        # longer than the nominal transition capacity.
        while self.size > self.capacity and len(self.episodes) > 1:
            self.size -= len(self.episodes.popleft())

    def sample(
        self,
        batch_size,
        unroll_steps,
        action_dim,
        device=None,
        starts=None,
    ):
        """Sample one root plus K recurrent targets without crossing episodes.

        ``rewards[:, k]`` is r_(t+k+1). ``policies/values[:, k+1]`` are
        pi/z_(t+k+1). A transition's ``policy_valid`` flag masks policy only;
        warm-up reward/value model learning remains active.
        """
        if not self.episodes:
            raise ValueError("Cannot sample an empty replay buffer")
        batch_size = int(batch_size)
        unroll_steps = int(unroll_steps)
        action_dim = int(action_dim)
        if batch_size <= 0 or unroll_steps < 0 or action_dim <= 0:
            raise ValueError(
                "batch_size/action_dim must be positive and unroll_steps nonnegative"
            )

        episodes = random.choices(
            list(self.episodes),
            weights=[len(episode) for episode in self.episodes],
            k=batch_size,
        )
        if starts is None:
            starts = [random.randrange(len(episode)) for episode in episodes]
        if len(starts) != batch_size:
            raise ValueError("starts must contain one index per batch element")

        observations = []
        actions = np.zeros((batch_size, unroll_steps), np.int64)
        policies = np.zeros(
            (batch_size, unroll_steps + 1, action_dim),
            np.float32,
        )
        values = np.zeros((batch_size, unroll_steps + 1, 1), np.float32)
        rewards = np.zeros((batch_size, unroll_steps, 1), np.float32)
        policy_masks = np.zeros_like(values)
        # Absorbing post-terminal states are explicitly trained toward zero
        # value/reward, so these masks deliberately remain active.
        value_masks = np.ones_like(values)
        reward_masks = np.ones_like(rewards)

        for batch_index, (episode, start) in enumerate(zip(episodes, starts)):
            start = int(start)
            if not 0 <= start < len(episode):
                raise ValueError("sample start is outside its episode")
            observations.append(episode[start]["observation"])
            alive = True
            for depth in range(unroll_steps + 1):
                index = start + depth
                if alive and index < len(episode):
                    transition = episode[index]
                    policy = transition["policy"]
                    if policy.shape != (action_dim,):
                        raise ValueError(
                            f"stored policy shape {policy.shape} does not match "
                            f"action_dim={action_dim}"
                        )
                    policies[batch_index, depth] = policy
                    values[batch_index, depth, 0] = transition["value"]
                    policy_masks[batch_index, depth, 0] = float(
                        transition.get("policy_valid", True)
                    )

                if depth < unroll_steps:
                    if alive and index < len(episode):
                        transition = episode[index]
                        actions[batch_index, depth] = transition["action"]
                        rewards[batch_index, depth, 0] = transition["reward"]
                        if transition["terminated"] or transition["truncated"]:
                            alive = False
                    else:
                        # Fixed dummy action for absorbing recurrent padding.
                        actions[batch_index, depth] = 0

        arrays = (
            np.asarray(observations),
            actions,
            policies,
            values,
            rewards,
            policy_masks,
            value_masks,
            reward_masks,
        )
        return tuple(torch.as_tensor(array, device=device) for array in arrays)

    def state_dict(self):
        return {
            "capacity": self.capacity,
            "episodes": list(self.episodes),
            "size": self.size,
        }

    def load_state_dict(self, state):
        self.capacity = int(state["capacity"])
        if self.capacity <= 0:
            raise ValueError("saved replay capacity must be positive")
        self.episodes = deque(
            tuple(episode) for episode in state.get("episodes", [])
        )
        self.size = int(
            state.get("size", sum(len(episode) for episode in self.episodes))
        )

    def resize(self, capacity):
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        self._trim()

    def __len__(self):
        return self.size


# Compatibility alias for legacy-informed callers and saved training code.
LatentReplayBuffer = EpisodeReplayBuffer
