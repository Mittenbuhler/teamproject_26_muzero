from collections import deque, namedtuple
import random

import numpy as np
import torch


class DynamicsReplayBuffer:
    """Stores one-step transitions for dynamics model training."""

    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action_one_hot, next_state, reward):
        transition = (
            np.asarray(state, dtype=np.float32),
            np.asarray(action_one_hot, dtype=np.float32),
            np.asarray(next_state, dtype=np.float32),
            np.float32(reward),
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, device=None):
        if batch_size > len(self.buffer):
            raise ValueError(f"Cannot sample {batch_size} from {len(self.buffer)} transitions")

        batch = random.sample(self.buffer, batch_size)
        states, actions, next_states, rewards = zip(*batch)

        states = torch.as_tensor(np.asarray(states), dtype=torch.float32)
        actions = torch.as_tensor(np.asarray(actions), dtype=torch.float32)
        next_states = torch.as_tensor(np.asarray(next_states), dtype=torch.float32)
        rewards = torch.as_tensor(np.asarray(rewards), dtype=torch.float32).unsqueeze(1)

        if device is not None:
            states = states.to(device)
            actions = actions.to(device)
            next_states = next_states.to(device)
            rewards = rewards.to(device)

        return states, actions, next_states, rewards

    def __len__(self):
        return len(self.buffer)


class LatentReplayBuffer:
    """Stores complete episodes and samples contiguous MuZero unrolls."""

    def __init__(self, capacity=20000):
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        self.episodes = deque()
        self.size = 0
        self.experience = namedtuple(
            "LatentExperience",
            [
                "observation",
                "action",
                "next_observation",
                "policy",
                "value",
                "reward",
                "terminated",
            ],
        )

    def add_episode(self, transitions):
        episode = tuple(
            self.experience(
                np.asarray(transition["observation"], dtype=np.float32),
                int(transition["action"]),
                np.asarray(transition["next_observation"], dtype=np.float32),
                np.asarray(transition["policy"], dtype=np.float32),
                np.float32(transition["value"]),
                np.float32(transition["reward"]),
                np.float32(transition.get("terminated", False)),
            )
            for transition in transitions
        )
        if not episode:
            return
        if len(episode) > self.capacity:
            episode = episode[-self.capacity:]

        self.episodes.append(episode)
        self.size += len(episode)
        while self.size > self.capacity and len(self.episodes) > 1:
            self.size -= len(self.episodes.popleft())

    def sample(self, batch_size, unroll_steps, action_dim, device=None):
        if unroll_steps <= 0:
            raise ValueError("unroll_steps must be positive")
        if batch_size > self.size:
            raise ValueError(f"Cannot sample {batch_size} from {self.size} targets")

        episodes = list(self.episodes)
        sampled_episodes = random.choices(
            episodes,
            weights=[len(episode) for episode in episodes],
            k=batch_size,
        )
        starts = [random.randrange(len(episode)) for episode in sampled_episodes]

        observations = []
        next_observations = []
        actions = np.zeros((batch_size, unroll_steps, action_dim), dtype=np.float32)
        policies = np.zeros((batch_size, unroll_steps, action_dim), dtype=np.float32)
        values = np.zeros((batch_size, unroll_steps, 1), dtype=np.float32)
        rewards = np.zeros((batch_size, unroll_steps, 1), dtype=np.float32)
        terminals = np.zeros((batch_size, unroll_steps, 1), dtype=np.float32)
        masks = np.zeros((batch_size, unroll_steps, 1), dtype=np.float32)

        for batch_index, (episode, start) in enumerate(zip(sampled_episodes, starts)):
            observations.append(episode[start].observation)
            padded_next_observation = np.zeros_like(episode[start].next_observation)
            episode_next_observations = []
            for step in range(unroll_steps):
                index = start + step
                if index < len(episode):
                    experience = episode[index]
                    padded_next_observation = experience.next_observation
                    actions[batch_index, step, experience.action] = 1.0
                    policies[batch_index, step] = experience.policy
                    values[batch_index, step, 0] = experience.value
                    rewards[batch_index, step, 0] = experience.reward
                    terminals[batch_index, step, 0] = experience.terminated
                    masks[batch_index, step, 0] = 1.0
                episode_next_observations.append(padded_next_observation)
            next_observations.append(episode_next_observations)

        tensors = [
            torch.as_tensor(array, dtype=torch.float32)
            for array in (
                np.asarray(observations),
                np.asarray(actions),
                np.asarray(next_observations),
                np.asarray(policies),
                np.asarray(values),
                np.asarray(rewards),
                np.asarray(terminals),
                np.asarray(masks),
            )
        ]

        if device is not None:
            tensors = [tensor.to(device) for tensor in tensors]

        return tuple(tensors)

    def __len__(self):
        return self.size
