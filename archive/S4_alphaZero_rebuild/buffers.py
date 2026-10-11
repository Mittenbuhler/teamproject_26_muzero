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


class PolicyValueReplayBuffer:
    """Stores MCTS targets for policy/value training."""

    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)
        self.experience = namedtuple("Experience", ["state", "policy", "value"])

    def add(self, state, policy, value):
        self.memory.append(
            self.experience(
                np.asarray(state, dtype=np.float32),
                np.asarray(policy, dtype=np.float32),
                np.float32(value),
            )
        )

    def sample(self, batch_size, device=None):
        if batch_size > len(self.memory):
            raise ValueError(f"Cannot sample {batch_size} from {len(self.memory)} targets")

        batch = random.sample(self.memory, batch_size)
        states = torch.as_tensor(np.asarray([e.state for e in batch]), dtype=torch.float32)
        policies = torch.as_tensor(np.asarray([e.policy for e in batch]), dtype=torch.float32)
        values = torch.as_tensor(np.asarray([[e.value] for e in batch]), dtype=torch.float32)

        if device is not None:
            states = states.to(device)
            policies = policies.to(device)
            values = values.to(device)

        return states, policies, values

    def __len__(self):
        return len(self.memory)
