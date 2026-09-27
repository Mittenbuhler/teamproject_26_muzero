import random
import numpy as np
import torch


class ReplayBuffer:
    """
    Simple replay buffer for dynamics model training.

    Stores transitions:

        (state, action, next_state, reward)
    """

    def __init__(self, capacity=100000):

        self.capacity = capacity

        self.buffer = []

        self.position = 0

    def add(
        self,
        state,
        action,
        next_state,
        reward
    ):
        """
        Add transition to replay buffer.
        """

        transition = (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            np.float32(reward)
        )

        # buffer not full yet
        if len(self.buffer) < self.capacity:

            self.buffer.append(transition)

        # overwrite oldest transition
        else:

            self.buffer[self.position] = transition

        # circular pointer
        self.position = (
            self.position + 1
        ) % self.capacity

    def sample(self, batch_size, device=None):
        """
        Sample random batch from buffer.

        Returns:
            states
            actions
            next_states
            rewards

        as PyTorch tensors.
        """

        if batch_size > len(self.buffer):
            raise ValueError(
                f"Cannot sample batch_size={batch_size} from buffer with {len(self.buffer)} transitions"
            )

        batch = random.sample(
            self.buffer,
            batch_size
        )

        states = []
        actions = []
        next_states = []
        rewards = []

        for s, a, s_next, r in batch:

            states.append(s)
            actions.append(a)
            next_states.append(s_next)
            rewards.append(r)

        states = torch.tensor(
            np.array(states),
            dtype=torch.float32
        )

        actions = torch.tensor(
            np.array(actions),
            dtype=torch.float32
        )

        next_states = torch.tensor(
            np.array(next_states),
            dtype=torch.float32
        )

        rewards = torch.tensor(
            np.array(rewards),
            dtype=torch.float32
        )

        if device is not None:
            states = states.to(device)
            actions = actions.to(device)
            next_states = next_states.to(device)
            rewards = rewards.to(device)

        return (
            states,
            actions,
            next_states,
            rewards
        )

    def __len__(self):
        """
        Allows:
            len(buffer)
        """

        return len(self.buffer)

    def clear(self):
        """
        Remove all transitions.
        """

        self.buffer = []

        self.position = 0
