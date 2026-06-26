import numpy as np

from alphazero.replay_buffer import ReplayBuffer


def test_replay_buffer_respects_max_size():
    """Older experiences should be discarded when the fixed buffer is full."""
    buffer = ReplayBuffer(buffer_size=3, batch_size=2)

    for index in range(5):
        buffer.add(
            observation=np.array([index], dtype=np.float32),
            value=index,
            prev_obs=np.array([index - 1], dtype=np.float32),
            policy=np.array([0.5, 0.5], dtype=np.float32),
        )

    assert len(buffer) == 3
    assert [experience.value for experience in buffer.memory] == [2, 3, 4]


def test_replay_buffer_sample_returns_configured_batch_size():
    """Sampling should return exactly the batch size used for NN training."""
    buffer = ReplayBuffer(buffer_size=10, batch_size=4)

    for index in range(6):
        buffer.add(
            observation=np.array([index], dtype=np.float32),
            value=float(index),
            prev_obs=np.array([index], dtype=np.float32),
            policy=np.array([1.0, 0.0], dtype=np.float32),
        )

    sample = buffer.sample()

    assert len(sample) == 4
    assert all(experience.policy.shape == (2,) for experience in sample)
