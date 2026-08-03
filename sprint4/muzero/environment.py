"""Native MinAtar observations and lightweight, headless rendering."""

from __future__ import annotations

from collections import deque
import os
import re
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "minatar_muzero_matplotlib"),
)

import gymnasium as gym
import numpy as np
from PIL import Image


def minatar_env_id(game: str) -> str:
    """Return a Gymnasium MinAtar v1 id for a short or fully qualified name.

    ``breakout``, ``space_invaders`` and ``MinAtar/Seaquest-v1`` are accepted.
    Full ids are kept verbatim, which also lets callers explicitly request a v0
    six-action environment.
    """
    name = str(game).strip()
    if not name:
        raise ValueError("game must not be empty")
    if name.lower().startswith("minatar/"):
        name = name.split("/", 1)[1]

    match = re.fullmatch(r"(.+?)(?:-v([01]))?", name, flags=re.IGNORECASE)
    if match is None:  # pragma: no cover - guarded by the non-empty check
        raise ValueError(f"invalid MinAtar game name: {game!r}")
    base, version = match.group(1), match.group(2) or "1"
    words = [word for word in re.split(r"[-_\s]+", base) if word]
    compact = "".join(words).lower()
    official_names = {
        "asterix": "Asterix",
        "breakout": "Breakout",
        "freeway": "Freeway",
        "seaquest": "Seaquest",
        "spaceinvaders": "SpaceInvaders",
    }
    game_id = official_names.get(
        compact,
        "".join(word[:1].upper() + word[1:].lower() for word in words),
    )
    return f"MinAtar/{game_id}-v{version}"


def _native_to_chw(observation) -> np.ndarray:
    """Convert MinAtar's boolean HWC state into finite float32 CHW planes."""
    array = np.asarray(observation)
    if array.ndim != 3:
        raise ValueError(
            "MinAtar observations must have shape [height, width, channels], "
            f"got {array.shape}"
        )
    result = np.moveaxis(array, -1, 0).astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise ValueError("MinAtar observation contains non-finite values")
    if result.size and (float(result.min()) < 0.0 or float(result.max()) > 1.0):
        raise ValueError("MinAtar observations must be bounded in [0, 1]")
    return np.ascontiguousarray(result)


_CHANNEL_COLORS = np.asarray(
    [
        (238, 238, 238),
        (61, 169, 252),
        (255, 103, 103),
        (112, 224, 124),
        (255, 207, 84),
        (191, 128, 255),
        (255, 143, 61),
        (78, 224, 207),
        (247, 126, 190),
        (166, 196, 255),
        (205, 231, 118),
        (230, 158, 117),
    ],
    dtype=np.uint8,
)


def render_minatar_state(observation, cell_size: int = 32) -> np.ndarray:
    """Render native CHW feature planes without opening a GUI window."""
    state = np.asarray(observation)
    if state.ndim != 3:
        raise ValueError("state must have CHW dimensions")
    hwc = np.moveaxis(state, 0, -1)
    height, width, channels = hwc.shape
    active = hwc.astype(bool)
    rgb = np.full((height, width, 3), 12, dtype=np.uint8)
    for channel in range(channels):
        mask = active[..., channel]
        color = _CHANNEL_COLORS[channel % len(_CHANNEL_COLORS)]
        # Max-composition keeps collocated objects visible without game rules.
        rgb[mask] = np.maximum(rgb[mask], color)
    size = int(cell_size)
    if size <= 0:
        raise ValueError("cell_size must be positive")
    image = Image.fromarray(rgb, "RGB").resize(
        (width * size, height * size), Image.Resampling.NEAREST
    )
    return np.asarray(image)


class MinAtarAdapter:
    """Gymnasium MinAtar adapter exposing normalized channel-first states."""

    def __init__(
        self,
        game="breakout",
        render_cell_size=32,
        sticky_action_prob=0.1,
        difficulty_ramping=True,
    ):
        try:
            from minatar.gym import register_envs
        except ImportError as error:
            raise ImportError(
                "MinAtar is required. Install sprint4/requirements.txt in the "
                "project virtual environment (pip install minatar)."
            ) from error

        self.requested_game = str(game)
        self.env_id = minatar_env_id(game)
        # Gymnasium 1.x no longer auto-loads third-party registration entry
        # points. MinAtar 1.0.15 therefore needs one explicit registration.
        if self.env_id not in gym.envs.registry:
            register_envs()
        self.sticky_action_prob = float(sticky_action_prob)
        if not 0.0 <= self.sticky_action_prob <= 1.0:
            raise ValueError("sticky_action_prob must be in [0, 1]")
        self.difficulty_ramping = bool(difficulty_ramping)
        try:
            self._env = gym.make(
                self.env_id,
                render_mode="rgb_array",
                disable_env_checker=True,
                sticky_action_prob=self.sticky_action_prob,
                difficulty_ramping=self.difficulty_ramping,
            )
        except Exception as error:
            raise ValueError(
                f"Could not construct MinAtar environment {self.env_id!r} "
                f"from --game {game!r}: {error}"
            ) from error
        if not isinstance(self._env.action_space, gym.spaces.Discrete):
            self._env.close()
            raise ValueError(f"{self.env_id} must expose a discrete action space")
        raw_shape = tuple(int(size) for size in self._env.observation_space.shape)
        if len(raw_shape) != 3:
            self._env.close()
            raise ValueError(
                f"{self.env_id} must expose an HWC observation, got {raw_shape}"
            )
        self.base_observation_shape = (raw_shape[2], raw_shape[0], raw_shape[1])
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=self.base_observation_shape,
            dtype=np.float32,
        )
        self.action_space = self._env.action_space
        self.render_cell_size = int(render_cell_size)
        if self.render_cell_size <= 0:
            self._env.close()
            raise ValueError("render_cell_size must be positive")
        self._last_observation = None

    @property
    def action_dim(self):
        return int(self.action_space.n)

    def reset(self, *, seed=None, options=None):
        observation, info = self._env.reset(seed=seed, options=options)
        self._last_observation = _native_to_chw(observation)
        return self._last_observation.copy(), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self._env.step(int(action))
        self._last_observation = _native_to_chw(observation)
        return (
            self._last_observation.copy(),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def render(self):
        if self._last_observation is None:
            raise RuntimeError("reset the environment before rendering")
        frame = self._env.render() if hasattr(self._env, "render") else None
        if frame is None:
            return render_minatar_state(
                self._last_observation,
                cell_size=self.render_cell_size,
            )
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[-1] not in (3, 4):
            raise ValueError("MinAtar rgb_array render must be HWC RGB/RGBA")
        if np.issubdtype(array.dtype, np.floating) and (
            float(array.max()) if array.size else 0.0
        ) <= 1.0:
            array = array * 255.0
        image = Image.fromarray(
            np.clip(array[..., :3], 0, 255).astype(np.uint8),
            "RGB",
        )
        height, width = array.shape[:2]
        image = image.resize(
            (width * self.render_cell_size, height * self.render_cell_size),
            Image.Resampling.NEAREST,
        )
        return np.asarray(image)

    def close(self):
        self._env.close()


def make_minatar_env(game="breakout", **kwargs):
    return MinAtarAdapter(game=game, **kwargs)


class ObservationHistory:
    """Concatenate a configurable history of native CHW observations."""

    def __init__(self, history_length=1, observation_shape=(4, 10, 10)):
        if int(history_length) <= 0:
            raise ValueError("history_length must be positive")
        self.history_length = int(history_length)
        self.observation_shape = tuple(int(size) for size in observation_shape)
        if len(self.observation_shape) != 3 or any(
            size <= 0 for size in self.observation_shape
        ):
            raise ValueError("observation_shape must be positive (channels, height, width)")
        self.frames = deque(maxlen=self.history_length)

    @property
    def output_shape(self):
        channels, height, width = self.observation_shape
        return channels * self.history_length, height, width

    @property
    def real_frame_count(self):
        return len(self.frames)

    def reset(self, observation):
        self.frames.clear()
        return self.append(observation)

    def append(self, observation):
        frame = np.asarray(observation, dtype=np.float32)
        if frame.shape != self.observation_shape:
            raise ValueError(
                f"Expected observation shape {self.observation_shape}, got {frame.shape}"
            )
        if not np.isfinite(frame).all():
            raise ValueError("observation must contain only finite values")
        self.frames.append(frame.copy())
        return self.observation()

    def observation(self):
        missing = self.history_length - len(self.frames)
        frames = [
            np.zeros(self.observation_shape, dtype=np.float32)
            for _ in range(missing)
        ]
        frames.extend(self.frames)
        return np.concatenate(frames, axis=0)
