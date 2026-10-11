"""MinAtar observations built exclusively from headless RGB screenshots."""

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


DEFAULT_IMAGE_SIZE = 32
DEFAULT_HISTORY_LENGTH = 4


def screenshot_preprocessing(image_size):
    """Versioned by checkpoint format; describes the exact pixel input contract."""
    if int(image_size) <= 0:
        raise ValueError("image_size must be positive")
    return {
        "source": "render_rgb_array",
        "color_mode": "RGB",
        "image_size": int(image_size),
        "resize": "nearest",
        "normalization": "uint8/255",
    }


def _render_to_rgb(frame) -> np.ndarray:
    """Convert an RGB(A) render (uint8 or floating point) to RGB bytes."""
    if frame is None:
        raise ValueError("MinAtar render() returned no screenshot")
    array = np.asarray(frame)
    if (array.ndim != 3 or array.shape[-1] not in (3, 4)
            or not array.shape[0] or not array.shape[1]):
        raise ValueError("MinAtar rgb_array render must be a nonempty HWC RGB/RGBA frame")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("MinAtar screenshot must contain finite numeric pixels")
    if array.min() < 0 or array.max() > 255:
        raise ValueError("MinAtar screenshot pixels must be bounded in [0, 255]")
    if np.issubdtype(array.dtype, np.floating) and array.max() <= 1.0:
        array = array * 255.0
    return np.ascontiguousarray(np.rint(array[..., :3]).astype(np.uint8))


class MinAtarAdapter:
    """Discard native feature grids and expose only preprocessed render() pixels."""

    def __init__(
        self,
        game="breakout",
        render_cell_size=32,
        sticky_action_prob=0.1,
        difficulty_ramping=True,
        image_size=DEFAULT_IMAGE_SIZE,
    ):
        self.preprocessing = screenshot_preprocessing(image_size)
        self.image_size = self.preprocessing["image_size"]
        self.render_cell_size = int(render_cell_size)
        if self.render_cell_size <= 0:
            raise ValueError("render_cell_size must be positive")
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
        self.base_observation_shape = (3, self.image_size, self.image_size)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=self.base_observation_shape,
            dtype=np.float32,
        )
        self.action_space = self._env.action_space
        self._last_screenshot = None

    @property
    def action_dim(self):
        return int(self.action_space.n)

    def reset(self, *, seed=None, options=None):
        _, info = self._env.reset(seed=seed, options=options)
        return self._capture_observation(), info

    def step(self, action):
        _, reward, terminated, truncated, info = self._env.step(int(action))
        return (
            self._capture_observation(),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def _capture_observation(self):
        # The native observation returned by reset/step is deliberately unused.
        # Never synthesize pixels from feature planes if rendering fails.
        self._last_screenshot = None
        screenshot = _render_to_rgb(self._env.render())
        image = Image.fromarray(screenshot).resize(
            (self.image_size, self.image_size), Image.Resampling.NEAREST
        )
        self._last_screenshot = screenshot
        return np.ascontiguousarray(
            np.moveaxis(np.asarray(image), -1, 0), dtype=np.float32
        ) / 255.0

    def render(self):
        """Return the cached screenshot used for the latest model observation."""
        if self._last_screenshot is None:
            raise RuntimeError("reset the environment before rendering")
        image = Image.fromarray(self._last_screenshot)
        height, width = self._last_screenshot.shape[:2]
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
    """Concatenate RGB screenshots, oldest first, with initial black padding."""

    def __init__(
        self,
        history_length=DEFAULT_HISTORY_LENGTH,
        observation_shape=(3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
    ):
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
