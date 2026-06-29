from dataclasses import dataclass
import os
from typing import Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from PIL import Image


CropBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class ScreenshotConfig:
    width: int = 84
    height: int = 84
    binary: bool = False
    threshold: int = 128
    normalize: bool = True
    channel_first: bool = True
    crop_box: Optional[CropBox] = None


class ScreenshotPreprocessor:
    """Turn RGB env renders into downscaled grayscale CNN observations."""

    def __init__(self, config: ScreenshotConfig = ScreenshotConfig()):
        self.config = config

    @property
    def output_shape(self):
        if self.config.channel_first:
            return (1, self.config.height, self.config.width)
        return (self.config.height, self.config.width, 1)

    def process(self, frame):
        if frame is None:
            raise ValueError("Expected a frame. Build envs with render_mode='rgb_array'.")

        image = Image.fromarray(frame).convert("L")
        if self.config.crop_box is not None:
            image = image.crop(self.config.crop_box)

        image = image.resize((self.config.width, self.config.height), Image.Resampling.BILINEAR)
        array = np.asarray(image)

        if self.config.binary:
            array = (array >= self.config.threshold).astype(np.float32)
        else:
            array = array.astype(np.float32)
            if self.config.normalize:
                array /= 255.0

        if self.config.channel_first:
            return array[np.newaxis, :, :]
        return array[:, :, np.newaxis]

    def __call__(self, frame):
        return self.process(frame)


class ScreenshotObservationWrapper(gym.Wrapper):
    """Optional wrapper that returns screenshots instead of vector observations."""

    def __init__(self, env, preprocessor=None):
        super().__init__(env)
        self.preprocessor = preprocessor or ScreenshotPreprocessor()
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=self.preprocessor.output_shape,
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        return self._render_observation(), info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        return self._render_observation(), reward, terminated, truncated, info

    def _render_observation(self):
        return self.preprocessor.process(self.env.render())


def make_image_env(env_id="CartPole-v1", screenshot_config=ScreenshotConfig(), headless=True):
    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    env = gym.make(env_id, render_mode="rgb_array")
    return ScreenshotObservationWrapper(env, ScreenshotPreprocessor(screenshot_config))


def stack_observations(observations: Sequence[np.ndarray]):
    if not observations:
        raise ValueError("Cannot stack an empty observation sequence.")
    return np.concatenate([np.asarray(obs, dtype=np.float32) for obs in observations], axis=0)
