from gym.envs.registration import register

from .core import FlappyBird, FlappyBirdConfig, FlappyBirdGame
from .envs import FlappyBirdEnvV0

__all__ = [
    "FlappyBirdEnvV0",
    "FlappyBirdConfig",
    "FlappyBird",
    "FlappyBirdGame",
    "core",
    "envs",
]

register(
    id="FlappyBird-v0",
    entry_point="flappy_bird.envs:FlappyBirdEnvV1",
)
