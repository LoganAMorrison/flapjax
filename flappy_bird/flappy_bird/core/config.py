import json
from typing import Optional, Tuple

import attrs
import cv2 as cv
from attrs import field, validators

CV_INTERPS = [
    cv.INTER_AREA,
    cv.INTER_CUBIC,
    cv.INTER_NEAREST,
    cv.INTER_LINEAR,
    cv.INTER_LANCZOS4,
]


def ge_or_none(value):
    def f(*args):
        val = args[-1]
        assert val is None or val >= value, f"Value must be None or >= {value}"

    return f


def gt_or_none(value):
    def f(*args):
        val = args[-1]
        assert val is None or val > value, f"Value must be None or >= {value}"

    return f


@attrs.define
class FlappyBirdConfig:
    """
    Configuration for the FlappyBird game.

    Attributes
    ----------
    bird_color: str
        Color of the bird. Can be 'blue' or 'red'.
    bird_jump_velocity: float
        Velocity of the bird after flap.
    bird_jump_frequency: int
        Number of steps before bird can falp again.
    bird_start_position: Tuple[int, int]
        Starting position of the bird.
    bird_dead_on_hit_ground: bool
        If True, game is over when bird hits the ground.
    bird_constrained_to_screen: bool
        If True, the bird cannot go fly above the screen.
    bird_max_speed: Optional[float]
        If not None, the bird's speed cannot exceed `bird_max_speed`.

    pipe_color: str
        Color of the pipes. Can be 'green' or 'red'.
    pipe_speed: float
        Speed of the pipes.
    pipe_gap_size: int
        Size of gap between pipes.
    pipe_spacing: int
        Space between pipes.

    background: str
        Type of background. Can be 'day' or 'night'.
    gravity: float
        Gravitational acceleration.
    dt: float
        Time between frames.
    fps: int
        Frames per second of the game.
    """

    bird_color: str = field(default="blue", validator=validators.in_(["blue", "red"]))
    bird_jump_velocity: float = field(default=4.0, validator=validators.gt(0.0))
    bird_jump_frequency: int = field(default=7, validator=validators.ge(0))
    bird_start_position: Tuple[int, int] = field(default=(100, 250))
    bird_dead_on_hit_ground: bool = field(default=True)
    bird_constrained_to_screen: bool = field(default=True)
    bird_max_speed: Optional[float] = field(default=None, validator=gt_or_none(0.0))
    bird_rotate: bool = field(default=True)

    pipe_color: str = field(default="green", validator=validators.in_(["red", "green"]))
    pipe_speed: float = field(default=3, validator=validators.gt(0))
    pipe_gap_size: int = field(default=150, validator=validators.gt(0))
    pipe_spacing: int = field(default=200, validator=validators.gt(0))

    background: str = field(default="day", validator=validators.in_(["day", "night"]))
    window_width: Optional[int] = field(default=None, validator=gt_or_none(0))
    window_height: Optional[int] = field(default=None, validator=gt_or_none(0))
    hide_screen: bool = field(default=False)
    show_score: bool = field(default=True)
    grayscale: bool = field(default=False)
    show_game_over_screen: bool = field(default=True)
    scaling_interpolation: int = field(
        default=cv.INTER_AREA, validator=validators.in_(CV_INTERPS)
    )

    gravity: float = field(default=2.0 / 5.0, validator=validators.gt(0.0))
    dt: float = field(default=1.0, validator=validators.gt(0.0))
    fps: Optional[int] = field(default=60, validator=gt_or_none(0))

    @classmethod
    def from_json(cls, file):
        with open(file, "r") as f:
            config = json.load(f)
        return cls(**config)
