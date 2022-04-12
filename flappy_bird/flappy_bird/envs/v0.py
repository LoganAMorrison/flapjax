from typing import Tuple

import cv2 as cv
import gym
import numpy as np
import pygame
from gym import spaces

from flappy_bird.core.config import FlappyBirdConfig
from flappy_bird.core.flappy import FlappyBird

ActType = int
ObsType = np.ndarray

config = FlappyBirdConfig(
    bird_color="blue",
    bird_jump_velocity=4.0,
    bird_jump_frequency=4,
    bird_start_position=(100, 250),
    bird_dead_on_hit_ground=True,
    bird_constrained_to_screen=True,
    bird_max_speed=None,
    bird_rotate=True,
    pipe_color="green",
    pipe_speed=3,
    pipe_gap_size=150,
    pipe_spacing=200,
    background="day",
    window_width=84,
    window_height=84,
    hide_screen=True,
    show_score=False,
    show_game_over_screen=False,
    grayscale=True,
    scaling_interpolation=cv.INTER_LINEAR,
    gravity=0.4,
    dt=1.0,
    fps=None,
)


class FlappyBirdEnvV0(gym.Env):
    metadate = {"render.modes": ["human", "none"]}

    def __init__(self) -> None:
        self.flappy = FlappyBird(config)
        self.show_game_over_screen = config.show_game_over_screen
        self.bird_dead_on_hit_ground = config.bird_dead_on_hit_ground
        self.grayscale = config.grayscale
        self.hide_screen = config.hide_screen

        # if config.window_height is not None:
        #     self.height = config.window_height
        # else:
        #     self.height = self.flappy.height

        # if config.window_width is not None:
        #     self.width = config.window_width
        # else:
        #     self.height = self.flappy.height

        # if self.grayscale:
        #     shape = (self.height, self.width)
        # else:
        #     shape = (self.height, self.width, 3)
        shape = (self.flappy.height, self.flappy.width, 3)

        self.observation_space = spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8
        )
        self.action_space = spaces.Discrete(2)

        self.game_over = True

    def _observation(self):
        self.flappy._render(self.hide_screen)
        assert self.flappy.screen is not None

        obs = np.array(pygame.surfarray.pixels3d(self.flappy.screen), dtype=np.uint8)

        # if (not self.width == self.flappy.width
        #       or not self.height == self.flappy.height):
        #     obs = cv.resize(obs, (self.width, self.height))

        # if self.grayscale:
        #     obs = cv.cvtColor(obs, cv.COLOR_BGR2GRAY)
        #     return np.transpose(obs, axes=(1, 0))

        return np.transpose(obs, axes=(1, 0, 2))

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        assert not self.game_over, "Call reset before step."
        state = self.flappy.step(action)

        if state["hit-pipe"]:
            self.game_over = True

        if state["hit-ground"] and self.bird_dead_on_hit_ground:
            self.game_over = True

        obs = self._observation()
        reward = state["reward"]
        done = self.game_over
        info = dict()

        return obs, reward, done, info

    def render(self, mode: str = "none"):
        hidden = mode == "none"
        if self.game_over and self.show_game_over_screen:
            self.flappy.game_over_screen(hidden)
        else:
            self.flappy.render(hidden)

    def close(self) -> None:
        self.flappy.close()

    def reset(self) -> ObsType:
        self.flappy.reset()
        self.game_over = False
        return self._observation()
