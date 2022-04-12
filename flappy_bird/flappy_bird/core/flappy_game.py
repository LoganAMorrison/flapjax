import numpy as np
import pygame

from .config import FlappyBirdConfig
from .flappy import FlappyBird

CONFIG = FlappyBirdConfig(
    bird_color="blue",
    bird_jump_velocity=4.0,
    bird_jump_frequency=7,
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
    window_width=500,
    window_height=500,
    hide_screen=False,
    show_score=True,
    gravity=0.4,
    dt=1.0,
    fps=60,
)


class FlappyBirdGame:
    def __init__(self):
        self.game = FlappyBird(CONFIG)
        self.action_keys = [pygame.K_SPACE, pygame.K_UP, pygame.K_KP_ENTER]
        self.game_over = False

    def _step(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key in self.action_keys:
                    return self.game.step(1)
        return self.game.step(0)

    def step(self):
        assert not self.game_over, "Game is over. Call reset."
        state = self._step()
        if state["hit-ground"] or state["hit-pipe"]:
            self.game_over = True

    def render(self):
        self.game.render()

    def reset(self):
        self.game.reset()
        self.game_over = False

    def _play(self):
        while not self.game_over:
            self.step()
            self.render()

    def game_over_screen(self):
        self.game.game_over_screen()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key in self.action_keys:
                        return True
                    if event.key == pygame.K_ESCAPE:
                        return False

    def play(self):
        amp = 5
        T = 50
        t = 0

        y0 = self.game.bird.y
        while True:
            self.render()
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key in self.action_keys:
                        self._play()
                        if self.game_over_screen():
                            self.reset()
                            y0 = self.game.bird.y
                        else:
                            return
                    if event.key == pygame.K_ESCAPE:
                        self.game.close()
                        return

            self.game.bird.y = y0 + amp * np.sin(2 * np.pi * t / T)
            t = (t + 1) % T
