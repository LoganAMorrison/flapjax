from typing import Optional
import numpy as np
import pygame

from .config import FlappyBirdConfig
from .resources import bird_images
from .types import PyGameImage, PyGameRect, PyGameSurface


class Bird:
    def __init__(self, x0, y0, window_height, config: FlappyBirdConfig):
        # Config
        self.jump_velocity = config.bird_jump_velocity
        self.jump_frequency = config.bird_jump_frequency
        self.gravity = config.gravity
        self.omega = 45.0 * self.gravity / (2 * self.jump_velocity)
        self.images = bird_images[config.bird_color]
        self.image: PyGameImage = self.images["midflap"]
        self.dt = config.dt
        self.x0 = x0
        self.y0 = y0
        self.window_height = window_height
        self.rotate = config.bird_rotate
        self.max_speed: Optional[float] = config.bird_max_speed
        self.num_flaps = int(np.ceil(self.jump_velocity / (self.gravity * self.dt)))

        # State
        self.x = x0
        self.y = y0
        self.velocity_y = 0.0
        self.dead = False
        self.angle = 0.0
        self.jump_counter = 0
        self.flap_counter = 0
        self.flap_type: str = "midflap"

        self.reset()

    @property
    def left(self) -> int:
        return int(self.x - self.image.get_width() / 2.0)

    @property
    def right(self) -> int:
        return int(self.x + self.image.get_width() / 2.0)

    @property
    def top(self) -> int:
        return int(self.y - self.image.get_height() / 2.0)

    @property
    def bottom(self) -> int:
        return int(self.y + self.image.get_height() / 2.0)

    def reset(self):
        self.x = self.x0
        self.y = self.y0
        self.velocity_y = 0.0
        self.dead = False
        self.angle = 0.0
        self.jump_counter = 0

    def flap(self):
        if self.jump_counter == 0:
            self.velocity_y = -self.jump_velocity
            self.jump_velocity = self.jump_frequency
            self.angle = 45.0
            self.flap_type = "upflap"
            self.image = self.images[self.flap_type]
            self.flap_counter = self.num_flaps
            self.jump_counter = self.jump_frequency

    def step(self, action: int):
        if self.flap_counter > 0:
            self.flap_counter -= 1
            if self.flap_type == "upflap":
                self.flap_type = "downflap"
            elif self.flap_type == "downflap":
                self.flap_type = "upflap"
            self.image = self.images[self.flap_type]
        else:
            self.flap_type = "midflap"
            self.image = self.images["midflap"]

        if self.jump_counter > 0:
            self.jump_counter -= 1

        if action == 1:
            self.flap()

        self.angle = np.clip(self.angle - self.omega * self.dt, -90.0, 45.0)
        self.y += self.velocity_y * self.dt
        self.velocity_y += self.gravity * self.dt

        if self.max_speed is not None:
            maxv = abs(self.max_speed)
            self.velocity_y = np.clip(self.velocity_y, -maxv, maxv)

        ymax = self.window_height - self.image.get_height() / 2.0
        if self.y > ymax:
            self.velocity_y = 0.0
            self.y = ymax
        if self.y < 0.0:
            self.y = 0.0
            self.velocity_y = 0.0

    @property
    def rect(self) -> PyGameRect:
        rect = self.image.get_rect()
        rect.left = self.left
        rect.top = self.top
        return rect

    def draw(self, surface: PyGameSurface) -> None:
        image = self.image
        rect = self.rect
        if self.rotate:
            image = pygame.transform.rotate(image, self.angle)
        surface.blit(image, rect)
