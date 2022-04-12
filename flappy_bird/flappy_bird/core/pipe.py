import pygame

from .config import FlappyBirdConfig
from .resources import pipe_images
from .types import PyGameImage, PyGameSurface, RngGenerator


class Pipe:
    def __init__(
        self,
        config: FlappyBirdConfig,
        x: float,
        ymin: float,
        ymax: float,
        rng: RngGenerator,
    ):
        # Config
        self.gap_size = config.pipe_gap_size
        self.velocity_x = config.pipe_speed
        self.ymin = ymin
        self.ymax = ymax
        image: PyGameImage = pipe_images[config.pipe_color]
        self.top_image = pygame.transform.rotate(image, 180.0)
        self.top_rect = self.top_image.get_rect()
        self.bottom_image = image
        self.bottom_rect = self.bottom_image.get_rect()
        self.width = self.top_rect.width
        self.dt = config.dt

        # State
        self.x = x
        self.y = 0.0
        self.reset(x, rng)

    def step(self) -> None:
        self.x -= self.velocity_x * self.dt
        left = int(self.left)
        self.top_rect.left = left
        self.bottom_rect.left = left

    def reset(self, x, rng: RngGenerator) -> None:
        self.x = x
        self.y = rng.uniform(low=self.ymin, high=self.ymax)

        left = int(self.left)
        self.top_rect.left = left
        self.top_rect.bottom = int(self.y - self.gap_size / 2.0)

        self.bottom_rect.left = left
        self.bottom_rect.top = self.top_rect.bottom + self.gap_size

    def draw(self, surface: PyGameSurface):
        surface.blit(self.top_image, self.top_rect)
        surface.blit(self.bottom_image, self.bottom_rect)

    @property
    def left(self) -> float:
        return self.x - self.width / 2.0

    @property
    def right(self) -> float:
        return self.x + self.width / 2.0
