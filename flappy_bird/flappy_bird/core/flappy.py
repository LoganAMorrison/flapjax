from typing import List, Optional

import numpy as np
import pygame

from .bird import Bird
from .config import FlappyBirdConfig
from .pipe import Pipe
from .resources import background_images, pipe_images
from .types import PyGameImage, PyGameSurface, RngGenerator


class FlappyBird:
    def __init__(self, config: FlappyBirdConfig, rng: Optional[RngGenerator] = None):
        # Config
        self.dead_on_hit_ground = config.bird_dead_on_hit_ground
        self.bird_constrained_to_screen = config.bird_constrained_to_screen
        self.background: PyGameImage = background_images[config.background]
        self.base: PyGameImage = background_images["base"]
        self.hide_screen = config.hide_screen
        self.show_score = config.show_score
        self.fps = config.fps
        if rng is None:
            self.rng = np.random.default_rng()

        # Screen/PyGame init
        pygame.init()
        pygame.display.init()
        self.screen: Optional[PyGameSurface] = None
        self.width = self.background.get_width()
        self.height = self.background.get_height()  # + self.base.get_height()
        self.rect = pygame.rect.Rect(0, 0, self.width, self.height)
        self.y_ground = self.background.get_height() - self.base.get_height()

        # Setup bases
        self.base_rects = [self.base.get_rect() for _ in range(3)]
        for i, rect in enumerate(self.base_rects):
            rect.top = self.y_ground
            rect.left = i * rect.width

        # Bird setup
        x0 = self.width / 2.0
        y0 = self.background.get_height() / 2.0
        self.bird = Bird(x0, y0, self.background.get_height(), config)

        # Pipe setup
        pipe_rect = pipe_images[config.pipe_color].get_rect()
        self.pipe_spacing = config.pipe_spacing
        self.pipe_gap_size = config.pipe_gap_size
        self.pipe_width = pipe_rect.width
        self.pipe_speed = config.pipe_speed
        npipes = int(np.ceil(self.width / (self.pipe_spacing + self.pipe_width)))

        bkg_h = self.background.get_height()
        ymin = bkg_h - pipe_rect.height - self.pipe_gap_size / 2.0
        ymax = pipe_rect.height + self.pipe_gap_size / 2.0
        shift = self.width + self.pipe_width / 2.0

        self.pipes: List[Pipe] = []
        for i in range(npipes):
            x = shift + i * (self.pipe_width + self.pipe_spacing)
            self.pipes.append(Pipe(config, x, ymin, ymax, self.rng))

        # Game state
        self.game_over = False
        self.score = 0
        self.next_pipe = 0
        self.best_score = 0
        self.clock = pygame.time.Clock()

    def flap(self):
        self.bird.flap()

    def step(self, action: int):
        assert action in [0, 1], "Invalid action. Must be 0 or 1."
        state = {"reward": 0, "hit-pipe": False, "hit-ground": False}

        self.bird.step(action)

        for i, pipe in enumerate(self.pipes):
            pipe.step()

            if pipe.right < 0.0:
                # New left position of the pipe
                left = self.pipes[i - 1].right + self.pipe_spacing
                # Make sure the new pipe starts off the screen
                left = np.clip(left, self.width, None)
                pipe.reset(left, self.rng)

        # Detect if player has passed a pipe
        if self.bird.left > self.pipes[self.next_pipe].right:
            self.next_pipe = (self.next_pipe + 1) % len(self.pipes)
            state["reward"] = 1

        # Detect if bird hit a pipe
        for pipe in self.pipes:
            if pipe.top_rect.colliderect(self.bird.rect):
                state["hit-pipe"] = True
            if pipe.bottom_rect.colliderect(self.bird.rect):
                state["hit-pipe"] = True

        # detect if bird hit ground
        if self.bird.rect.bottom > self.y_ground:
            state["hit-ground"] = True

        self.score += state["reward"]

        return state

    def _render(self, hidden: Optional[bool] = None):
        force_reinit = False
        if not (self.hide_screen == hidden):
            self.hide_screen = hidden
            force_reinit = True

        if self.screen is None or force_reinit:
            pygame.init()
            pygame.display.init()
            mode = pygame.SHOWN if not self.hide_screen else pygame.HIDDEN
            self.screen = pygame.display.set_mode(self.rect.size, flags=mode)

        self.screen.fill((0, 0, 0))
        self.screen.blit(self.background, (0, 0))

        self.bird.draw(self.screen)

        for pipe in self.pipes:
            pipe.draw(self.screen)

        # Step bases
        for i, base_rect in enumerate(self.base_rects):
            base_rect.left -= int(self.pipe_speed)
            if base_rect.right < 0:
                base_rect.left = self.base_rects[i - 1].right - int(self.pipe_speed)
            self.screen.blit(self.base, base_rect)

    def _flip(self):
        if not self.hide_screen:
            pygame.event.pump()
            if self.fps is not None:
                self.clock.tick(self.fps)
            pygame.display.flip()

    def render(self, hidden: Optional[bool] = None):
        self._render(hidden)
        assert self.screen is not None

        if self.show_score:
            score = pygame.font.Font("freesansbold.ttf", 32).render(
                f"{self.score}", True, (255, 255, 255)
            )
            rect = score.get_rect()
            rect.left = self.background.get_rect().left + 5
            rect.top = self.background.get_rect().top + 5
            self.screen.blit(score, rect)

        self._flip()

    def game_over_screen(self, hidden: Optional[bool] = None):
        self._render(hidden)
        assert self.screen is not None

        if self.show_score:
            score = pygame.font.Font("freesansbold.ttf", 32).render(
                f"Score: {self.score}", True, (255, 255, 255)
            )
            rect = score.get_rect()
            rect.left = self.background.get_rect().width // 2 - rect.width // 2
            rect.top = self.background.get_rect().height // 3
            self.screen.blit(score, rect)

            best_score = pygame.font.Font("freesansbold.ttf", 32).render(
                f"Best Score: {self.best_score}", True, (255, 255, 255)
            )
            rect = best_score.get_rect()
            rect.left = self.background.get_rect().width // 2 - rect.width // 2
            rect.top = self.background.get_rect().height // 3 + 40
            self.screen.blit(best_score, rect)

        self._flip()

    def reset(self):
        self.bird.reset()

        shift = self.width + self.pipe_width / 2.0
        for i, pipe in enumerate(self.pipes):
            x = shift + i * (self.pipe_spacing + pipe.top_rect.width)
            pipe.reset(x, self.rng)

        self.base_rects = [self.base.get_rect() for _ in range(3)]
        for i, rect in enumerate(self.base_rects):
            rect.top = self.y_ground
            rect.left = i * rect.width

        self.game_over = False
        self.score = 0
        self.next_pipe = 0

    def close(self):
        self.screen = None
        pygame.display.quit()
        pygame.quit()
