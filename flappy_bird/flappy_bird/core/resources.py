import pathlib

import pygame

__all__ = ["background_images", "pipe_images", "bird_images"]

# 560, 120
# 480, 200
# scale_y = 5.0 / 3.0
# scale_x = 1.094

resources_dir = pathlib.Path(__file__).parent.absolute().joinpath("resources")


def _load_image(name: str):
    return pygame.image.load(resources_dir.joinpath(name).as_posix())


background_images = {
    "day": _load_image("background-day.png"),
    "night": _load_image("background-night.png"),
    "base": _load_image("background-base.png"),
}

for key, image in background_images.items():
    w = image.get_rect().width * 5.0 / 3.0
    h = image.get_rect().height * (0.714 if key == "base" else 1.25)
    background_images[key] = pygame.transform.scale(image, (w, h))

pipe_images = {
    "red": _load_image("pipe-red.png"),
    "green": _load_image("pipe-green.png"),
}

for key, image in pipe_images.items():
    w = image.get_rect().width * 5.0 / 3.0
    h = image.get_rect().height * 1.094
    pipe_images[key] = pygame.transform.scale(image, (w, h))

bird_images = {
    "blue": {
        "upflap": _load_image("bluebird-upflap.png"),
        "midflap": _load_image("bluebird-midflap.png"),
        "downflap": _load_image("bluebird-downflap.png"),
    },
    "red": {
        "upflap": _load_image("redbird-upflap.png"),
        "midflap": _load_image("redbird-midflap.png"),
        "downflap": _load_image("redbird-downflap.png"),
    },
}

for color in bird_images.keys():
    for flap, image in bird_images[color].items():
        w = image.get_rect().width * 5.0 / 3.0
        h = image.get_rect().height * 5.0 / 3.0
        bird_images[color][flap] = pygame.transform.scale(image, (w, h))
