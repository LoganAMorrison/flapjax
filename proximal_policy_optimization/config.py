from typing import NamedTuple, Union

import optax


class PPOConfig(NamedTuple):
    horizon: int = 2048
    epochs: int = 10
    mini_batch_size: int = 64
    gamma: float = 0.99
    lam: float = 0.95
    n_actors: int = 1
    epsilon: Union[float, optax.Schedule] = 0.1
    c1: float = 0.5
    c2: float = 0.01
    total_frames: int = int(1e6)
    clip_reward: bool = True
