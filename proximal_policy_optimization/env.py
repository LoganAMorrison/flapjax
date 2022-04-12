from typing import Union, Tuple

import gym
import numpy as np
import jax
import jax.numpy as jnp

GymEnv = gym.Env
GymVecEnv = Union[gym.vector.AsyncVectorEnv, gym.vector.SyncVectorEnv]


def env_reset(env: Union[GymEnv, GymVecEnv]):
    """Reset environment and return jax array of observation."""
    observation = env.reset()
    return jnp.array(observation, dtype=jnp.float32)


def env_step(
    action: jnp.ndarray, env: Union[GymEnv, GymVecEnv]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Step environment and return jax array of observation, reward and terminal status."""
    act = np.array(jax.device_get(action), dtype=np.int32)

    if not isinstance(env, gym.vector.VectorEnv):
        observation, reward, done, _ = env.step(act[0])
    else:
        observation, reward, done, _ = env.step(act)

    observation = np.array(observation)
    reward = np.array(reward, dtype=np.int32)
    done = np.array(done, dtype=np.int32)

    # Make the batch dimension for non-vector environments
    if not isinstance(env, gym.vector.VectorEnv):
        observation = np.expand_dims(observation, 0)
        reward = np.expand_dims(reward, 0)
        done = np.expand_dims(done, 0)

    return observation, reward, done


class FrameSkip(gym.Wrapper):
    def __init__(self, env: gym.Env, frame_skip: int = 4):
        super().__init__(env)
        assert frame_skip > 0

        self.frame_skip = frame_skip

        # buffer of most recent two observations for max pooling
        assert env.observation_space.shape is not None
        self.obs_buffer = [
            np.empty(env.observation_space.shape, dtype=np.uint8),
            np.empty(env.observation_space.shape, dtype=np.uint8),
        ]

        self.observation_space = env.observation_space

    def step(self, action):
        r = 0.0

        done = False
        info = dict()
        for t in range(self.frame_skip):
            observation, reward, done, info = self.env.step(action)
            r += reward

            if done:
                break
            if t == self.frame_skip - 2:
                self.obs_buffer[1] = observation
            elif t == self.frame_skip - 1:
                self.obs_buffer[0] = observation

        return self._get_obs(), r, done, info

    def reset(self, **kwargs):
        self.obs_buffer[0] = self.env.reset(**kwargs)
        self.obs_buffer[1].fill(0)
        return self._get_obs()

    def _get_obs(self):
        if self.frame_skip > 1:  # more efficient in-place pooling
            np.maximum(self.obs_buffer[0], self.obs_buffer[1], out=self.obs_buffer[0])
        obs = self.obs_buffer[0]
        return obs
