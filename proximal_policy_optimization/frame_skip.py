import gym
import numpy as np


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
