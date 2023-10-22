import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces


class OrbitalProbeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None) -> None:
        super().__init__()

        # Positions and velocities of each of the 9 planets and the space probe in 2-D
        self.observation_space = spaces.Dict(
            {
                "bodyPositions": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(2, 10), dtype=np.float32
                ),
                "bodyVelocities": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(2, 10), dtype=np.float32
                ),
                "timeElapsed": spaces.Box(
                    low=0, high=np.inf, shape=(1, 1), dtype=np.float32
                ),
            }
        )
        self.action_space = spaces.Box(low=0, high=1, shape=(2, 1), dtype=np.float32)

    def _get_obs(self):
        # Placeholder at the moment
        return {
            "bodyPositions": self.bodyPositions,
            "bodyVelocities": self.bodyPositions,
            "timeElapsed": self.timeElapsed,
        }

    def _get_info(self):
        return {"Time Elapsed": self.timeElapsed}

    def reset(self):
        return self._get_obs, self._get_info

    def step(self, action):
        # Perform some calculations
        terminated = False
        reward = 69

        return (
            self._get_obs(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        # Render some stuff
        pass

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
