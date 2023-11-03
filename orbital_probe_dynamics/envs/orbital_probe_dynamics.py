import numpy as np
import pygame
import rebound
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces


class OrbitalProbeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, tmax=1e4, window_size=512) -> None:
        super().__init__()
        self.tmax = tmax  # Set the maximum sim time allowed, (1 year = 2pi)

        # Obs is Positions and velocities of each of the 9 planets and the space probe in 2-D
        self.observation_space = spaces.Dict(
            {
                "bodyPositions": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(10, 2), dtype=np.float64
                ),
                "bodyVelocities": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(10, 2), dtype=np.float64
                ),
                "timeElapsed": spaces.Box(
                    low=0, high=np.inf, shape=(1, 1), dtype=np.float64
                ),
            }
        )

        # Action is direction and magnitude of thrust
        self.action_space = spaces.Box(low=0, high=1, shape=(2, 1), dtype=np.float32)

        self.window_size = window_size
        self.window = None
        self.clock = None

    def reset(self, seed: int = None, options={"dt": 2 * np.pi / 365}):
        super().reset(seed=seed)  # Reconcile seeding in enviroment

        # Prepare a new simulation and set the integrator options
        self.sim = None  # Ensure the previous simulation is cleared
        self.sim = rebound.Simulation()
        self.sim.integrator = "mercurius"
        self.dt = options["dt"]
        self.sim.dt = options["dt"]  # Default timestep (decision interval is 1 day)
        self.sim.boundary = "open"
        self.sim.configure_box(200)
        self._initSolarSystem()

        # Plotting block, usedful for troubleshooting
        # rebound.OrbitPlot(self.sim)
        # plt.show()
        return self._get_obs(), self._get_info()

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
        return self._render_frame()

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((0, 0, 0))  # Set a

        # Rescale the pixel ratio and positions based on the furtherst bodies
        bodyPositionsAU = self._getBodyPositions()
        bounds = np.array([np.min(bodyPositionsAU), np.max(bodyPositionsAU)])
        bounds = bounds * 1.5  # Strech the bounds to get some buffer between the edge
        conversionRatio = self.window_size / (bounds[1] - bounds[0])
        bodyPositionsPX = (bodyPositionsAU - bounds[0]) * conversionRatio

        # Draw all the bodies
        for bodyPos in bodyPositionsPX:
            pygame.draw.circle(canvas, (0, 0, 255), bodyPos, 5)

        # The following line copies our drawings from `canvas` to the visible window
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # Update the clock based on the render framerate
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _initSolarSystem(self) -> None:
        def genRandAngle() -> float:
            """Helper funciton to generate random angles

            Returns:
                float: Random angle in circle (rad)
            """
            return self.np_random.uniform(low=0, high=2 * np.pi)

        # Now, add the orbits of each of the planets, with no incclination (ie. 2D)

        self.sim.add(m=1.0, r=0.005)  # Sun

        self.sim.add(
            m=3.3011e23 * 5.02785e-31,
            r=2439.7 * 6.68459e-9,
            a=0.307499,
            e=0.205630,
            omega=48.331 / 180 * np.pi,
            theta=genRandAngle(),
        )  # Mercury

        self.sim.add(
            m=4.8675e24 * 5.02785e-31,
            r=6051.8 * 6.68459e-9,
            a=0.723332,
            e=0.006772,
            omega=54.884 / 180 * np.pi,
            theta=genRandAngle(),
        )  # Venus

        self.sim.add(
            m=5.972168e24 * 5.02785e-31,
            r=6371.0 * 6.68459e-9,
            a=1.0003,
            e=0.0167086,
            omega=114.20783 / 180 * np.pi,
            theta=genRandAngle(),
        )  # Earth

        self.sim.add(
            m=6.4171e23 * 5.02785e-31,
            r=3389.5 * 6.68459e-9,
            a=1.52368055,
            e=0.0934,
            omega=286.5 / 180 * np.pi,
            theta=genRandAngle(),
        )  # Mars

        self.sim.add(
            m=1.8982e27 * 5.02785e-31,
            r=69911 * 6.68459e-9,
            a=5.2038,
            e=0.0489,
            omega=273.867 / 180 * np.pi,
            theta=genRandAngle(),
        )  # Jupiter

        self.sim.add(
            m=5.6834e26 * 5.02785e-31,
            r=58232 * 6.68459e-9,
            a=9.5826,
            e=0.0565,
            omega=339.392 / 180 * np.pi,
            theta=genRandAngle(),
        )  # Saturn

        self.sim.add(
            m=8.6810e25 * 5.02785e-31,
            r=25362 * 6.68459e-9,
            a=19.19126,
            e=0.04717,
            omega=96.998857 / 180 * np.pi,
            theta=genRandAngle(),
        )  # Uranus

        self.sim.add(
            m=1.02413e26 * 5.02785e-31,
            r=24622 * 6.68459e-9,
            a=30.07,
            e=0.008678,
            omega=273.187 / 180 * np.pi,
            theta=genRandAngle(),
        )  # Neptune

        self.sim.add(
            m=1.303e22 * 5.02785e-31,
            r=1188.3 * 6.68459e-9,
            a=39.482,
            e=0.2488,
            omega=113.834 / 180 * np.pi,
            theta=genRandAngle(),
        )  # Pluto

        # Add the spaceship in orbit around the earth

        self.sim.add(
            m=0,
            r=1.5e-10,  # 20 m radius
            a=0.000281848931423,  # Geosynchronous Orbit
            theta=genRandAngle(),
            primary=self.sim.particles[3],  # Orbiting Earth
        )

        # Notes on the meanings of the simulation parameters
        # m = mass,
        # r = radius
        # a = semi=major axis (average orbital radius)
        # e = eccentricity (how elipitcal an orbit is)
        # omega = argument of periapsis/perihelion (angle to the closest point of orbit, relative to +x)
        # f = angle of the starting position of the satellite to the perihelion
        # theta = starting angle of the satelitle relative to the +x axis (use either f or theta, not both)

    def _getBodyPositions(self) -> np.array:
        return np.array(
            [[particle.x, particle.y] for particle in self.sim.particles[1:]]
        )

    def _getBodyVelocities(self) -> np.array:
        return np.array(
            [[particle.vx, particle.vy] for particle in self.sim.particles[1:]]
        )

    def _getTimeElapsed(self) -> float:
        return self.sim.t

    def _get_obs(self) -> dict:
        return {
            "bodyPositions": self._getBodyPositions(),
            "bodyVelocities": self._getBodyVelocities(),
            "timeElapsed": np.array(self._getTimeElapsed()),
        }

    def _get_info(self) -> dict:
        return {"Time Elapsed": self._getTimeElapsed()}
