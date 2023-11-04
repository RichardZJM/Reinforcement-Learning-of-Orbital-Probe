import numpy as np
import pygame
import rebound
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces

import orbital_probe_dynamics.envs.planetaryProperties as planetProp


class OrbitalProbeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, tmax=1e4, window_size=1024) -> None:
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

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((0, 0, 0))  # Set black for space background

        # Rescale the pixel ratio and positions based on the furthest bodies
        # Stretch the bounds to get some buffer between the window edge
        bodyPositionsAU = self._getBodyPositions()
        maxBounds = np.max(np.abs(bodyPositionsAU)) * 2
        maxBounds = 50  #    Temporarily set a maximum bound
        conversionRatio = self.window_size / 2 / maxBounds
        bodyPositionsPX = (bodyPositionsAU * conversionRatio) + self.window_size / 2

        # Draw the SUN
        pygame.draw.circle(
            canvas,
            planetProp.renderColourProperties[0],
            np.full(2, self.window_size / 2),
            planetProp.renderSizeProperties[0],
        )

        # Draw all the planets
        for bodyPos, colour, size in zip(
            bodyPositionsPX[:-1],
            planetProp.renderColourProperties[1:],
            planetProp.renderSizeProperties[1:],
        ):
            pygame.draw.circle(canvas, colour, bodyPos, size)

        # Draw the spaceship
        pygame.draw.circle(
            canvas,
            planetProp.renderColourProperties[-1],
            bodyPositionsPX[-1],
            planetProp.renderSizeProperties[-1],
        )

        self.sim.integrate(
            self.sim.t + 1  # self.dt
        )  # Temporarily animate in the render (should be step)

        # The following line copies our drawings from `canvas` to the visible window
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # Update the clock based on the render framerate
        self.clock.tick(self.metadata["render_fps"])

    def _initSolarSystem(self) -> None:
        def genRandAngle() -> float:
            """Helper function to generate random angles

            Returns:
                float: Random angle in circle (rad)
            """
            return self.np_random.uniform(low=0, high=2 * np.pi)

        # Now, add the orbits of each of the bodies, with no inclination (ie. 2D)

        self.sim.add(m=1.0, r=0.005)  # Sun

        # Add in the planets from the property sheet
        for planetProperties in planetProp.orbitalProperties:
            self.sim.add(
                m=planetProperties["m"],
                r=planetProperties["r"],
                a=planetProperties["a"],
                e=planetProperties["e"],
                omega=planetProperties["omega"],
                theta=genRandAngle(),
            )

        # Add the spaceship in orbit around the earth
        self.sim.add(
            m=0,
            r=1.5e-10,  # 20 m radius
            a=0.000281848931423 * 10,  # Geosynchronous Orbit
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
