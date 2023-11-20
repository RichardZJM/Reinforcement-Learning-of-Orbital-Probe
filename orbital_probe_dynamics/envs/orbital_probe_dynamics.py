import numpy as np
import pygame
import rebound
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces

import orbital_probe_dynamics.envs.planetaryProperties as planetProp


class OrbitalProbeEnv(gym.Env):
    """
    Custom enviroment following the Gymnasium RL enviroiment standard. A simple probe game which a spacecraft must navigate the Solar System to reach a target destination. Inherit from gymnasium.Env.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 512,
    }  # supported render modes and fps

    def __init__(
        self,
        render_mode=None,
        dt=2 * np.pi / 365,
        tmax=2 * np.pi * 100,
        window_size=1024,
    ) -> None:
        """Initialization of orbital probe enviroment.

        Args:
            render_mode (str, optional): Rendering mode. Defaults to None.
            tmax (float, optional): . Maximum time to simulation. 1 Year = 2pi. Defaults to 1e4.
            window_size (int, optional): Window size for pygame rendering. Defaults to 1024.
        """
        super().__init__()
        self.dt = dt
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
                    low=0, high=np.inf, shape=(1,), dtype=np.float64
                ),
                "fuel": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float64),
            }
        )

        # Action is direction and magnitude of thrust
        self.action_space = spaces.Discrete(5)

        # Set variables for the Pygame rendering
        self.window_size = window_size
        self.window = None
        self.clock = None

    def reset(self, seed: int = None, options={"rendering": False}) -> tuple:
        """A reset function to begin a new episode for training.

        Args:
            seed (int, optional): Seeding for planetary start positions. Defaults to None.
            options (dict, optional): Options, mostly for the timestep. Defaults to {"dt": 2 * np.pi / 365}.

        Returns:
            tuple: A tuple that contains the initial observation and extra info of the new initial state.
        """
        seed = 69  # Temporarily force a deterministic enviroment for testing
        super().reset(seed=seed)  # Reconcile seeding in enviroment
        self.rendering = options["rendering"]

        # Prepare a new simulation and set the integrator options
        self.sim = None  # Ensure the previous simulation is cleared
        self.sim = rebound.Simulation()
        self.sim.integrator = "mercurius"
        self.sim.dt = self.dt  # Default timestep/ decision interval is 1 day
        self.sim.boundary = "open"
        self.sim.configure_box(400)
        self.sim.additional_forces = self._rocketThrustForce
        self.sim.force_is_velocity_dependent = 1
        self.sim.collision = "none"
        # self.sim.exit_max_distance = 50.0
        self.sim.ri_mercurius.hillfac = 10
        self._initSolarSystem()

        # Spaceship properties
        self.fuel = 1
        self.deltaV = np.array([0, 0], dtype="float64")

        # Find the initial distance to pluto
        self.closestEncounter = self._getDistanceToPluto()
        self.previousHighestEnergy = self._getSpaceshipEnergy()

        # Plotting block, usedful for troubleshooting
        # rebound.OrbitPlot(self.sim)
        # plt.show()

        return self._get_obs(), self._get_info()

    def step(self, action):
        """Takes a step in the simulation, getting the next state and reward

        Args:
            action (dict): A dict of the actions according to the action space set out in the init function.

        Returns:
            tuple: a tuple containing the next state observation, the reward gain, whether the simulation has terminated, whether the truncation condition is satisfied (always False for our enviroment), and additional info
        """
        reward = 0  # Start with netural reward
        terminated = False  # Start with in non-terminate assesment

        shipV = np.array([self.sim.particles[-1].vx, self.sim.particles[-1].vy])
        thrustDirection = shipV / np.linalg.norm(
            shipV
        )  # When action is 1 or 0 doesn't matter (posigrade/ no burn)

        thrustMagnitude = 0  # No thrust by default

        if not action == 0 and self.fuel > 0:
            self.fuel -= planetProp.spaceShipThrustProperties["thrustPerDT"]
            thrustMagnitude = (
                planetProp.spaceShipThrustProperties["thrustPerDT"]
                * planetProp.spaceShipThrustProperties["availableDeltaV"]
            )
            if action == 2:
                thrustDirection = -thrustDirection  # Retrograde burn
            if action == 3:
                thrustDirection = thrustDirection[::-1]  # Radial Burn
            if action == 4:
                thrustDirection = -thrustDirection[::-1]  # Anti-Radial Burn

        # Convert thrust into desired deltaV
        self.deltaV = thrustDirection * thrustMagnitude

        takingInput = False  # Flag for whether to take input from the policy, ie. loop continously if not

        while not takingInput and not terminated:
            # Try to step and stop if things get too close
            try:
                self.sim.step()
            except rebound.Encounter as error:
                terminated = True

            self.deltaV *= 0  # Clear the thurst after each integration

            def rewardScalingFunction(value: float) -> float:
                return -(value**0.15) / 50**0.15 + 1

            # Find the distance to pluto, and dispense a reward
            distance = self._getDistanceToPluto()
            reward += (
                self._progressivizeDistanceReward(
                    distance, self.closestEncounter, rewardScalingFunction
                )
                * 1000
            )
            self.closestEncounter = min(
                distance, self.closestEncounter
            )  # Update the closest previous encounter

            # We also reward the agent for gaining system energy
            spaceshipEnergy = self._getSpaceshipEnergy()
            if spaceshipEnergy > 3:
                reward -= spaceshipEnergy / 10
            # DIspense a reward for increasing energy up until there is too much energy. 0 energy is the escape energy, so stop giving rewards past near that point.
            reward += (
                max(min(spaceshipEnergy, 0.1) - self.previousHighestEnergy, 0) * 000 / 4
            )
            self.previousHighestEnergy = max(
                spaceshipEnergy, self.previousHighestEnergy
            )  # Update the previous highest energy

            # Terminate on reaching pluto
            if distance <= 0.1:
                reward += 1000
                reward += self.fuel * 3
                terminated = True
                print(
                    "Closest Encounter: %2.5f,   Highest Energy: %1.5f,     Success!"
                    % (self.closestEncounter, self.previousHighestEnergy)
                )

            # Terminate on reaching time limit
            if self._getTimeElapsed() > self.tmax:
                print(
                    "Closest Encounter: %2.5f,   Highest Energy: %1.5f"
                    % (self.closestEncounter, self.previousHighestEnergy)
                )
                terminated = True

            # Terminate and punish on out of bounds
            if len(self.sim.particles[1:]) == 9:
                pos = self.lastPositions
                vs = self.lastVel
                print(
                    "Closest Encounter: %2.5f,   Highest Energy: %1.5f,     Out of Bounds!"
                    % (self.closestEncounter, self.previousHighestEnergy)
                )
                print(pos[-1], vs[-1], self._getTimeElapsed())
                return (
                    {
                        "bodyPositions": pos,
                        "bodyVelocities": vs,
                        "timeElapsed": np.array(
                            [self._getTimeElapsed()], dtype="float64"
                        ),
                        "fuel": np.array([self.fuel], dtype="float64"),
                    },
                    reward - 400,
                    True,
                    False,
                    self._get_info(),
                )
            self.lastPositions = self._getBodyPositions()
            self.lastVel = self._getBodyVelocities()

            # If the fuel is not zero, we continue to the next step. Always continue if the rendering option is true.
            if self.fuel > 0 or self.rendering == True:
                takingInput = True

        return (
            self._get_obs(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def render(self) -> None:
        """Renders the current state of the enviroment using pygame for human consumption."""

        # Generate a new PyGame window if one doesn't exist
        if self.window is None:
            pygame.init()
            pygame.display.set_caption("Mission to Pluto")
            pygame.display.init()
            pygame.font.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        # Set the clock. Used later to match the requested FPS.
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Blank black canvas to represent empty space.
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((0, 0, 0))  # Set black for space background

        # Rescale the pixel ratio and positions based on the the maximum bounds
        # Stretch the bounds to get some buffer between the window edge
        bodyPositionsAU = self._getBodyPositions()
        # maxBounds = np.max(np.abs(bodyPositionsAU)) * 2
        maxBounds = 60  #    Temporarily set a maximum bound
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

        # Draw the current time and fuel
        gameFont = pygame.font.SysFont("Comic Sans MS", 40)
        timeSurface = gameFont.render(
            "Year: " + format(self._getTimeElapsed() / 2 / np.pi, ".2f"),
            False,
            (255, 255, 255),
        )
        fuelSurface = gameFont.render(
            "Fuel: " + format(self.fuel * 100, ".1f") + "%",
            False,
            (255, 255, 255),
        )

        # The following line copies our drawings from `canvas` to the visible window
        self.window.blit(canvas, canvas.get_rect())
        self.window.blit(timeSurface, (0, 0))
        self.window.blit(fuelSurface, (0, 50))
        pygame.event.pump()
        pygame.display.update()

        # Update the clock based on the render framerate
        self.clock.tick(self.metadata["render_fps"])

    def close(self) -> None:
        """Function to ensure all pygame constructs are cleared after rendering is done. Should be user called."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _initSolarSystem(self) -> None:
        """Helper function to intialize the simulation with the solar system."""

        def genRandAngle() -> float:
            """Helper function to generate random angles

            Returns:
                float: Random angle in circle (rad)
            """
            return self.np_random.uniform(low=0, high=2 * np.pi)

        # ====Now, add the orbits of each of the bodies, with no inclination (ie. 2D)=====
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
            a=0.05,  # Geosynchronous Orbit *100
            theta=genRandAngle(),
            primary=self.sim.particles[3],  # Orbiting Earth
        )

        # =====Notes on the meanings of the simulation parameters=====
        # m = mass,
        # r = radius
        # a = semi=major axis (average orbital radius)
        # e = eccentricity (how elipitcal an orbit is)
        # omega = argument of periapsis/perihelion (angle to the closest point of orbit, relative to +x)
        # f = angle of the starting position of the satellite to the perihelion
        # theta = starting angle of the satelitle relative to the +x axis (use either f or theta, not both)

    def _rocketThrustForce(self, sim) -> None:
        ps = self.sim.particles
        ps[-1].vx += self.deltaV[0]
        ps[-1].vy += self.deltaV[1]

    def _getDistanceToPluto(self) -> float:
        """Helper function: gets distance of spaceship from pluto

        Returns:
            float: distance to pluto
        """
        dp = self.sim.particles[-1] - self.sim.particles[-2]
        return np.sqrt(dp.x**2 + dp.y**2)

    def _getSpaceshipEnergy(self) -> float:
        potentialEnergy = 0
        for body in self.sim.particles[:1]:
            dp = body - self.sim.particles[-1]
            potentialEnergy -= body.m / np.sqrt(dp.x**2 + dp.y**2)
        speedSquared = self.sim.particles[-1].vx ** 2 + self.sim.particles[-1].vy ** 2

        # Return the sum of the kinetic energy and potential energy
        return potentialEnergy + speedSquared / 2

    def _progressivizeDistanceReward(
        self, currentValue: float, previousValue: float, rewardFunction: callable
    ) -> float:
        """Converts a final sparse reward distribution into a progressive set of rewards

        Args:
            currentValue (float): current distance
            previousValue (float): previous distance

        Returns:
            float: reward to dispence
        """
        return max(
            rewardFunction(currentValue) - rewardFunction(previousValue),
            0,
        )

    def _getBodyPositions(self) -> np.array:
        """Returns the body positions of the simulation

        Returns:
            np.array: 10 by 2 array of positions of all bodies except the sun (x and y).
        """
        return np.array(
            [[particle.x, particle.y] for particle in self.sim.particles[1:]]
        )

    def _getBodyVelocities(self) -> np.array:
        """Returns the body positions of the simulation

        Returns:
            np.array: 10 by 2 array of velocities of all bodies except the sun (x and y).
        """
        return np.array(
            [[particle.vx, particle.vy] for particle in self.sim.particles[1:]]
        )

    def _getTimeElapsed(self) -> float:
        """Returns the time elapsed in the simulation

        Returns:
            float: time elapsed
        """
        return self.sim.t

    def _get_obs(self) -> dict:
        """Helper function to extract the current observation of the state of the simulation.

        Returns:
            dict: Dict of observations formated according to the init function.
        """
        return {
            "bodyPositions": self._getBodyPositions(),
            "bodyVelocities": self._getBodyVelocities(),
            "timeElapsed": np.array([self._getTimeElapsed()], dtype="float64"),
            "fuel": np.array([self.fuel], dtype="float64"),
        }

    def _get_info(self) -> dict:
        """Helper function to get the additional information to output to the user.

        Returns:
            dict: Dict of additonal information.
        """
        return {}  # {"Time Elapsed": self._getTimeElapsed()}
