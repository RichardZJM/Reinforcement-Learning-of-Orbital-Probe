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
        "render_fps": 120,
    }  # supported render modes and fps

    def __init__(
        self,
        render_mode=None,
        dt=2 * np.pi / 365,
        tmax=2 * np.pi * 100,
        window_size=1024,
        trainingStage=0,
        maxDeviation=0.003,
    ) -> None:
        """Initializes an orbital probe enviroment

        Args:
            render_mode (str, optional): Rendering mode: human or None. Defaults to None.
            dt (float, optional): The time step of the simulations. Defaults to 2*np.pi/365 (1 earth day).
            tmax (float, optional): The maximum duration to simulate. Defaults to 2*np.pi*100 (1 earth year).
            window_size (int, optional): Window size in pixels. Defaults to 1024.
            trainingStage (int, optional): Traning stage in curriculum training. Defaults to 0 (full simulation).
            maxDeviation (float, optional):  Randomness of the inital conditions as applied to the training stage. Defaults to 0.003.
        """

        super().__init__()
        self.dt = dt
        self.tmax = tmax  # Set the maximum sim time allowed, (1 year = 2pi)
        self.trainingStage = trainingStage  # Set the current training stage
        self.maxDeviation = maxDeviation

        # Obs is Positions and velocities of each of the 9 planets and the space probe in 2-D
        self.observation_space = spaces.Dict(
            {
                "bodyPositions": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(10, 2), dtype=np.float64
                ),
                "bodyVelocities": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(10, 2), dtype=np.float64
                ),
            }
        )

        # Action is direction and magnitude of thrust
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float64)

        # Set variables for the Pygame rendering
        self.window_size = window_size
        self.window = None
        self.clock = None

    def reset(self, seed: int = None, options={"rendering": False}) -> tuple:
        """A reset function to begin a new episode for training.

        Args:
            seed (int, optional): Seeding for planetary start positions. Defaults to None.
            options (dict, optional): Needed to force rendering in some cases. Defaults to {"rendering": False}.

        Returns:
            tuple: A tuple that contains the initial observation and extra info of the new initial state.
        """
        super().reset(seed=seed)  # Reconcile seeding in enviroment
        self.rendering = options["rendering"]

        # ===== Prepare a new simulation and set the integrator options =====
        self.sim = None  # Ensure the previous simulation is cleared
        self.sim = rebound.Simulation()
        self.sim.integrator = "mercurius"
        self.sim.dt = self.dt  # Default timestep/ decision interval is 1 day
        self.sim.boundary = "open"
        self.sim.configure_box(200)
        self.sim.additional_forces = self._rocketThrustForce
        self.sim.force_is_velocity_dependent = 1
        self.sim.collision = "none"
        self.sim.ri_mercurius.hillfac = 10

        # ===== Spaceship properties =====
        self.fuel = 1
        self.deltaV = np.array([0, 0], dtype="float64")

        self._initSolarSystem()  # Initalize the solar system

        # Find the distance ti and potential of reaching Pluto
        self.closestEncounter = self._getDistanceToPluto()
        self.previousHighestEnergy = self._getSpaceshipEnergy()

        # ===== Plotting block, usedful for troubleshooting =====
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
        action = action * 0.5 + 0.5  # Convert from normalized input

        reward = 0  # Start with a net neutral reward
        terminated = False  # Epsiode is not terminated by default

        action[0] = (
            action[0] * planetProp.spaceShipThrustProperties["thrustPerDT"]
        )  # Limit the agent to using a fraction of his deltaV per day

        projectedFuel = self.fuel - action[0]  # Calculate the project fuel remaining
        deltaVMagnitude = min(
            action[0], self.fuel
        )  # Calculate what trhis is to be used.
        # self.fuel = max(0, projectedFuel)     # This enables finite fuel

        # Convert input into desired deltaV using trig to get the direction
        self.deltaV[0] = np.cos(action[1] * 2 * np.pi)
        self.deltaV[1] = np.sin(action[1] * 2 * np.pi)
        self.deltaV *= (
            deltaVMagnitude * planetProp.spaceShipThrustProperties["availableDeltaV"]
        )

        # Try to step and stop if things get too close
        try:
            self.sim.step()
        except rebound.Encounter as error:
            terminated = True

        self.deltaV *= 0  # Clear the thurst after each integration

        def rewardScalingFunction(value: float) -> float:
            """Shaping function for the closest encounter

            Args:
                value (float): new closest encounter

            Returns:
                float: shaped reward
            """
            return -(value**0.2) / 50**0.2 + 1

        # Find the distance to pluto, and dispense a shaped reward
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

        # We also reward the agent for gaining potential energy
        spaceshipEnergy = self._getSpaceshipEnergy()
        if spaceshipEnergy > 3:
            reward -= spaceshipEnergy / 10

        reward += (
            max(min(spaceshipEnergy, -0.1) - self.previousHighestEnergy, 0) * 1000 / 6
        )  # DIspense a reward for increasing energy up until there is too much energy. 0 energy is the escape energy, so stop giving rewards past near that point.
        self.previousHighestEnergy = max(
            spaceshipEnergy, self.previousHighestEnergy
        )  # Update the previous highest energy

        # Terminate on reaching pluto (defined as being within 0.1 a.u.)
        if distance <= 0.1:
            reward += 1000
            reward += self.fuel * 3000
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
        if (
            len(self.sim.particles[1:]) == 9
        ):  # On out of bounds, the simulator kills the particle
            pos = self.lastPositions
            vs = self.lastVel
            print(
                "Closest Encounter: %2.5f,   Highest Energy: %1.5f,     Out of Bounds!"
                % (self.closestEncounter, self.previousHighestEnergy)
            )
            return (
                {
                    "bodyPositions": pos,
                    "bodyVelocities": vs,
                    "timeElapsed": np.array([self._getTimeElapsed()], dtype="float64"),
                    "fuel": np.array([self.fuel], dtype="float64"),
                },
                reward,
                True,
                False,
                self._get_info(),
            )
        # Store the last positions and velocities for debugging in the case that a particle escapes
        self.lastPositions = self._getBodyPositions()
        self.lastVel = self._getBodyVelocities()

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
        maxBounds = 60  #    You can also just physically set a maximum bound
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
        # Kill the pygame window if it's still alive
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _initSolarSystem(self) -> None:
        """Helper function to intialize the simulation with the solar system."""

        def randomAngleDeviation() -> float:
            """Helper function to generate random angle deviations

            Returns:
                float: Random angle in circle (rad)
            """
            out = self.np_random.uniform(low=-np.pi, high=np.pi)

            # If stage 2 is selected, we always use the maximum possible angle deviation in order to generate rotational variance
            if self.trainingStage == 2:
                return out
            return out * self.maxDeviation

        def randomPercentDeviaton(baseValue: float):
            """Generates a deviation based on the baseValue provided

            Args:
                baseValue (float): base value

            Returns:
                _type_: a devation based on the base value
            """
            return (
                baseValue * self.np_random.uniform(low=-1, high=1) * self.maxDeviation
            )

        # ===== Now, add the orbits of each of the bodies, with no inclination (ie. 2D) =====
        self.sim.add(m=1.0, r=0.005)  # Sun

        # Add in the planets from the property sheet
        for planetProperties in planetProp.orbitalProperties:
            self.sim.add(
                m=planetProperties["m"],
                r=planetProperties["r"],
                a=planetProperties["a"],
                e=planetProperties["e"],
                omega=planetProperties["omega"],
                theta=planetProperties["baseTheta"] + randomAngleDeviation(),
            )

        # In training stage one, we place the spaceship closer with a good trajectory
        if self.trainingStage == 1:
            self.sim.add(
                m=0,
                r=1.5e-10,  # 20 m radius
                a=35 + randomPercentDeviaton(1),
                theta=0,
            )
            self.sim.particles[-1].vx = 0.5 + randomPercentDeviaton(0.5)
            self.sim.particles[-1].vy = 0.4 + randomPercentDeviaton(0.38)
            self.fuel = 0.1 + randomPercentDeviaton(0.2)
            self.fuelPunishment = 0
            print(len(self.sim.particles))

        # In training stage 2, we use a slightly lower orbit and more rotational variance
        elif self.trainingStage == 2:
            self.sim.add(
                m=0,
                r=1.5e-10,  # 20 m radius
                a=31 + randomPercentDeviaton(1),
                e=0.2488,
                omega=113.834 / 180 * np.pi,
                theta=self.sim.particles[-1].theta - 0.5,
            )

            self.sim.particles[-1].vx *= 2 + randomPercentDeviaton(0.3)
            self.sim.particles[-1].vy *= 2 + randomPercentDeviaton(0.3)

            self.fuel = 0.1 + randomPercentDeviaton(0.2)
            self.fuelPunishment = 0

        # In the general case, we start from Earth
        else:
            # Add the spaceship in orbit around the earth
            self.sim.add(
                m=0,
                r=1.5e-10,  # 20 m radius
                a=1.1,  # Geosynchronous Orbit *100
                theta=4.262976026939784 + randomAngleDeviation(),
                primary=self.sim.particles[0],  # Around Earth
            )
            self.fuel = 1
            self.fuelPunishment = 3

        # ===== Notes on the meanings of the simulation parameters =====
        # m = mass,
        # r = radius
        # a = semi=major axis (average orbital radius)
        # e = eccentricity (how elipitcal an orbit is)
        # omega = argument of periapsis/perihelion (angle to the closest point of orbit, relative to +x)
        # f = angle of the starting position of the satellite to the perihelion
        # theta = starting angle of the satelitle relative to the +x axis (use either f or theta, not both)

    def _rocketThrustForce(self, sim) -> None:
        """Helper function that is passes to the simulation to pass spaceship thrust to the Rebound simulator

        Args:
            sim (_type_): the simulation
        """
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
        """Helper function that calculate the spaceships potential to reach Pluto

        Returns:
            float: the potential as an energy
        """
        potentialEnergy = 0
        for body in self.sim.particles[:1]:
            dp = body - self.sim.particles[-1]
            potentialEnergy -= body.m / np.sqrt(dp.x**2 + dp.y**2)
        speedSquared = self.sim.particles[-1].vx ** 2 + self.sim.particles[-1].vy ** 2

        # Return the sum of the kinetic energy and potential energy
        # Disable the kinetic energy if we don't want to use in the potential definition
        return potentialEnergy  # + speedSquared / 2

    def _progressivizeDistanceReward(
        self, currentValue: float, previousValue: float, rewardFunction: callable
    ) -> float:
        """Converts a final sparse reward distribution into a incremental set of rewards on the improvement over the previous best

        Args:
            currentValue (float): current distance
            previousValue (float): previous distance
            rewardFunction (callable): the shaping function to apply

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
            # "timeElapsed": np.array([self._getTimeElapsed()], dtype="float64"),
            # "fuel": np.array([self.fuel], dtype="float64"),
        }

    def _get_info(self) -> dict:
        """Helper function to get the additional information to output to the user.

        Returns:
            dict: Dict of additonal information.
        """
        return {}  # {"Time Elapsed": self._getTimeElapsed()}
