import gymnasium
import orbital_probe_dynamics

env = gymnasium.make("orbitalProbeDynamics-v1")

print(env.metadata)
