from gymnasium.envs.registration import register

register(
    id="orbitalProbeDynamics-v1",
    entry_point="orbital_probe_dynamics.envs:OrbitalProbeEnv",
    max_episode_steps=5000,
)
