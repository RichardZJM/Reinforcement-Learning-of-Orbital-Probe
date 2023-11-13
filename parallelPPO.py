import gymnasium as gym
import orbital_probe_dynamics
import numpy as np
from stable_baselines3 import A2C, PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from sb3_contrib import TRPO


def make_env(env_id: str, rank: int, seed: int = 0):
    def _init():
        env = gym.make(env_id, render_mode="human")
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


if __name__ == "__main__":
    env_id = "orbitalProbeDynamics-v1"
    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    vec_env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)

    policy_kwargs = dict(
        net_arch=dict(pi=[512, 512], vf=[512, 512]),
    )
    model = PPO("MultiInputPolicy", vec_env, verbose=1, policy_kwargs=policy_kwargs)

    # model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=100_000)

    obs = vec_env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render()
