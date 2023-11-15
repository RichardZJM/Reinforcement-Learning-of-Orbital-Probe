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
    vec_env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=DummyVecEnv)

    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64, 64, 64, 64, 64], vf=[64, 64, 64, 64, 64, 64]),
    )
    model = PPO("MultiInputPolicy", vec_env, verbose=1, policy_kwargs=policy_kwargs)

    # ### ===== To train a new model =====
    # model = PPO.load(
    #     "tempModels/trial4",
    #     vec_env,
    #     verbose=1,
    # )
    for i in range(1000):
        model.learn(total_timesteps=30000)
        model.save("tempModels/trial6")

    # env = gym.make("orbitalProbeDynamics-v1", render_mode=None, window_size=1024)

    #### ===== To load an existing model =====
    env = gym.make("orbitalProbeDynamics-v1", render_mode="human", window_size=1024)
    model = PPO.load("tempModels/trial5")
    print(model)

    obs, _ = env.reset(options={"rendering": True})
    print(obs)
    for i in range(30000):
        action, _states = model.predict(obs)

        obs, rewards, dones, info, _ = env.step(action)

        print(action, rewards)
        # obs = env.step(np.array([0.3, 1]))
        env.render()

    env.close()
