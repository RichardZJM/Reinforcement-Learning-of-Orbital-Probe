import gymnasium as gym
import orbital_probe_dynamics
import numpy as np
from stable_baselines3 import A2C, PPO, TD3
from sb3_contrib import TRPO

env = gym.make("orbitalProbeDynamics-v1", render_mode="human", window_size=1024)

model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=3)
# model.save("tempModels/trial1")


obs = env.reset()
print(obs)
for i in range(9000):
    obs = env.step(np.array([0.3, 1]))
    env.render()

# env.close()

# for i in range(1000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     vec_env.render("human")
