import gymnasium as gym
import orbital_probe_dynamics
from stable_baselines3 import A2C, PPO, TD3
from sb3_contrib import TRPO

env = gym.make("CartPole-v1", render_mode="rgb_array")

model = TRPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("models/samplePoleCart")

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
