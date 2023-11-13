import gymnasium as gym
import orbital_probe_dynamics
import numpy as np
from stable_baselines3 import A2C, PPO, TD3
from sb3_contrib import TRPO

env = gym.make("orbitalProbeDynamics-v1", render_mode=None, window_size=1024)


policy_kwargs = dict(
    net_arch=dict(pi=[64, 64], vf=[64, 64]),
)
model = PPO("MultiInputPolicy", env, verbose=1, policy_kwargs=policy_kwargs)

### ===== To train a new model =====
model = PPO.load("tempModels/trial2")
model.set_env(env)
model.learn(total_timesteps=5e7)
model.save("tempModels/trial2")

# #### ===== To load an existing model =====
# env = gym.make("orbitalProbeDynamics-v1", render_mode="human", window_size=1024)
# model = PPO.load("tempModels/trial1")
# print(model)

obs, _ = env.reset()
print(obs)
for i in range(30000):
    action, _states = model.predict(obs)

    obs, rewards, dones, info, _ = env.step(action)

    print(action, rewards)
    # obs = env.step(np.array([0.3, 1]))
    env.render()

env.close()

# for i in range(1000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     vec_env.render("human")
