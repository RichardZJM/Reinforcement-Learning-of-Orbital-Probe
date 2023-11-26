import gymnasium as gym
import orbital_probe_dynamics
import numpy as np
from stable_baselines3 import A2C, PPO, TD3, HerReplayBuffer, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import TRPO, TQC


def make_env(env_id: str, rank: int, seed: int = 0):
    def _init():
        env = gym.make(
            env_id,
            trainingStage=69,
            maxDeviation=0.05,
        )
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=90000,
    save_path="./tempContinuousCurriculum/",
    name_prefix="stage2",
    save_replay_buffer=True,
    save_vecnormalize=True,
    verbose=2,
)


if __name__ == "__main__":
    env_id = "orbitalProbeDynamics-v1"
    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    vec_env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)

    policy_kwargs = dict(net_arch=[256, 256, 256, 256])
    model = TQC(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        gamma=0.999,
        batch_size=64,
        top_quantiles_to_drop_per_net=2,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./modelTensorBoard/",
    )

    ## ===== To train a new model =====
    # model = TQC.load(
    #     "tempCuriculumModel3/stage1_52200000_steps.zip",
    #     vec_env,
    #     verbose=1,
    #     learning_rate=3e-4,
    #     gamma=0.999,
    #     batch_size=64,
    # )
    print(model.policy)
    model.learn(total_timesteps=1e10, callback=checkpoint_callback)

    env = gym.make("orbitalProbeDynamics-v1", render_mode=None, window_size=1024)

    #### ===== To load an existing model =====
    env = gym.make(
        "orbitalProbeDynamics-v1",
        render_mode="human",
        window_size=1024,
        trainingStage=2,
        maxDeviation=1,
    )
    # model = TQC.load("tempCuriculumModel3/stage2_24480000_steps.zip")

    obs, _ = env.reset(options={"rendering": True})

    for i in range(30000):
        action, _states = model.predict(obs, deterministic=True)

        obs, rewards, dones, info, _ = env.step(action)

        print(action, rewards)
        env.render()

    env.close()
