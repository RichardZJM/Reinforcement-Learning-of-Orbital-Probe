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
    """Helper function to create an enviroment for parallel processing

    Args:
        env_id (str): id of the env to create
        rank (int): rank of the env
        seed (int, optional): The seed to use. Defaults to 0.
    """

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


# A checkpoint callback to be used to save periodically
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

    policy_kwargs = dict(
        net_arch=[256, 256, 256, 256]
    )  # The atrributes of the networks, critics and actors

    model = TQC(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        gamma=0.999,
        batch_size=64,
        top_quantiles_to_drop_per_net=2,  # Very important, controls the overestimation bias correction
        policy_kwargs=policy_kwargs,
        tensorboard_log="./modelTensorBoard/",
    )  # Generate the model and experiment with the hyperparameters as needed

    ## ===== To train a new model =====
    # model = TQC.load(
    #     "tempContinuousCurriculum/stage2_6120000_steps.zip",
    #     vec_env,
    #     verbose=1,
    #     learning_rate=3e-4,
    #     gamma=0.999,
    #     batch_size=64,
    # )
    # print(model.policy)
    # model.learn(total_timesteps=1e10, callback=checkpoint_callback)

    #### ===== To load an existing model =====
    env = gym.make(
        "orbitalProbeDynamics-v1",
        render_mode="human",
        window_size=1024,
        trainingStage=0,
        maxDeviation=0,
    )
    model = TQC.load("models/bestFinalModel.zip")

    # ===== Visualize the model =====
    obs, _ = env.reset(
        options={"rendering": True}
    )  # Call the reset function with rendering on

    # Visualize a maximum of 30k steps, aro. 100 years
    for i in range(30000):
        action, _states = model.predict(
            obs, deterministic=True
        )  # Use policy to predict and action
        obs, rewards, dones, info, _ = env.step(
            action
        )  # Take the action and get the reward and next observation

        print(action, rewards)
        env.render()

    env.close()
