# About

This is a reinforcement learning environment based on the Gymnasium API, https://gymnasium.farama.org. It is a simulation of a space probe which travels from Earth to a target near the outside of the solar system. The probe does not have sufficient $\Delta V$ to directly thrust to the outer planet, requiring the agent to learn the foundations of orbital dynamics, especially gravity assists. Training currently using SB3, https://github.com/DLR-RM/stable-baselines3/tree/master.

# Installation

Clone the custom environment and navigate into its root folder (orbital_probe_rl)
It is highly recommended that you use a virtual environment. Install the custom environment by running:

```bash
pip install -e orbital_probe_dynamics
```

# Important files:

| File Name                                             | Description                                                                                                                                                      |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| orbital_probe_dynamics/envs/orbital_probe_dynamics.py | Implementation of the environment using the Rebound simulation package.                                                                                          |
| ppo_training.py                                       | An example training file that utilizes the environment and Gymnasium. You should create a similar file for each training algorithm you want to use. |
