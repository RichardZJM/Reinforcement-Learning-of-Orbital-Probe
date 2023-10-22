# About

This is reinforcement learning enviroment based on the Gymnasium API, https://gymnasium.farama.org. It is a simulation of a space probe which travels from Earth to a target near the outside of the solar system. The probe does not have sufficient $\delta V$ to directly thrust to the outer planet, requiring the agent to learn the foundations of orbital dynamics, espeically gravity assists.

# Installation

Clone the custom enviroment and navigate into it's root folder (orbital_probe_rl)
It is highly recommended that you use a virtual enviroment. Install the custom enviroment by running:

```bash
pip install -e orbital_probe_dynamics
```

# Important files:

| File Name                                             | Description                                                                                                                                                      |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| orbital_probe_dynamics/envs/orbital_probe_dynamics.py | Implementation of the environment using the Rebound simulation package.                                                                                          |
| ppo_training.py                                       | An example training file that utilizes the environment and Gymnasium for training. You should create a similar file for each training algorithm you want to use. |
