from setuptools import setup

setup(
    name="orbital_probe_dynamics",
    version="0.0.1",
    install_requires=[
        "gymnasium==0.29.1",
        "pygame==2.4.0",
        "rebound==3.28.4",
        "stable-baselines3==2.1.0",
        "tensorboard==2.15.0",
    ],
)
