# inm707 - Task 3
PPO algorithm implementation for solving OpenAI CartPole-v0 environment.

## Requirements
yaml (https://pypi.org/project/PyYAML/)
pandas
matplotlib.pyplot
numpy
gym
torch

## Files
- `task_3.ipynb` : Presentation of Task
- `experiment.py`: Experimentation with different hyper-parameters from .yaml file
- `model.py`: Implementation of PPO algorithm to learn to play the game.
- `memory.py`: Saves trajectory items to memory and retrieves from memory. 
- `PPO_learning.py`: Main loop for sampling the environment and optimization of networks. 
