# DQN Framework for Web-Based Game Automation

## Overview
This project develops a customizable Deep Q-Network (DQN) framework for automating web-based games using reinforcement learning. The primary focus is on the Chrome Dino game as a baseline model. The framework is built with PyTorch and uses Selenium for real-time game data extraction.

## Code Structure
- `dqn_agent.py`: Implements the DQN agent, capable of decision-making and learning.
- `environment.py`: Manages the interaction between the DQN agent and the game.
- `model.py`: Defines the neural network architecture for the DQN agent.
- `main.py`: The main executable script for training the agent.
- `config.py`: Contains hyperparameter settings for the DQN agent.

## Current Progress
- The DQN agent is functional, interacting with the game environment, but requires further optimization and testing.
- The game environment integration allows real-time data collection, though improvements are needed for better synchronization between the agent's actions and the game's responses.
- Initial hyperparameters are set up for easy customization.

## Challenges and Future Work
- Synchronization issues between the agent's actions and game response.
- Optimization of the DQN model for faster learning and decision-making.
- Refinement of neural network dimensions and architecture for adaptability to different game environments.

## References
- [Chrome Dino Game Simulation on CodePen](https://codepen.io/MysticReborn/pen/rygqao)
- [Double DQN Implementation](https://luungoc2005.github.io/blog/2020-06-15-chrome-dino-game-reinforcement-learning/)
- [Medium Post on DQN on Chrome Dino](https://medium.com/deelvin-machine-learning/how-to-play-google-chrome-dino-game-using-reinforcement-learning-d5b99a5d7e04)
- [Custom DQN Architecture Development](https://unnatsingh.medium.com/deep-q-network-with-pytorch-d1ca6f40bfda)
