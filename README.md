# DQN Framework for Web-Based Game Automation

## Overview
This project develops a customizable Deep Q-Network (DQN) framework for automating web-based games using reinforcement learning. The framework is built with PyTorch and uses Selenium for real-time game data extraction for web games. The primary application is the Chrome Dino game, serving as a baseline model. The framework is tested with multiple open ai gym environemtns to ensure compatibility and extendability. 

## Installation
1. Clone the repository: `git clone https://github.com/prabhath-r/DQN_Framework_RL-Game`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the main script: `python main.py --game_env [game_name]`

## Code Structure
- `dqn_agent.py`: Implements the DQN agent, capable of decision-making and learning.
- `environment.py`: Manages the interaction between the DQN agent and the game.
- `model.py`: Defines the neural network architecture for the DQN agent.
- `config.py`: Contains hyperparameter settings for the DQN agent.
- `train_utils.py`: Contains helper functions to dynamically save and load the previous checkpoints
- `main.py`: The main executable script for training the agent.

## Usage
- **Training the Agent**: Execute `main.py` with the desired game environment.
- **Modifying Hyperparameters**: Adjust settings in `config.py` for custom training scenarios.
- **Environment Adaptation**: Use `environment.py` to interface with new or existing game environments.

## Key Features
- **Customizable DQN Agent**: Implements state-of-the-art decision-making and learning capabilities.
- **Dynamic Environment Management**: Seamlessly integrates with various web-based game environments.
- **Flexible Neural Network Architecture**: Tailors to specific game dynamics, enhancing learning efficiency.
- **Comprehensive Training Suite**: Includes tools and scripts for effective training and performance evaluation.

## Documentation
- [Project Report](lhttps://drive.google.com/file/d/1w6oYCdXDG5010LkHaduZOur9Gla2Q5Tl/view?usp=sharing)

## References
- [Chrome Dino Game Simulation on CodePen](https://codepen.io/MysticReborn/pen/rygqao)
- [Double DQN Implementation](https://luungoc2005.github.io/blog/2020-06-15-chrome-dino-game-reinforcement-learning/)
- [Medium Post on DQN on Chrome Dino](https://medium.com/deelvin-machine-learning/how-to-play-google-chrome-dino-game-using-reinforcement-learning-d5b99a5d7e04)
- [Custom DQN Architecture Development](https://unnatsingh.medium.com/deep-q-network-with-pytorch-d1ca6f40bfda)
