# file to configure settings for the DQN agent and environment
# this acts file to manage the hyperparameters that decide the performance of our RL agent

# GAME_ENV = 'Breakout'
# GAME_ENV = 'ChromeDino'
# GAME_ENV = 'Pong'
GAME_ENV = None 


# STATE_SIZE = (1, 80, 80)  # image state size (channels, height, width)

GAMMA = 0.97  # discount factor
EPSILON = 0.9  # starting value of epsilon for the epsilon-greedy policy
EPSILON_MIN = 0.05  # Minimum value of epsilon as learning progress
EPSILON_DECAY = 0.99  # factor per episode for decreasing epsilon
LEARNING_RATE = 0.0005  # learning rate
MEMORY_SIZE = 1000000  # replay buffer size
BATCH_SIZE = 1024  # training batch size
LR_SCHEDULER_STEP_SIZE = 100  # no episodes after which to reduce the learning rate
LR_SCHEDULER = 0.9 

NUM_EPISODES = 10000