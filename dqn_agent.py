import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import DQN
from collections import deque
import config

class DQNAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(config.ACTION_SIZE).to(self.device)
        self.memory = deque(maxlen=config.MEMORY_SIZE)
        self.gamma = config.GAMMA
        self.epsilon = config.EPSILON_START
        self.epsilon_min = config.EPSILON_END
        self.epsilon_decay = config.EPSILON_DECAY
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.batch_size = config.BATCH_SIZE
        self.action_size = config.ACTION_SIZE
        self.criterion = nn.MSELoss()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        # Add a channel dimension and a batch dimension to the state
        state = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_values = self.model(state)
        return np.argmax(action_values.cpu().data.numpy())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)

        # Zero the parameter gradients
        self.optimizer.zero_grad()

        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(self.device)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(self.device)

            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model(next_state_tensor).cpu().data.numpy())

            target_f = self.model(state_tensor).cpu().data.numpy()
            target_f[0][action] = target
            target_f_tensor = torch.tensor(target_f, dtype=torch.float).to(self.device)

            output = self.model(state_tensor)
            loss = self.criterion(output, target_f_tensor)
            loss.backward()

        # Perform a single optimization step after processing the minibatch
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def learn(self):
        self.replay()
