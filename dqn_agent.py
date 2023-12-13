import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import config
from model import DQN
from collections import deque

class DQNAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_size = config.ACTION_SIZE
        self.model = DQN(self.action_size).to(self.device)
        self.memory = deque(maxlen=config.MEMORY_SIZE)
        self.gamma = config.GAMMA
        self.epsilon = config.EPSILON
        self.epsilon_min = config.EPSILON_MIN
        self.epsilon_decay = config.EPSILON_DECAY
        self.learning_rate = config.LEARNING_RATE
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=config.LR_SCHEDULER_STEP_SIZE, gamma=config.LR_SCHEDULER)
        self.batch_size = config.BATCH_SIZE
        self.criterion = nn.MSELoss()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_values = self.model(state)
        return np.argmax(action_values.cpu().data.numpy())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None  # not enough data to form a training batch

        minibatch = random.sample(self.memory, self.batch_size)
        losses = []  # To store loss for each sample in the minibatch
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
            losses.append(loss.item())

        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        average_loss = sum(losses) / len(losses) if losses else None
        return average_loss

    def save_checkpoint(self, episode_number):
        env_name = config.GAME_ENV  
        folder_name = f"{env_name.lower()}_model"  
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)  

        filename = f"{folder_name}/{env_name}_checkpoint_{episode_number}.pth"  
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'replay_memory': list(self.memory),
            'epsilon': self.epsilon
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, episode_number):
        env_name = config.GAME_ENV  
        folder_name = f"{env_name.lower()}_model"  
        filename = f"{folder_name}/{env_name}_checkpoint_{episode_number}.pth"  

        if os.path.exists(filename):
            checkpoint = torch.load(filename)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.memory = deque(checkpoint['replay_memory'], maxlen=config.MEMORY_SIZE)
            self.epsilon = checkpoint['epsilon']
        else:
            print(f"No checkpoint file found at {filename}. Training will start from scratch.")

    def learn(self):
        self.replay()
        self.scheduler.step()