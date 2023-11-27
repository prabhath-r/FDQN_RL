import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1_input_size = self._calculate_fc1_input_size() #fc layer size
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc1_input_size, 512)
        self.fc2 = nn.Linear(512, action_size)

    def _calculate_fc1_input_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 80, 80)
            output = self.conv3(self.conv2(self.conv1(dummy_input)))
            output_size = int(np.prod(output.size()[1:]))
            # print(f"fc1 input size: {output_size}")  # db print 1  
            return output_size

    def forward(self, x):
        # print(f"input layer shape: {x.shape}")  # db print 2
        x = F.relu(self.conv1(x))
        # print(f"conv1 shape: {x.shape}") 
        x = F.relu(self.conv2(x))
        # print(f"conv2 shape: {x.shape}") 
        x = F.relu(self.conv3(x))
        # print(f"conv3 shape: {x.shape}") 
        x = x.view(x.size(0), -1)
        # print(f"flatten output size: {x.shape}") 

        x = F.relu(self.fc1(x))
        # print(f"fc1 shape: {x.shape}")

        return self.fc2(x)