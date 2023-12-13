import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# # Deep DQN
# class DQN(nn.Module):
#     def __init__(self, action_size):
#         super(DQN, self).__init__()

#         # Conv layers
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
#         self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1)  
#         self.fc1_input_size = self._calculate_fc1_input_size()

#         # FC layers
#         self.fc1 = nn.Linear(self.fc1_input_size, 1024)  
#         self.fc2 = nn.Linear(1024, 512)
#         self.fc3 = nn.Linear(512, action_size)

#     def _calculate_fc1_input_size(self):
#         with torch.no_grad():
#             dummy_input = torch.zeros(1, 1, 80, 80)
#             output = self.conv4(self.conv3(self.conv2(self.conv1(dummy_input))))
#             output_size = int(np.prod(output.size()[1:]))
#             return output_size

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))  
#         x = x.view(x.size(0), -1)

#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))  
#         return self.fc3(x)

#Simple DQN
class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        # conv layers -> 4
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)  # batch norm for conv1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)   # batch norm for conv2
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)  # batch norm for conv3
        # ip of first fc layer
        self.fc1_input_size = self._calculate_fc1_input_size()
        # fc layers -> 2
        self.fc1 = nn.Linear(self.fc1_input_size, 512)
        self.fc2 = nn.Linear(512, action_size)

    def _calculate_fc1_input_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 80, 80)
            # Forward pass through conv layers without batch norm
            output = self.conv3(self.conv2(self.conv1(dummy_input)))
            output_size = int(np.prod(output.size()[1:]))
            return output_size

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)







