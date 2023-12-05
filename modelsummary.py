from torchsummary import summary
import torch
from model import DQN  # Import your model here
import config

ACTION_SIZE = 6 ## (ChromeDino=2), (Breakout=4), (Pong=6)

model = DQN(ACTION_SIZE) 

input_tensor = torch.zeros(1, 1, 80, 80)  

# Use torchsummary to print the model summary
summary(model, input_size=input_tensor.shape[1:], device='cpu')

