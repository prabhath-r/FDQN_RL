from torchsummary import summary
import torch
from model import DQN  

ACTION_SIZE = 6 ## (ChromeDino=2), (Breakout=4), (Pong=6)

model = DQN(ACTION_SIZE) 
input_tensor = torch.zeros(1, 1, 80, 80)  
summary(model, input_size=input_tensor.shape[1:], device='cpu')

