from torchsummary import summary
import torch
from model import DQN  

ACTION_SIZE = 2 # change based on the env action size to see how the model parameters change

model = DQN(ACTION_SIZE) 
input_tensor = torch.zeros(1, 1, 80, 80)  
summary(model, input_size=input_tensor.shape[1:], device='cpu')