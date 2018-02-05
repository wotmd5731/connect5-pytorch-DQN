# -*- coding: utf-8 -*-

import math
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


class DQN(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.state_space = args.state_space
        self.fc1 = nn.Linear(self.state_space,args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size,args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size,args.action_space)
        
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return x
    
    def parameter_update(self , source):
        self.load_state_dict(source.state_dict())
  