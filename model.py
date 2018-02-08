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
        self.hidden_size = args.hidden_size
        self.action_space = args.action_space
        
        self.fc1 = nn.Linear(self.state_space,self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size,self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size,self.action_space)
        
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return x
    
    def parameter_update(self , source):
        self.load_state_dict(source.state_dict())
  
    

class DQN_conv2d(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.state_space = args.state_space
        self.hidden_size = args.hidden_size
        self.action_space = args.action_space
        
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 512, 3, padding=1)
        self.fc1 = nn.Linear(self.state_space * 512 , self.action_space )
        
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.fc1(x.view(x.size(0),-1)))
        return x
    
    def parameter_update(self , source):
        self.load_state_dict(source.state_dict())
  
    

class DQN_conv3d(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.state_space = args.state_space
        self.hidden_size = args.hidden_size
        self.action_space = args.action_space
        self.history_length = args.history_length

        self.conv1 = nn.Conv3d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv3d(64, 256, 3, padding=1)
#        self.conv3 = nn.Conv2d(256, 512, 3, padding=1)
        self.fc1 = nn.Linear(self.state_space * self.history_length* 256 , self.action_space )
        
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
#        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.fc1(x.view(x.size(0),-1)))
        return x
    
    def parameter_update(self , source):
        self.load_state_dict(source.state_dict())
  
    


