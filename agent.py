# -*- coding: utf-8 -*-
import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os



class Basic_Agent(nn.Module):
    
    def __init__(self,args,DQN_model):
        super().__init__()
        
        self.batch_size = args.batch_size 
        self.discount = args.discount
        self.max_gradient_norm = args.max_gradient_norm
        self.epsilon = args.epsilon
        self.action_space = args.action_space
        self.hidden_size = args.hidden_size
        self.state_space = args.state_space
        
        self.main_dqn= DQN_model(args)
        self.target_dqn = DQN_model(args)
        
        if args.cuda:
            self.main_dqn.cuda()
            self.target_dqn.cuda()
        
        self.target_dqn_update()
        #target_param=list(target_dqn.parameters())
        #print("target update done ",main_param[0][0] , target_param[0][0])
        self.optimizer = optim.Adam(self.main_dqn.parameters(), lr=args.lr)
    
    def save(self,path ='./param.p'):
        torch.save(self.main_dqn.state_dict(),path)
        
    def load(self,path ='./param.p'):
        if os.path.exists(path):
            self.main_dqn.load_state_dict(torch.load(path))
        else :
            print("file not exist")
    
    def target_dqn_update(self):
        self.target_dqn.parameter_update(self.main_dqn)
        
    def get_action(self,state):
        ret = self.main_dqn(Variable(state,volatile=True).type(torch.FloatTensor).view(1,-1))
        action_value = ret.max(1)[0].data[0]
        action = ret.max(1)[1].data[0] #return max index call [1] 
        return action, action_value
        
    
    def learn(self,memory):
        random.seed(time.time())
        
        batch = memory.sample(self.batch_size)
        [states, actions, rewards, next_states, dones] = zip(*batch)
        state_batch = Variable( torch.stack(states,0).type(torch.FloatTensor))
        action_batch = Variable(torch.LongTensor(actions))
        reward_batch = Variable(torch.FloatTensor(rewards))
        next_states_batch = Variable(torch.stack(next_states,0).type(torch.FloatTensor))
        done_batch = Variable(torch.FloatTensor(dones))
        done_batch = -done_batch +1

        
        state_action_values = self.main_dqn(state_batch.view(self.batch_size,-1)).gather(1, action_batch.view(-1,1)).view(-1)
        next_states_batch.volatile = True
        next_state_values = self.target_dqn(next_states_batch.view(self.batch_size,-1)).max(1)[0]
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.discount * done_batch) + reward_batch
        expected_state_action_values.volatile = False
        loss = F.mse_loss(state_action_values, expected_state_action_values)        
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.main_dqn.parameters(), self.max_gradient_norm)  # Clip gradients (normalising by max value of gradient L2 norm)
        self.optimizer.step()
          
        
class Agent_conv2d(Basic_Agent):
    def __init__(self,args,DQN_model):
        super().__init__(args,DQN_model)
        
    
    def get_action(self,state):
        ret = self.main_dqn(Variable(state.unsqueeze(0).unsqueeze(0),volatile=True).type(torch.FloatTensor))
        action_value = ret.max(1)[0].data[0]
        action = ret.max(1)[1].data[0] #return max index call [1] 
        return action, action_value
        
    
    def learn(self,memory):
        random.seed(time.time())
        
        batch = memory.sample(self.batch_size)
        [states, actions, rewards, next_states, dones] = zip(*batch)
        state_batch = Variable( torch.stack(states,0).type(torch.FloatTensor),volatile = True).unsqueeze(1)
        action_batch = Variable(torch.LongTensor(actions),volatile = True)
        reward_batch = Variable(torch.FloatTensor(rewards),volatile = True)
        next_states_batch = Variable(torch.stack(next_states,0).type(torch.FloatTensor),volatile = True).unsqueeze(1)
        done_batch = Variable(torch.FloatTensor(dones),volatile = True)
        done_batch = -done_batch +1


                
        state_action_values = self.main_dqn(state_batch).gather(1, action_batch.view(-1,1)).view(-1)
        state_action_values.volatile = False
        state_action_values.requires_grad = True
        
        next_state_values = self.target_dqn(next_states_batch).max(1)[0]
        next_state_values.volatile = False
        next_state_values.requires_grad = True
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.discount * done_batch) + reward_batch
        expected_state_action_values.volatile = False
        expected_state_action_values.requires_grad = True
        
        loss = F.mse_loss(state_action_values, expected_state_action_values)        
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.main_dqn.parameters(), self.max_gradient_norm)  # Clip gradients (normalising by max value of gradient L2 norm)
        self.optimizer.step()
    
    
    
    
    
    
    
    
    