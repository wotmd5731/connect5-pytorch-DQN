# -*- coding: utf-8 -*-


import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
#import torchvision.transforms as T
import sys



import argparse
from argument import get_args
args = get_args('DQN')

from checkerboard import Checkerboard
board_max = 10
env = Checkerboard(board_max)


#args.game = 'MountainCar-v0'
args.max_step = 100000
args.action_space = env.max_size*env.max_size
args.state_space = env.max_size*env.max_size
args.memory_capacity = 10000000
args.learn_start = 10000
#args.render= True

#from env import Env
#env = Env(args)

from memory import ReplayMemory 
memory = ReplayMemory(args)

#args.memory_capacity = 1000
#args.learn_start = 1000
#args.render= True
from agent import Agent
agent = Agent(args)

#agent.load()
#agent.target_dqn_update()



"""
define test function
"""
from plot import _plot_line
current_time = time.time()
Ts, Trewards, Qs = [], [], []
def test(main_episode):
    global current_time
    prev_time = current_time
    current_time = time.time()
    T_rewards, T_Qs = [], []
    Ts.append(main_episode)
    total_reward = 0
    
    episode = 0
    while episode < args.evaluation_episodes:
        episode += 1
        T=0
        reward_sum=0
        state = env.reset()
        while T < args.max_step:
            T += 1
            if args.render:
                env.render()
                
            action = agent.get_action(state,evaluate=True)
            next_state , reward , done, _ = env.step(action)
            state = next_state
    
            total_reward += reward
            reward_sum += reward
            if done:
                break
        T_rewards.append(reward_sum)
    ave_reward = total_reward/args.evaluation_episodes
    # Append to results
    Trewards.append(T_rewards)
#        Qs.append(T_Qs)
    
    # Plot
    _plot_line(Ts, Trewards, 'rewards_'+args.name+args.game, path='results')
#        _plot_line(Ts, Qs, 'Q', path='results')
    
    # Save model weights
#        main_dqn.save('results')
    print('episode: ',main_episode,'Evaluation Average Reward:',ave_reward, 'delta time:',current_time-prev_time)
#            if ave_reward >= 300:
#                break
    




random.seed(time.time())
"""
main loop
"""
global_count = 0
episode = 0
while episode < args.max_episode_length:
    
    T=0
    turn = 0
    max_action_value = 0
    state = env.reset()
#    args.epsilon -= 0.8/args.max_episode_length
    while T < args.max_step:
        action_value = 0
        if T%2 == 0 :
            turn = env.black
            env.change_enemy(env.white,env.black)
        else:
            turn = env.white
            env.change_enemy(env.black,env.white)
            
        if random.random() <= args.epsilon or global_count < args.learn_start:
            action = env.get_random_xy_flat()
        else:
            action, action_value = agent.get_action(state)
       
        max_action_value = max(max_action_value,action_value)
        
        next_state , reward , done, _ = env.step_flat(action,turn)
#        if args.reward_clip > 0:
#            reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
        memory.push([state, action, reward, next_state, done])
        state = next_state
        if global_count % args.replay_interval == 0 and global_count > args.learn_start:
            agent.basic_learn(memory)
        if global_count % args.target_update_interval == 0 :
            agent.target_dqn_update()
        T += 1
        global_count += 1
        
        if done :
            if args.render:
                env.render()
            break
    
    if episode%10000 ==0 :
        print('save')
        agent.save()
    print('episode : ', episode, '  step : ',T, ' max_action ',max_action_value)
    
#    if episode % args.evaluation_interval == 0 :
#        test(episode)
    episode += 1