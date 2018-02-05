# -*- coding: utf-8 -*-
import random
from collections import namedtuple
import torch
from torch.autograd import Variable

import numpy as np



class ReplayMemory(object):
    def __init__(self, args):
        self.capacity = args.memory_capacity
        self.memory = []

    def push(self, args):
        if len(self.memory) > self.capacity:
            #overflow mem cap pop the first element
            self.memory.pop(0)
        self.memory.append(args)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



