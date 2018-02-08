# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 02:19:46 2018

@author: JAE
"""

import torch.multiprocessing as mp
#from model import MyModel
import sys
import time 

def train(rank):
    
    print(rank)
    time.sleep(rank)
    sys.stdout.flush()
    # Construct data_loader, optimizer, etc.
    for i in range(10):
        print(i)
    
#    for data, labels in :
#        optimizer.zero_grad()
#        loss_fn(model(data), labels).backward()
#        optimizer.step()  # This will update the shared parameters
    return rank

if __name__ == '__main__':
    num_processes = 4
#    model = MyModel()
    # NOTE: this is required for the ``fork`` method to work
#    model.share_memory()
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=train, args=(rank,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
        
    print('done')