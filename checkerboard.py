# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 20:29:26 2018

@author: JAE
"""
import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import display


class Checkerboard():
    empty = 0
    black = 1
    white = 2
    block = 3
    
    def __init__(self,max_size):
        self.inline_draw = True
        self.max_size = max_size
        self.board = [[ self.empty ]*max_size for i in range(max_size)]  
            
        self.fig = plt.figure(figsize=(3,3))
        
        self.ax = self.fig.add_subplot(1,1,1)
        self.ax.set(xlim=[-1, max_size], ylim=[-1, max_size], title='Example', xlabel='xAxis', ylabel='yAxis')
        # Major ticks every 20, minor ticks every 5
        major_ticks = np.arange(0, max_size+1, 1)
#        minor_ticks = np.arange(0, max_size+1, 1)
        self.ax.set_xticks(major_ticks)
#        self.ax.set_xticks(minor_ticks, minor=True)
        self.ax.set_yticks(major_ticks)
#        self.ax.set_yticks(minor_ticks, minor=True)
        # And a corresponding grid
        self.ax.grid(which='both')
        # Or if you want different settings for the grids:
#        self.ax.grid(which='minor', alpha=0.5)
        self.ax.grid(which='major', alpha=0.5)
        self.ax.grid(color='black', linestyle='-', linewidth=0.5)
#        plt.xticks(range(11))
#        plt.show()
        
        
    def __repr__(self):
        return "info  max_size %d " % (self.max_size)

    def __str__(self):
#        for i in reversed(range(self.max_size)):
        for i in range(self.max_size):
            print(i,self.board[i])
        return '-----end-----'
    
    def reset(self):
        for y in range(self.max_size):
            for x in range(self.max_size):
                self.board[y][x] = 0
        self.ax.patches.clear()
        ss = torch.LongTensor(self.board)
        return ss
    
    def _check_rec(self, x, y, dx, dy , stone):
        if x<0 or x>=self.max_size or y<0 or y>=self.max_size or self.get_xy(x,y)!=stone:
            return 0
        #board data == stone   . next rec 
        return self._check_rec(x+dx,y+dy,dx,dy,stone) + 1
    
    def _check_done(self,x,y,stone):
        'stone으로 들어온게 5개 만들면 끝 과 리워드 +1 '
        '아니면 리턴 0 '
        delta = [[1,0],[0,1],[1,1],[1,-1]]
        max_ret = 0
        for dx,dy in delta:
            ret = self._check_rec(x+dx,y+dy,dx,dy,stone) + self._check_rec(x-dx,y-dy,-dx,-dy,stone) + 1
            max_ret = max(max_ret,ret)
        
        print('max ret' , max_ret)
        # 5개 완성 시 리턴 1 
        if max_ret == 5:
            return 1
        #6개 완성시 자동 패배
        elif max_ret == 6:
            return -1
        
        return 0

    def step_flat(self, num , stone):
        return self.step(int(num%self.max_size),int(num/self.max_size),stone)
        
    def step(self, x, y, stone):
        self._set_xy(x,y,stone)
        ss_ = torch.LongTensor(self.board)
        dd = self._check_done(x,y,stone)
        rr = dd #win 
        return ss_ , rr , dd , 0
    
    def _set_xy(self, x, y, stone):
        self.board[y][x] = stone
        if stone == self.black:
            ax_stone = patches.Circle((x, y), 0.5, facecolor='black', edgecolor='black',linewidth=1)
            self.ax.add_patch(ax_stone)
        elif stone == self.white:
            ax_stone = patches.Circle((x, y), 0.5, facecolor='white', edgecolor='black',linewidth=1)
            self.ax.add_patch(ax_stone)
        elif stone == self.block:
            ax_stone = patches.Circle((x, y), 0.5, facecolor='blue', edgecolor='black',linewidth=1)
            self.ax.add_patch(ax_stone)
        pass
    def get_xy(self, x ,y ):
        return self.board[y][x]
    
    def get_random_xy(self):
        x,y = random.randint(0,self.max_size-1),random.randint(0,self.max_size-1)
        while not board.get_xy(x,y)==board.empty:
            x,y = random.randint(0,self.max_size-1),random.randint(0,self.max_size-1)
        return x,y
    def get_random_xy_flat(self):
        x,y = random.randint(0,self.max_size-1),random.randint(0,self.max_size-1)
        while not self.get_xy(x,y)==self.empty:
            x,y = random.randint(0,self.max_size-1),random.randint(0,self.max_size-1)
        
        return x+y*self.max_size
    
    def change_enemy(self, from_num, to_num):
        for y in range(self.max_size):
            for x in range(self.max_size):
                if self.board[y][x] == from_num :
                    self.board[y][x] = to_num
                elif self.board[y][x] == to_num :
                    self.board[y][x] = from_num

    def draw(self):
#        plt.clf()
#        self.ax.draw()
#        self.ax.figure.canvas.draw()
#        self.fig.canvas.draw()
#        self.fig.update()
#        self.ax.update()
#        plt.draw()
#        self.fig.clf()
        plt.show()
#        plt.pause(0.001)
        # ipython command 
        if self.inline_draw:
            display(self.fig)
        pass
    def render(self):
        self.draw()



if __name__=="__main__":
    board = Checkerboard(10)
    board.reset()
    
    for i in range(100):
        "x,y = black agent .get_action(state)"
        x,y = board.get_random_xy()
        ss_ , rr, dd,_ = board.step(x,y,board.black)
        board.draw()
        if dd:
            print("done black win")
            break
        elif dd==-1:
            print("connect 6 black lose")
            break
        
        "x,y = white agent .get_action(state)"
        x,y = board.get_random_xy()
        ss_ , rr, dd,_ = board.step(x,y,board.white)
        board.draw()
        if dd:
            print("done white win")
            break
        elif dd==-1:
            print("connect 6 white lose")
            break
        
    


    
    
    
    