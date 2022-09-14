
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

#批评网络
class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, seed):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        #添加batch normalization层
        self.ln0 = nn.LayerNorm(state_size)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        #self.bn1 = nn.BatchNorm1d(hidden1_size)
        self.fc2 = nn.Linear(hidden_size+action_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        ##################################################
        #self.bn2 = nn.BatchNorm1d(hidden2_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state, action):
        # 按维数1拼接
        state = self.ln0(state)
        x = self.fc1(state)
        x = self.ln1(x)
        xs = F.relu(x)
        x = torch.cat((xs, action), dim = 1)
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        return self.fc3(x)

#选择动作的网络
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, seed):
        super(Actor, self).__init__()  
        self.seed = torch.manual_seed(seed)

        #如何添加batch normalization层
        self.ln0 = nn.LayerNorm(state_size)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, x):
        x = self.ln0(x)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        #至于actor网络最后一层为什么是tanh，论文是这样设置的,对动作的范围进行限制
        x = torch.tanh(self.fc3(x))
        return x
