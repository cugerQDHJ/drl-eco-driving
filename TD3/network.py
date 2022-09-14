import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, init_w=3e-3):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        
        self.max_action = max_action

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.max_action * torch.tanh(self.fc3(x)) #动作的范围在[-2,2]，而tanh的范围是[-1,1]
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, init_w=3e-3):
        super(Critic, self).__init__()
        #q1 architecture
        self.fc1_q1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2_q1 = nn.Linear(256, 256)
        self.fc3_q1 = nn.Linear(256, 1)
        
        #q2 architecture
        self.fc1_q2 = nn.Linear(state_dim + action_dim, 256)
        self.fc2_q2 = nn.Linear(256, 256)
        self.fc3_q2 = nn.Linear(256, 1)

        self.fc3_q1.weight.data.uniform_(-init_w, init_w)
        self.fc3_q1.bias.data.uniform_(-init_w, init_w)
        self.fc3_q2.weight.data.uniform_(-init_w, init_w)
        self.fc3_q2.bias.data.uniform_(-init_w, init_w)
    
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        
        q1 = F.relu(self.fc1_q1(x))
        q1 = F.relu(self.fc2_q1(q1))
        q1 = self.fc3_q1(q1)
        
        q2 = F.relu(self.fc1_q2(x))
        q2 = F.relu(self.fc2_q2(q2))
        q2 = self.fc3_q2(q2)
        
        return q1, q2
    
    #只算q1,不算q2
    def Q1(self, state, action):
        x = torch.cat([state, action], 1)
        
        q1 = F.relu(self.fc1_q1(x))
        q1 = F.relu(self.fc2_q1(q1))
        q1 = self.fc3_q1(q1)
        
        return q1