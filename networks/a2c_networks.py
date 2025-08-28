"""
Нейронные сети для A2C алгоритма.

Содержит Actor и Critic сети для A2C.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class A2CActor(nn.Module):
    """Actor сеть для A2C."""
    
    def __init__(self, state_size, action_size):
        super(A2CActor, self).__init__()
        
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.action_head = nn.Linear(128, action_size)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.action_head(x)


class A2CCritic(nn.Module):
    """Critic сеть для A2C."""
    
    def __init__(self, state_size):
        super(A2CCritic, self).__init__()
        
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.value_head = nn.Linear(128, 1)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.value_head(x)
