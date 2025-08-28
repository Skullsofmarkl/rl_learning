"""
Нейронные сети для SAC алгоритма.

Содержит Actor и Critic сети для SAC.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SACActor(nn.Module):
    """Actor сеть для SAC."""
    
    def __init__(self, state_size, action_size):
        super(SACActor, self).__init__()
        
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


class SACCritic(nn.Module):
    """Critic сеть для SAC."""
    
    def __init__(self, state_size, action_size):
        super(SACCritic, self).__init__()
        
        input_size = state_size + action_size
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.q_head = nn.Linear(128, 1)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state, action_onehot):
        x = torch.cat([state, action_onehot], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.q_head(x)
