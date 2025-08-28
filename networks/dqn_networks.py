"""
Нейронные сети для DQN алгоритма.

Содержит архитектуру Q-сети для оценки Q-значений действий.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    """Нейронная сеть для DQN."""
    
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.q_head = nn.Linear(128, action_size)
        
        # Инициализация весов
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
        return self.q_head(x)
