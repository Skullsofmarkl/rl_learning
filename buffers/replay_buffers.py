"""
Буферы опыта для различных алгоритмов RL.

Содержит базовый класс и специализированные буферы для:
- DQN
- PPO  
- SAC
- A2C
"""

import random
import numpy as np


class BaseBuffer:
    """Базовый класс для буфера опыта."""
    
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, **kwargs):
        """Добавляет переход в буфер."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = kwargs
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Случайно выбирает батч переходов."""
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        
        batch = random.sample(self.buffer, batch_size)
        return batch
    
    def __len__(self):
        return len(self.buffer)


class DQNBuffer(BaseBuffer):
    """Буфер опыта для DQN."""
    
    def push(self, state, action, reward, next_state, done):
        super().push(state=state, action=action, reward=reward, 
                    next_state=next_state, done=done)


class PPOBuffer(BaseBuffer):
    """Буфер опыта для PPO."""
    
    def push(self, state, action, reward, next_state, done, action_probs, value):
        super().push(state=state, action=action, reward=reward, 
                    next_state=next_state, done=done, action_probs=action_probs, value=value)


class SACBuffer(BaseBuffer):
    """Буфер опыта для SAC."""
    
    def push(self, state, action, reward, next_state, done):
        super().push(state=state, action=action, reward=reward, 
                    next_state=next_state, done=done)


class A2CBuffer(BaseBuffer):
    """Буфер опыта для A2C."""
    
    def push(self, state, action, reward, next_state, done, action_probs, value):
        super().push(state=state, action=action, reward=reward, 
                    next_state=next_state, done=done, action_probs=action_probs, value=value)
