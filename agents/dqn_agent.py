"""
DQN агент для обучения автономному вождению.

Реализует алгоритм Deep Q-Network с:
- Experience Replay Buffer
- Target Network
- Epsilon-greedy exploration
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from networks.dqn_networks import DQNNetwork
from buffers.replay_buffers import DQNBuffer


class DQNAgent:
    """Агент DQN для обучения автономному вождению."""
    
    def __init__(self, state_size, action_size, device='cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # Нейронные сети
        self.q_network = DQNNetwork(state_size, action_size).to(device)
        self.target_network = DQNNetwork(state_size, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Оптимизатор
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0003)
        
        # Гиперпараметры
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.update_target_every = 100
        
        # Буфер опыта
        self.memory = DQNBuffer(50000)
        
        # Счетчики
        self.steps = 0
        self.episodes = 0
    
    def act(self, state, training=True):
        """Выбирает действие используя epsilon-greedy стратегию."""
        state_flat = state.flatten() if hasattr(state, 'flatten') else state
        state_tensor = torch.FloatTensor(state_flat).unsqueeze(0).to(self.device)
        
        if training and np.random.random() < self.epsilon:
            action = np.random.randint(self.action_size)
        else:
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action = q_values.argmax().item()
        
        return action
    
    def remember(self, state, action, reward, next_state, done):
        """Сохраняет переход в буфер воспроизведения."""
        state_flat = state.flatten() if hasattr(state, 'flatten') else state
        next_state_flat = next_state.flatten() if hasattr(next_state, 'flatten') else next_state
        
        self.memory.push(state_flat, action, reward, next_state_flat, done)
    
    def replay(self):
        """Обновляет Q-сеть используя опыт из буфера."""
        if len(self.memory) < self.batch_size:
            return
        
        batch = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(np.array([item['state'] for item in batch])).to(self.device)
        actions = torch.LongTensor(np.array([item['action'] for item in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([item['reward'] for item in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([item['next_state'] for item in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([item['done'] for item in batch])).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * (1 - dones) * next_q_values
        
        loss = F.mse_loss(current_q_values.squeeze(1), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Обновляем epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Обновляем целевую сеть
        if self.steps % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, filename):
        """Сохраняет модель."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes
        }, filename)
    
    def load(self, filename):
        """Загружает модель."""
        checkpoint = torch.load(filename)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
