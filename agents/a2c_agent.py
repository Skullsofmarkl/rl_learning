"""
A2C агент для обучения автономному вождению.

Реализует алгоритм Advantage Actor-Critic с:
- Actor-Critic архитектурой
- Advantage estimation
- Entropy regularization
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from networks.a2c_networks import A2CActor, A2CCritic
from buffers.replay_buffers import A2CBuffer


class A2CAgent:
    """Агент A2C для обучения автономному вождению."""
    
    def __init__(self, state_size, action_size, device='cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # Нейронные сети
        self.actor = A2CActor(state_size, action_size).to(device)
        self.critic = A2CCritic(state_size).to(device)
        
        # Оптимизаторы
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0003)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.0003)
        
        # Гиперпараметры A2C
        self.gamma = 0.99
        self.entropy_coef = 0.01
        
        # Буфер опыта
        self.memory = A2CBuffer(100000)
        
        # Счетчики
        self.steps = 0
        self.episodes = 0
    
    def act(self, state, training=True):
        """Выбирает действие используя текущую политику."""
        state_flat = state.flatten() if hasattr(state, 'flatten') else state
        state_tensor = torch.FloatTensor(state_flat).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_logits = self.actor(state_tensor)
            value = self.critic(state_tensor)
            
            if training:
                dist = Categorical(logits=action_logits)
                action = dist.sample()
                action_probs = dist.probs
            else:
                action = action_logits.argmax(dim=-1)
                dist = Categorical(logits=action_logits)
                action_probs = dist.probs
            
            action = action.squeeze(0).cpu().numpy()
            value = value.squeeze(0).cpu().numpy()
            action_probs = action_probs.squeeze(0).cpu().numpy()
            
            if hasattr(action, '__len__') and hasattr(action, 'size') and action.size == 1:
                action = action.item()
            if hasattr(value, '__len__') and hasattr(value, 'size') and value.size == 1:
                value = value.item()
        
        return action, action_probs, value
    
    def remember(self, state, action, reward, next_state, done, action_probs, value):
        """Сохраняет переход в буфер воспроизведения."""
        state_flat = state.flatten() if hasattr(state, 'flatten') else state
        next_state_flat = next_state.flatten() if hasattr(next_state, 'flatten') else next_state
        
        self.memory.push(state_flat, action, reward, next_state_flat, done, action_probs, value)
    
    def update(self, batch_size=64):
        """Обновляет сети агента."""
        if len(self.memory) < batch_size:
            return
        
        batch = self.memory.sample(batch_size)
        
        states = torch.FloatTensor(np.array([item['state'] for item in batch])).to(self.device)
        actions = torch.LongTensor(np.array([item['action'] for item in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([item['reward'] for item in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([item['next_state'] for item in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([item['done'] for item in batch])).to(self.device)
        old_action_probs = torch.FloatTensor(np.array([item['action_probs'] for item in batch])).to(self.device)
        old_values = torch.FloatTensor(np.array([item['value'] for item in batch])).to(self.device)
        
        # Вычисляем преимущества (advantages)
        with torch.no_grad():
            next_values = self.critic(next_states).squeeze(1)
            target_values = rewards + self.gamma * (1 - dones) * next_values
            advantages = target_values - old_values
        
        # Обновляем Critic (Value-функция)
        current_values = self.critic(states).squeeze(1)
        critic_loss = F.mse_loss(current_values, target_values.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        # Обновляем Actor (политика)
        action_logits = self.actor(states)
        dist = Categorical(logits=action_logits)
        
        action_probs = dist.probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        ratio = action_probs / (old_action_probs.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-8)
        
        actor_loss = -(ratio * advantages.detach()).mean()
        entropy = dist.entropy().mean()
        actor_loss = actor_loss - self.entropy_coef * entropy
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
    
    def save(self, filename):
        """Сохраняет модель."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'steps': self.steps,
            'episodes': self.episodes
        }, filename)
    
    def load(self, filename):
        """Загружает модель."""
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
