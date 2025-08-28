"""
PPO агент для обучения автономному вождению.

Реализует алгоритм Proximal Policy Optimization с:
- Actor-Critic архитектурой
- GAE (Generalized Advantage Estimation)
- Clipped surrogate objective
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from networks.ppo_networks import PPOActor, PPOCritic
from buffers.replay_buffers import PPOBuffer


class PPOAgent:
    """Агент PPO для обучения автономному вождению."""
    
    def __init__(self, state_size, action_size, device='cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # Нейронные сети
        self.actor = PPOActor(state_size, action_size).to(device)
        self.critic = PPOCritic(state_size).to(device)
        
        # Оптимизаторы
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0003)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.0003)
        
        # Гиперпараметры PPO
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_ratio = 0.2
        self.entropy_coef = 0.01
        
        # Буфер опыта
        self.memory = PPOBuffer(100000)
        
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
        
        # Извлекаем вероятность выбранного действия
        if hasattr(action_probs, '__len__') and hasattr(action_probs, 'size') and action_probs.size > 1:
            action_probs = action_probs[action] if action < len(action_probs) else 0.2
        else:
            action_probs = action_probs if hasattr(action_probs, 'item') else action_probs
        
        self.memory.push(state_flat, action, reward, next_state_flat, done, action_probs, value)
    
    def update(self, states, actions, old_action_probs, advantages, returns):
        """Обновляет сети агента."""
        # Преобразуем состояния в плоский вектор
        states_flat = np.array([state.flatten() if hasattr(state, 'flatten') else state for state in states])
        states = torch.FloatTensor(states_flat).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        old_action_probs = torch.FloatTensor(np.array(old_action_probs)).to(self.device)
        advantages = torch.FloatTensor(np.array(advantages)).to(self.device)
        returns = torch.FloatTensor(np.array(returns)).to(self.device)
        
        # Обновляем Critic
        current_values = self.critic(states).squeeze(1)
        critic_loss = F.mse_loss(current_values, returns)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        # Обновляем Actor
        action_logits = self.actor(states)
        dist = Categorical(logits=action_logits)
        action_probs = dist.probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        ratio = action_probs / (old_action_probs.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-8)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        
        actor_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        entropy = dist.entropy().mean()
        actor_loss = actor_loss - self.entropy_coef * entropy
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
    
    def compute_gae(self, rewards, values, dones, next_value):
        """Вычисляет Generalized Advantage Estimation."""
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        returns = advantages + values
        return advantages, returns
    
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
