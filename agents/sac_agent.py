"""
SAC агент для обучения автономному вождению.

Реализует алгоритм Soft Actor-Critic с:
- Maximum entropy RL
- Twin critics
- Automatic temperature tuning
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from networks.sac_networks import SACActor, SACCritic
from buffers.replay_buffers import SACBuffer


class SACAgent:
    """Агент SAC для обучения автономному вождению."""
    
    def __init__(self, state_size, action_size, device='cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # Нейронные сети
        self.actor = SACActor(state_size, action_size).to(device)
        self.critic1 = SACCritic(state_size, action_size).to(device)
        self.critic2 = SACCritic(state_size, action_size).to(device)
        self.target_critic1 = SACCritic(state_size, action_size).to(device)
        self.target_critic2 = SACCritic(state_size, action_size).to(device)
        
        # Копируем веса в целевые сети
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Оптимизаторы
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0003)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=0.0003)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=0.0003)
        
        # Гиперпараметры SAC
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2
        self.auto_alpha = True
        
        # Автоматическая настройка температуры
        if self.auto_alpha:
            self.target_entropy = -np.log(1.0 / action_size) * 0.98
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=0.0003)
        
        # Буфер опыта
        self.memory = SACBuffer(100000)
        
        # Счетчики
        self.steps = 0
        self.episodes = 0
    
    def act(self, state, training=True):
        """Выбирает действие используя текущую политику."""
        state_flat = state.flatten() if hasattr(state, 'flatten') else state
        state_tensor = torch.FloatTensor(state_flat).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_logits = self.actor(state_tensor)
            
            if training:
                dist = Categorical(logits=action_logits / self.alpha)
                action = dist.sample()
            else:
                action = action_logits.argmax(dim=-1)
            
            action = action.squeeze(0).cpu().numpy()
            
            if hasattr(action, '__len__') and hasattr(action, 'size') and action.size == 1:
                action = action.item()
        
        return action
    
    def remember(self, state, action, reward, next_state, done):
        """Сохраняет переход в буфер воспроизведения."""
        state_flat = state.flatten() if hasattr(state, 'flatten') else state
        next_state_flat = next_state.flatten() if hasattr(next_state, 'flatten') else next_state
        
        self.memory.push(state_flat, action, reward, next_state_flat, done)
    
    def update(self, batch_size=64):
        """Обновляет все сети агента."""
        if len(self.memory) < batch_size:
            return
        
        batch = self.memory.sample(batch_size)
        
        states = torch.FloatTensor(np.array([item['state'] for item in batch])).to(self.device)
        actions = torch.LongTensor(np.array([item['action'] for item in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([item['reward'] for item in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([item['next_state'] for item in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([item['done'] for item in batch])).to(self.device)
        
        actions_onehot = F.one_hot(actions, num_classes=self.action_size).float()
        
        # Обновляем Critic сети
        with torch.no_grad():
            next_action_logits = self.actor(next_states)
            next_dist = Categorical(logits=next_action_logits / self.alpha)
            next_actions = next_dist.sample()
            next_actions_onehot = F.one_hot(next_actions, num_classes=self.action_size).float()
            
            next_log_probs = next_dist.log_prob(next_actions)
            next_entropy = -next_log_probs
            
            target_q1 = self.target_critic1(next_states, next_actions_onehot)
            target_q2 = self.target_critic2(next_states, next_actions_onehot)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + self.gamma * (1 - dones) * (target_q.squeeze(1) + self.alpha * next_entropy)
        
        # Обновляем Critic 1
        current_q1 = self.critic1(states, actions_onehot)
        critic1_loss = F.mse_loss(current_q1, target_q.unsqueeze(1))
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)
        self.critic1_optimizer.step()
        
        # Обновляем Critic 2
        current_q2 = self.critic2(states, actions_onehot)
        critic2_loss = F.mse_loss(current_q2, target_q.unsqueeze(1))
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)
        self.critic2_optimizer.step()
        
        # Обновляем Actor
        action_logits = self.actor(states)
        dist = Categorical(logits=action_logits / self.alpha)
        actions_sample = dist.sample()
        actions_sample_onehot = F.one_hot(actions_sample, num_classes=self.action_size).float()
        
        log_probs = dist.log_prob(actions_sample)
        entropy = -log_probs
        
        q1 = self.critic1(states, actions_sample_onehot)
        q2 = self.critic2(states, actions_sample_onehot)
        q = torch.min(q1, q2).squeeze(1)
        
        actor_loss = (self.alpha * entropy - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        # Автоматическая настройка температуры
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (entropy.detach() + self.target_entropy)).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = torch.exp(self.log_alpha).item()
        
        # Обновляем целевые сети
        self._update_target_networks()
    
    def _update_target_networks(self):
        """Обновляет целевые сети с использованием soft update."""
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, filename):
        """Сохраняет модель."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'alpha': self.alpha,
            'steps': self.steps,
            'episodes': self.episodes
        }, filename)
    
    def load(self, filename):
        """Загружает модель."""
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        self.alpha = checkpoint['alpha']
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
