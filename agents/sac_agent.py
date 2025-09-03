"""
SAC (Soft Actor-Critic) агент для обучения автономному вождению.

SAC - это современный off-policy алгоритм обучения с подкреплением, который
максимизирует как ожидаемую награду, так и энтропию политики.

Ключевые особенности SAC:
- Maximum Entropy RL: максимизирует награду + энтропию политики
- Twin Critics: использует два Q-критика для уменьшения переоценки
- Automatic Temperature Tuning: автоматическая настройка коэффициента энтропии
- Off-policy обучение: эффективное использование опыта
- Soft Updates: плавное обновление целевых сетей

Математические основы:
- Objective: J(π) = E[Σ γ^t (r_t + α H(π(·|s_t)))]
- Q-function: Q^π(s,a) = E[R_t + γ E[Q^π(s_{t+1}, a_{t+1}) - α log π(a_{t+1}|s_{t+1})]]
- Policy: π* = argmax_π E[Q^π(s,a) - α log π(a|s)]
- Temperature: α* = argmin_α E[-α log π(a|s) - α H_target]

Преимущества SAC:
- Стабильное и эффективное обучение
- Автоматический баланс exploration/exploitation
- Хорошая производительность на непрерывных задачах
- Относительно простые гиперпараметры
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from networks.sac_networks import SACActor, SACCritic
from buffers.replay_buffers import SACBuffer


class SACAgent:
    """
    SAC (Soft Actor-Critic) агент для обучения автономному вождению.
    
    SAC агент реализует современный off-policy алгоритм максимальной энтропии,
    который обеспечивает стабильное обучение через автоматический баланс
    между исследованием и эксплуатацией.
    
    Архитектура:
    - Actor: определяет стохастическую политику π(a|s)
    - Twin Critics: два Q-критика для уменьшения переоценки
    - Target Networks: стабилизируют обучение через soft updates
    - Temperature: автоматически настраиваемый коэффициент энтропии
    
    Ключевые компоненты:
    - Maximum entropy objective для лучшего исследования
    - Double Q-learning для стабильности
    - Automatic temperature tuning для баланса
    - Soft target updates для плавного обучения
    """
    
    def __init__(self, state_size, action_size, device='cpu'):
        """
        Инициализация SAC агента.
        
        Args:
            state_size (int): размер вектора состояния
            action_size (int): количество возможных действий
            device (str): устройство для вычислений ('cpu' или 'cuda')
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # ========== НЕЙРОННЫЕ СЕТИ ==========
        # Actor: определяет стохастическую политику π(a|s)
        self.actor = SACActor(state_size, action_size).to(device)
        
        # Twin Critics: два Q-критика для уменьшения переоценки
        self.critic1 = SACCritic(state_size, action_size).to(device)
        self.critic2 = SACCritic(state_size, action_size).to(device)
        
        # Target Networks: стабилизируют обучение
        self.target_critic1 = SACCritic(state_size, action_size).to(device)
        self.target_critic2 = SACCritic(state_size, action_size).to(device)
        
        # Инициализация целевых сетей копированием весов
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # ========== ОПТИМИЗАТОРЫ ==========
        # Отдельные оптимизаторы для каждой сети
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0003)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=0.0003)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=0.0003)
        
        # ========== ГИПЕРПАРАМЕТРЫ SAC ==========
        self.gamma = 0.99          # Коэффициент дисконтирования
        self.tau = 0.005           # Коэффициент soft updates для target networks
        self.alpha = 0.2           # Начальное значение температуры (коэффициент энтропии)
        self.auto_alpha = True     # Автоматическая настройка температуры
        
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
        """
        Выбирает действие используя текущую политику Actor.
        
        SAC использует стохастическую политику с температурой для баланса
        между исследованием и эксплуатацией. Температура α контролирует
        степень стохастичности политики.
        
        Режимы работы:
        - training=True: стохастический выбор с температурой α
        - training=False: жадный выбор (действие с максимальной вероятностью)
        
        Процесс выбора действия:
        1. Преобразование состояния в тензор
        2. Получение логитов действий от Actor
        3. Применение температуры для контроля стохастичности
        4. Выбор действия согласно режиму
        
        Args:
            state: текущее состояние среды
            training (bool): режим обучения (влияет на способ выбора действия)
            
        Returns:
            int: выбранное действие
        """
        # Преобразование состояния в тензор
        state_flat = state.flatten() if hasattr(state, 'flatten') else state
        state_tensor = torch.FloatTensor(state_flat).unsqueeze(0).to(self.device)
        
        # Вычисление без градиентов (inference режим)
        with torch.no_grad():
            # Получение логитов действий от Actor
            action_logits = self.actor(state_tensor)
            
            if training:
                # Стохастический режим: применение температуры для контроля энтропии
                # Деление на α увеличивает стохастичность (больше исследования)
                dist = Categorical(logits=action_logits / self.alpha)
                action = dist.sample()
            else:
                # Жадный режим: выбор действия с максимальной вероятностью
                action = action_logits.argmax(dim=-1)
            
            # Преобразование в numpy
            action = action.squeeze(0).cpu().numpy()
            
            # Извлечение скалярного значения
            if hasattr(action, '__len__') and hasattr(action, 'size') and action.size == 1:
                action = action.item()
        
        return action
    
    def remember(self, state, action, reward, next_state, done):
        """
        Сохраняет переход (experience) в Experience Replay Buffer.
        
        SAC использует off-policy обучение, поэтому все переходы сохраняются
        в буфер для последующего использования. Это позволяет:
        - Разбить корреляции между последовательными переходами
        - Повторно использовать опыт для стабильного обучения
        - Эффективно обучаться на большом количестве данных
        
        Args:
            state: текущее состояние среды
            action: выбранное действие
            reward: полученная награда
            next_state: следующее состояние
            done: флаг завершения эпизода
        """
        # Преобразование состояний в плоские векторы
        state_flat = state.flatten() if hasattr(state, 'flatten') else state
        next_state_flat = next_state.flatten() if hasattr(next_state, 'flatten') else next_state
        
        # Сохранение перехода в буфер
        self.memory.push(state_flat, action, reward, next_state_flat, done)
    
    def update(self, batch_size=64):
        """
        Обновляет все сети SAC агента (Actor, Twin Critics, Temperature).
        
        SAC алгоритм обучения включает несколько этапов:
        1. Обновление Twin Critics с использованием target networks
        2. Обновление Actor для максимизации Q-значений + энтропии
        3. Автоматическая настройка температуры α
        4. Soft updates целевых сетей
        
        Ключевые особенности:
        - Double Q-learning: использует минимум из двух Q-критиков
        - Maximum entropy: добавляет энтропию в целевые Q-значения
        - Automatic temperature tuning: адаптивно настраивает α
        - Soft target updates: плавно обновляет целевые сети
        
        Args:
            batch_size (int): размер батча для обучения
        """
        # Проверяем достаточность данных в буфере
        if len(self.memory) < batch_size:
            return
        
        # Получаем случайный батч из буфера
        batch = self.memory.sample(batch_size)
        
        # Преобразование данных в тензоры
        states = torch.FloatTensor(np.array([item['state'] for item in batch])).to(self.device)
        actions = torch.LongTensor(np.array([item['action'] for item in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([item['reward'] for item in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([item['next_state'] for item in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([item['done'] for item in batch])).to(self.device)
        
        # One-hot кодирование действий для Q-критиков
        actions_onehot = F.one_hot(actions, num_classes=self.action_size).float()
        
        # ========== ОБНОВЛЕНИЕ CRITIC СЕТЕЙ ==========
        # Вычисляем целевые Q-значения с использованием target networks
        with torch.no_grad():
            # Получаем действия от текущей политики для следующих состояний
            next_action_logits = self.actor(next_states)
            next_dist = Categorical(logits=next_action_logits / self.alpha)
            next_actions = next_dist.sample()
            next_actions_onehot = F.one_hot(next_actions, num_classes=self.action_size).float()
            
            # Вычисляем энтропию для следующих действий
            next_log_probs = next_dist.log_prob(next_actions)
            next_entropy = -next_log_probs
            
            # Получаем Q-значения от target networks (twin critics)
            target_q1 = self.target_critic1(next_states, next_actions_onehot)
            target_q2 = self.target_critic2(next_states, next_actions_onehot)
            # Используем минимум для уменьшения переоценки (double Q-learning)
            target_q = torch.min(target_q1, target_q2)
            
            # Целевые Q-значения с энтропией: r + γ(1-done)(Q_target + α*H)
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
        
        # ========== ОБНОВЛЕНИЕ ACTOR ==========
        # Получаем действия от текущей политики
        action_logits = self.actor(states)
        dist = Categorical(logits=action_logits / self.alpha)
        actions_sample = dist.sample()
        actions_sample_onehot = F.one_hot(actions_sample, num_classes=self.action_size).float()
        
        # Вычисляем энтропию политики
        log_probs = dist.log_prob(actions_sample)
        entropy = -log_probs
        
        # Получаем Q-значения для новых действий
        q1 = self.critic1(states, actions_sample_onehot)
        q2 = self.critic2(states, actions_sample_onehot)
        q = torch.min(q1, q2).squeeze(1)  # Используем минимум (double Q-learning)
        
        # Actor loss: максимизируем Q-значения + энтропию
        # J(π) = E[Q(s,a) - α*log π(a|s)] = E[Q(s,a) + α*H(π)]
        actor_loss = (self.alpha * entropy - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        # ========== АВТОМАТИЧЕСКАЯ НАСТРОЙКА ТЕМПЕРАТУРЫ ==========
        if self.auto_alpha:
            # Целевая энтропия: -log(1/|A|) * 0.98
            # Loss для α: минимизируем разность между текущей и целевой энтропией
            alpha_loss = -(self.log_alpha * (entropy.detach() + self.target_entropy)).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            # Обновляем α (экспоненциальная параметризация для положительности)
            self.alpha = torch.exp(self.log_alpha).item()
        
        # ========== SOFT UPDATES ЦЕЛЕВЫХ СЕТЕЙ ==========
        self._update_target_networks()
    
    def _update_target_networks(self):
        """
        Обновляет целевые сети с использованием soft update.
        
        Soft update обеспечивает плавное обновление целевых сетей:
        θ_target = τ * θ_main + (1 - τ) * θ_target
        
        Это стабилизирует обучение, предотвращая резкие изменения
        в целевых значениях, что критически важно для стабильности
        Q-learning алгоритмов.
        
        Параметр τ (tau) контролирует скорость обновления:
        - τ = 1: полное копирование (как в DQN)
        - τ < 1: плавное обновление (рекомендуется для SAC)
        """
        # Soft update для target_critic1
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Soft update для target_critic2
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, filename):
        """
        Сохраняет все компоненты SAC агента в файл.
        
        Сохраняет:
        - Веса всех нейронных сетей (Actor, Critics, Target Critics)
        - Состояния всех оптимизаторов
        - Текущее значение температуры α
        - Счетчики шагов и эпизодов
        
        Args:
            filename (str): путь к файлу для сохранения
        """
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
        """
        Загружает все компоненты SAC агента из файла.
        
        Восстанавливает:
        - Веса всех нейронных сетей
        - Состояния всех оптимизаторов
        - Значение температуры α
        - Счетчики шагов и эпизодов
        
        Args:
            filename (str): путь к файлу для загрузки
        """
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
