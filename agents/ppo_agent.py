"""
PPO (Proximal Policy Optimization) агент для обучения автономному вождению.

PPO - это современный on-policy алгоритм обучения с подкреплением, который
сочетает стабильность обучения с высокой производительностью.

Ключевые особенности PPO:
- Actor-Critic архитектура: отдельные сети для политики и функции ценности
- Clipped Surrogate Objective: предотвращает слишком большие обновления политики
- Generalized Advantage Estimation (GAE): улучшенная оценка преимуществ
- On-policy обучение: использует только данные от текущей политики
- Множественные эпохи обучения на одном батче данных

Математические основы:
- Clipped objective: L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
- GAE: A^GAE(γ,λ) = Σ(γλ)^l δ_{t+l}^V
- Value loss: L^VF = (V_θ(s_t) - V_target)^2
- Entropy bonus: S[π_θ](s_t) для поощрения исследования

Преимущества PPO:
- Стабильное обучение без катастрофических обновлений
- Эффективное использование данных
- Хорошая производительность на широком спектре задач
- Относительно простые гиперпараметры для настройки
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from networks.ppo_networks import PPOActor, PPOCritic
from buffers.replay_buffers import PPOBuffer


class PPOAgent:
    """
    PPO (Proximal Policy Optimization) агент для обучения автономному вождению.
    
    PPO агент реализует современный on-policy алгоритм обучения с подкреплением,
    который обеспечивает стабильное обучение через clipped surrogate objective.
    
    Архитектура:
    - Actor: нейронная сеть, определяющая политику π(a|s)
    - Critic: нейронная сеть, оценивающая функцию ценности V(s)
    - Буфер: хранит эпизоды для on-policy обучения
    
    Ключевые компоненты:
    - Clipped surrogate objective для стабильности
    - GAE для улучшенной оценки преимуществ
    - Entropy regularization для исследования
    - Множественные эпохи обучения на одном батче
    """
    
    def __init__(self, state_size, action_size, device='cpu'):
        """
        Инициализация PPO агента.
        
        Args:
            state_size (int): размер вектора состояния
            action_size (int): количество возможных действий
            device (str): устройство для вычислений ('cpu' или 'cuda')
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # ========== НЕЙРОННЫЕ СЕТИ ==========
        # Actor: определяет политику π(a|s) - вероятности действий
        self.actor = PPOActor(state_size, action_size).to(device)
        # Critic: оценивает функцию ценности V(s) - ожидаемую награду
        self.critic = PPOCritic(state_size).to(device)
        
        # ========== ОПТИМИЗАТОРЫ ==========
        # Отдельные оптимизаторы для Actor и Critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0003)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.0003)
        
        # ========== ГИПЕРПАРАМЕТРЫ PPO ==========
        self.gamma = 0.99          # Коэффициент дисконтирования для будущих наград
        self.gae_lambda = 0.95     # Параметр GAE для сглаживания преимуществ
        self.clip_ratio = 0.2      # Параметр clipping для surrogate objective
        self.entropy_coef = 0.01   # Коэффициент энтропии для исследования
        
        # ========== БУФЕР ОПЫТА ==========
        # On-policy буфер для хранения эпизодов
        self.memory = PPOBuffer(100000)
        
        # ========== СЧЕТЧИКИ ==========
        self.steps = 0             # Общее количество шагов
        self.episodes = 0          # Общее количество эпизодов
    
    def act(self, state, training=True):
        """
        Выбирает действие используя текущую политику Actor.
        
        PPO использует стохастическую политику в режиме обучения и жадную
        политику в режиме оценки. В обоих случаях возвращается информация
        о вероятностях действий для последующего обучения.
        
        Режимы работы:
        - training=True: стохастический выбор (сэмплирование из распределения)
        - training=False: жадный выбор (действие с максимальной вероятностью)
        
        Процесс выбора действия:
        1. Преобразование состояния в тензор
        2. Получение логитов действий от Actor
        3. Получение оценки ценности от Critic
        4. Выбор действия согласно режиму
        5. Вычисление вероятностей для всех действий
        
        Args:
            state: текущее состояние среды
            training (bool): режим обучения (влияет на способ выбора действия)
            
        Returns:
            tuple: (action, action_probs, value)
                - action: выбранное действие
                - action_probs: вероятности всех действий
                - value: оценка ценности состояния
        """
        # Преобразование состояния в тензор
        state_flat = state.flatten() if hasattr(state, 'flatten') else state
        state_tensor = torch.FloatTensor(state_flat).unsqueeze(0).to(self.device)
        
        # Вычисление без градиентов (inference режим)
        with torch.no_grad():
            # Получение логитов действий от Actor
            action_logits = self.actor(state_tensor)
            # Получение оценки ценности от Critic
            value = self.critic(state_tensor)
            
            if training:
                # Стохастический режим: сэмплирование из распределения
                dist = Categorical(logits=action_logits)
                action = dist.sample()
                action_probs = dist.probs
            else:
                # Жадный режим: выбор действия с максимальной вероятностью
                action = action_logits.argmax(dim=-1)
                dist = Categorical(logits=action_logits)
                action_probs = dist.probs
            
            # Преобразование в numpy массивы
            action = action.squeeze(0).cpu().numpy()
            value = value.squeeze(0).cpu().numpy()
            action_probs = action_probs.squeeze(0).cpu().numpy()
            
            # Извлечение скалярных значений из массивов
            if hasattr(action, '__len__') and hasattr(action, 'size') and action.size == 1:
                action = action.item()
            if hasattr(value, '__len__') and hasattr(value, 'size') and value.size == 1:
                value = value.item()
        
        return action, action_probs, value
    
    def remember(self, state, action, reward, next_state, done, action_probs, value):
        """
        Сохраняет переход (experience) в PPO буфер.
        
        PPO использует on-policy обучение, поэтому сохраняет дополнительные
        данные, необходимые для вычисления clipped surrogate objective:
        - action_probs: вероятности действий для importance sampling ratio
        - value: оценки ценности для вычисления advantages
        
        Эти данные используются для:
        - Вычисления ratio = π_new(a|s) / π_old(a|s)
        - Clipped surrogate objective для стабильности
        - Advantage estimation для обновления политики
        
        Args:
            state: текущее состояние среды
            action: выбранное действие
            reward: полученная награда
            next_state: следующее состояние
            done: флаг завершения эпизода
            action_probs: вероятности действий от Actor
            value: оценка ценности от Critic
        """
        # Преобразование состояний в плоские векторы
        state_flat = state.flatten() if hasattr(state, 'flatten') else state
        next_state_flat = next_state.flatten() if hasattr(next_state, 'flatten') else next_state
        
        # Извлечение вероятности выбранного действия
        if hasattr(action_probs, '__len__') and hasattr(action_probs, 'size') and action_probs.size > 1:
            # Если action_probs - массив, берем вероятность выбранного действия
            action_probs = action_probs[action] if action < len(action_probs) else 0.2
        else:
            # Если action_probs - скаляр, используем как есть
            action_probs = action_probs if hasattr(action_probs, 'item') else action_probs
        
        # Сохранение перехода в буфер
        self.memory.push(state_flat, action, reward, next_state_flat, done, action_probs, value)
    
    def update(self, states, actions, old_action_probs, advantages, returns):
        """
        Обновляет Actor и Critic сети PPO агента.
        
        PPO алгоритм обучения включает два основных этапа:
        1. Обновление Critic для точной оценки функции ценности
        2. Обновление Actor с использованием clipped surrogate objective
        
        Ключевые особенности PPO:
        - Clipped Surrogate Objective: предотвращает слишком большие обновления
        - Importance Sampling Ratio: r(θ) = π_θ(a|s) / π_θ_old(a|s)
        - Entropy Regularization: поощряет исследование
        - Gradient Clipping: стабилизирует обучение
        
        Args:
            states: массив состояний из эпизода
            actions: массив действий из эпизода
            old_action_probs: старые вероятности действий
            advantages: вычисленные преимущества (GAE)
            returns: целевые значения (advantages + values)
        """
        # Преобразование данных в тензоры
        states_flat = np.array([state.flatten() if hasattr(state, 'flatten') else state for state in states])
        states = torch.FloatTensor(states_flat).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        old_action_probs = torch.FloatTensor(np.array(old_action_probs)).to(self.device)
        advantages = torch.FloatTensor(np.array(advantages)).to(self.device)
        returns = torch.FloatTensor(np.array(returns)).to(self.device)
        
        # ========== ОБНОВЛЕНИЕ CRITIC ==========
        # Получаем текущие оценки ценности
        current_values = self.critic(states).squeeze(1)
        # MSE loss между текущими и целевыми значениями
        critic_loss = F.mse_loss(current_values, returns)
        
        # Обновление Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        # ========== ОБНОВЛЕНИЕ ACTOR ==========
        # Получаем новые вероятности действий от текущей политики
        action_logits = self.actor(states)
        dist = Categorical(logits=action_logits)
        # Извлекаем вероятности выбранных действий
        action_probs = dist.probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Вычисляем importance sampling ratio
        # r(θ) = π_θ(a|s) / π_θ_old(a|s)
        ratio = action_probs / (old_action_probs.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-8)
        
        # Clipped surrogate objective
        # L^CLIP(θ) = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        
        # Actor loss с clipped objective
        actor_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        # Добавляем entropy regularization для исследования
        entropy = dist.entropy().mean()
        actor_loss = actor_loss - self.entropy_coef * entropy
        
        # Обновление Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
    
    def compute_gae(self, rewards, values, dones, next_value):
        """
        Вычисляет Generalized Advantage Estimation (GAE).
        
        GAE - это улучшенный метод оценки преимуществ, который объединяет
        преимущества различных методов оценки (MC, TD, n-step) в единую формулу.
        
        Формула GAE:
        A^GAE(γ,λ) = Σ(γλ)^l δ_{t+l}^V
        
        где δ_t^V = r_t + γV(s_{t+1}) - V(s_t) - это TD-ошибка
        
        Параметры:
        - γ (gamma): коэффициент дисконтирования
        - λ (lambda): параметр сглаживания (0 = TD, 1 = MC)
        
        Преимущества GAE:
        - Снижает дисперсию по сравнению с MC
        - Уменьшает bias по сравнению с TD
        - Плавно интерполирует между методами
        
        Args:
            rewards: массив наград за эпизод
            values: массив оценок ценности состояний
            dones: массив флагов завершения эпизода
            next_value: оценка ценности следующего состояния
            
        Returns:
            tuple: (advantages, returns)
                - advantages: вычисленные преимущества
                - returns: целевые значения (advantages + values)
        """
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        # Обратный проход по эпизоду для вычисления GAE
        for t in reversed(range(len(rewards))):
            # Определяем значение следующего состояния
            if t == len(rewards) - 1:
                next_value_t = next_value  # Последний шаг эпизода
            else:
                next_value_t = values[t + 1]  # Следующий шаг
            
            # TD-ошибка: δ_t = r_t + γV(s_{t+1}) - V(s_t)
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            
            # GAE: A_t = δ_t + γλ(1-done_t)A_{t+1}
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        # Returns = advantages + values (для обновления Critic)
        returns = advantages + values
        return advantages, returns
    
    def save(self, filename):
        """
        Сохраняет все компоненты PPO агента в файл.
        
        Сохраняет:
        - Веса нейронных сетей (Actor и Critic)
        - Состояния оптимизаторов
        - Счетчики шагов и эпизодов
        
        Args:
            filename (str): путь к файлу для сохранения
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'steps': self.steps,
            'episodes': self.episodes
        }, filename)
    
    def load(self, filename):
        """
        Загружает все компоненты PPO агента из файла.
        
        Восстанавливает:
        - Веса нейронных сетей
        - Состояния оптимизаторов
        - Счетчики шагов и эпизодов
        
        Args:
            filename (str): путь к файлу для загрузки
        """
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
