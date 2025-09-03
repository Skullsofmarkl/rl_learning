"""
A2C (Advantage Actor-Critic) агент для обучения автономному вождению.

A2C - это алгоритм обучения с подкреплением, который объединяет:
- Actor (актор): нейронная сеть, которая определяет политику (какие действия выбирать)
- Critic (критик): нейронная сеть, которая оценивает ценность состояний

Ключевые особенности A2C:
- Синхронное обновление: Actor и Critic обновляются одновременно
- Advantage estimation: использует преимущество действий для более стабильного обучения
- Entropy regularization: добавляет энтропию для исследования новых действий
- On-policy: обучается только на текущей политике (без replay buffer)

Преимущества A2C:
- Простота реализации
- Стабильность обучения
- Хорошая производительность на непрерывных задачах
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from networks.a2c_networks import A2CActor, A2CCritic
from buffers.replay_buffers import A2CBuffer


class A2CAgent:
    """
    Агент A2C для обучения автономному вождению.
    
    A2C агент состоит из двух нейронных сетей:
    - Actor: предсказывает вероятности действий (политика π(a|s))
    - Critic: оценивает ценность состояний (value function V(s))
    
    Алгоритм обучения:
    1. Агент взаимодействует со средой, собирая опыт
    2. Вычисляет advantage (преимущество) для каждого действия
    3. Обновляет Critic для минимизации ошибки предсказания ценности
    4. Обновляет Actor для максимизации ожидаемой награды с учетом advantage
    """
    
    def __init__(self, state_size, action_size, device='cpu'):
        """
        Инициализация A2C агента.
        
        Args:
            state_size (int): размерность состояния (количество входных признаков)
            action_size (int): количество возможных действий
            device (str): устройство для вычислений ('cpu' или 'cuda')
        """
        # Основные параметры агента
        self.state_size = state_size      # Размерность состояния (например, 84 для изображения)
        self.action_size = action_size    # Количество действий (например, 4: влево, прямо, вправо, тормоз)
        self.device = device              # Устройство для вычислений (CPU/GPU)
        
        # Нейронные сети A2C
        # Actor - предсказывает вероятности действий на основе состояния
        self.actor = A2CActor(state_size, action_size).to(device)
        # Critic - оценивает ценность состояния (сколько награды ожидается)
        self.critic = A2CCritic(state_size).to(device)
        
        # Оптимизаторы для обучения нейронных сетей
        # Adam - адаптивный алгоритм градиентного спуска
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0003)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.0003)
        
        # Гиперпараметры A2C алгоритма
        self.gamma = 0.99          # Коэффициент дисконтирования (важность будущих наград)
        self.entropy_coef = 0.01   # Коэффициент энтропии (поощрение исследования)
        
        # Буфер опыта для хранения переходов (state, action, reward, next_state)
        # A2C использует on-policy обучение, но буфер помогает с батчевой обработкой
        self.memory = A2CBuffer(100000)
        
        # Счетчики для отслеживания прогресса обучения
        self.steps = 0      # Общее количество шагов
        self.episodes = 0   # Количество завершенных эпизодов
    
    def act(self, state, training=True):
        """
        Выбирает действие используя текущую политику (Actor) и оценивает ценность состояния (Critic).
        
        В A2C алгоритме:
        - Actor предсказывает вероятности действий (политика π(a|s))
        - Critic оценивает ценность состояния V(s)
        
        Args:
            state: текущее состояние среды (может быть массивом или тензором)
            training (bool): режим обучения (True) или тестирования (False)
            
        Returns:
            action: выбранное действие (индекс)
            action_probs: вероятности всех действий
            value: оценка ценности текущего состояния
        """
        # Подготовка состояния для нейронной сети
        # Преобразуем состояние в одномерный массив (если это изображение или многомерный массив)
        state_flat = state.flatten() if hasattr(state, 'flatten') else state
        # Преобразуем в тензор PyTorch и добавляем batch dimension (unsqueeze(0))
        state_tensor = torch.FloatTensor(state_flat).unsqueeze(0).to(self.device)
        
        # Вычисляем действие и ценность без вычисления градиентов (для инференса)
        with torch.no_grad():
            # Actor предсказывает логиты (сырые оценки) для каждого действия
            action_logits = self.actor(state_tensor)
            # Critic оценивает ценность текущего состояния
            value = self.critic(state_tensor)
            
            if training:
                # В режиме обучения: используем стохастическую политику для исследования
                # Создаем категориальное распределение из логитов
                dist = Categorical(logits=action_logits)
                # Сэмплируем действие согласно вероятностям (exploration)
                action = dist.sample()
                # Получаем вероятности всех действий для вычисления advantage
                action_probs = dist.probs
            else:
                # В режиме тестирования: выбираем действие с максимальной вероятностью (exploitation)
                action = action_logits.argmax(dim=-1)
                # Все равно создаем распределение для получения вероятностей
                dist = Categorical(logits=action_logits)
                action_probs = dist.probs
            
            # Преобразуем результаты обратно в numpy массивы для совместимости
            action = action.squeeze(0).cpu().numpy()        # Убираем batch dimension и переводим на CPU
            value = value.squeeze(0).cpu().numpy()          # Убираем batch dimension и переводим на CPU
            action_probs = action_probs.squeeze(0).cpu().numpy()  # Убираем batch dimension и переводим на CPU
            
            # Преобразуем скалярные тензоры в обычные числа Python
            if hasattr(action, '__len__') and hasattr(action, 'size') and action.size == 1:
                action = action.item()
            if hasattr(value, '__len__') and hasattr(value, 'size') and value.size == 1:
                value = value.item()
        
        return action, action_probs, value
    
    def remember(self, state, action, reward, next_state, done, action_probs, value):
        """
        Сохраняет переход (experience) в буфер воспроизведения.
        
        В A2C алгоритме мы сохраняем:
        - state: текущее состояние
        - action: выбранное действие
        - reward: полученная награда
        - next_state: следующее состояние
        - done: флаг завершения эпизода
        - action_probs: вероятности действий (нужны для вычисления advantage)
        - value: оценка ценности состояния (нужна для вычисления advantage)
        
        Args:
            state: текущее состояние среды
            action: выбранное действие
            reward: награда за переход
            next_state: следующее состояние
            done: флаг завершения эпизода
            action_probs: вероятности действий от Actor
            value: оценка ценности состояния от Critic
        """
        # Подготавливаем состояния для сохранения (преобразуем в одномерные массивы)
        state_flat = state.flatten() if hasattr(state, 'flatten') else state
        next_state_flat = next_state.flatten() if hasattr(next_state, 'flatten') else next_state
        
        # Сохраняем переход в буфер для последующего обучения
        self.memory.push(state_flat, action, reward, next_state_flat, done, action_probs, value)
    
    def update(self, batch_size=64):
        """
        Обновляет нейронные сети агента (Actor и Critic) используя собранный опыт.
        
        A2C алгоритм обновления:
        1. Вычисляет target values (целевые значения) для Critic
        2. Вычисляет advantages (преимущества) для Actor
        3. Обновляет Critic для минимизации ошибки предсказания ценности
        4. Обновляет Actor для максимизации ожидаемой награды с учетом advantage
        
        Args:
            batch_size (int): размер батча для обучения
        """
        # Проверяем, достаточно ли опыта в буфере для обучения
        if len(self.memory) < batch_size:
            return
        
        # Получаем случайный батч опыта из буфера
        batch = self.memory.sample(batch_size)
        
        # Преобразуем данные батча в тензоры PyTorch
        states = torch.FloatTensor(np.array([item['state'] for item in batch])).to(self.device)
        actions = torch.LongTensor(np.array([item['action'] for item in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([item['reward'] for item in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([item['next_state'] for item in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([item['done'] for item in batch])).to(self.device)
        old_action_probs = torch.FloatTensor(np.array([item['action_probs'] for item in batch])).to(self.device)
        old_values = torch.FloatTensor(np.array([item['value'] for item in batch])).to(self.device)
        
        # ========== ВЫЧИСЛЕНИЕ ADVANTAGE (ПРЕИМУЩЕСТВА) ==========
        # Advantage показывает, насколько лучше/хуже было действие по сравнению с ожиданием
        # A(s,a) = Q(s,a) - V(s) = r + γV(s') - V(s)
        with torch.no_grad():
            # Получаем оценку ценности следующего состояния от Critic
            next_values = self.critic(next_states).squeeze(1)
            # Вычисляем целевые значения: r + γV(s') (если эпизод не завершен)
            target_values = rewards + self.gamma * (1 - dones) * next_values
            # Advantage = целевое значение - старая оценка ценности
            advantages = target_values - old_values
        
        # ========== ОБНОВЛЕНИЕ CRITIC (VALUE FUNCTION) ==========
        # Critic обучается предсказывать точные значения состояний
        # Цель: минимизировать ошибку между предсказанием и целевым значением
        current_values = self.critic(states).squeeze(1)  # Текущие предсказания Critic
        critic_loss = F.mse_loss(current_values, target_values.detach())  # MSE loss
        
        # Градиентный спуск для Critic
        self.critic_optimizer.zero_grad()  # Обнуляем градиенты
        critic_loss.backward()             # Вычисляем градиенты
        # Обрезаем градиенты для стабильности обучения
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()       # Обновляем веса
        
        # ========== ОБНОВЛЕНИЕ ACTOR (ПОЛИТИКА) ==========
        # Actor обучается максимизировать ожидаемую награду с учетом advantage
        action_logits = self.actor(states)  # Получаем логиты действий
        dist = Categorical(logits=action_logits)  # Создаем распределение
        
        # Вычисляем отношение вероятностей (importance sampling ratio)
        # ratio = π_new(a|s) / π_old(a|s)
        action_probs = dist.probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        ratio = action_probs / (old_action_probs.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-8)
        
        # Actor loss = -E[ratio * advantage] (максимизируем ожидаемую награду)
        actor_loss = -(ratio * advantages.detach()).mean()
        
        # Добавляем энтропию для поощрения исследования
        # Энтропия = -Σ π(a|s) * log(π(a|s))
        entropy = dist.entropy().mean()
        actor_loss = actor_loss - self.entropy_coef * entropy
        
        # Градиентный спуск для Actor
        self.actor_optimizer.zero_grad()  # Обнуляем градиенты
        actor_loss.backward()             # Вычисляем градиенты
        # Обрезаем градиенты для стабильности обучения
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()       # Обновляем веса
    
    def save(self, filename):
        """
        Сохраняет состояние агента в файл.
        
        Сохраняем:
        - Веса нейронных сетей (Actor и Critic)
        - Состояние оптимизаторов (для продолжения обучения)
        - Счетчики шагов и эпизодов
        
        Args:
            filename (str): путь к файлу для сохранения
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),              # Веса Actor сети
            'critic_state_dict': self.critic.state_dict(),            # Веса Critic сети
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),    # Состояние оптимизатора Actor
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),  # Состояние оптимизатора Critic
            'steps': self.steps,        # Общее количество шагов
            'episodes': self.episodes   # Количество завершенных эпизодов
        }, filename)
    
    def load(self, filename):
        """
        Загружает состояние агента из файла.
        
        Восстанавливает:
        - Веса нейронных сетей
        - Состояние оптимизаторов
        - Счетчики шагов и эпизодов
        
        Args:
            filename (str): путь к файлу для загрузки
        """
        # Загружаем checkpoint из файла
        checkpoint = torch.load(filename)
        
        # Восстанавливаем веса нейронных сетей
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        
        # Восстанавливаем состояние оптимизаторов (для продолжения обучения)
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        # Восстанавливаем счетчики
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
