"""
Нейронные сети для SAC (Soft Actor-Critic) алгоритма.

SAC использует три нейронные сети:
1. Actor (Актор) - предсказывает вероятности действий (политика π(a|s))
2. Critic Q1 - первая Q-функция для оценки Q-значений
3. Critic Q2 - вторая Q-функция для оценки Q-значений (double Q-learning)

Архитектура сетей:
- Полносвязные слои с ReLU активацией
- Xavier инициализация весов для стабильного обучения
- Глубокая архитектура для извлечения сложных признаков

Особенности SAC:
- Off-policy алгоритм с максимальной энтропией
- Double Q-learning для снижения переоценки Q-значений
- Soft policy updates для стабильного обучения
- Автоматическая настройка температуры (α) для энтропии
- Target networks для стабильных целевых значений
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SACActor(nn.Module):
    """
    Actor сеть для SAC алгоритма.
    
    Actor (Актор) - это нейронная сеть, которая определяет стохастическую политику агента.
    Она принимает состояние среды и выдает логиты (сырые оценки) для каждого возможного действия.
    
    Архитектура:
    - Входной слой: state_size -> 256 нейронов
    - Скрытые слои: 256 -> 256 -> 128 нейронов
    - Выходной слой: 128 -> action_size (логиты действий)
    
    Функция активации: ReLU (Rectified Linear Unit)
    Инициализация: Xavier uniform для стабильного обучения
    
    SAC особенности:
    - Стохастическая политика для исследования
    - Максимизация энтропии для лучшего исследования
    - Soft policy updates для стабильного обучения
    - Автоматическая настройка температуры энтропии
    """
    
    def __init__(self, state_size, action_size):
        """
        Инициализация Actor сети.
        
        Args:
            state_size (int): размерность входного состояния
            action_size (int): количество возможных действий
        """
        super(SACActor, self).__init__()
        
        # Определяем архитектуру сети
        # Полносвязные слои с постепенным уменьшением размерности
        self.fc1 = nn.Linear(state_size, 256)      # Первый скрытый слой
        self.fc2 = nn.Linear(256, 256)             # Второй скрытый слой
        self.fc3 = nn.Linear(256, 128)             # Третий скрытый слой
        self.action_head = nn.Linear(128, action_size)  # Выходной слой (логиты действий)
        
        # Инициализируем веса сети для стабильного обучения
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """
        Инициализация весов нейронной сети.
        
        Используется Xavier uniform инициализация:
        - Веса инициализируются из равномерного распределения
        - Дисперсия зависит от размера слоя
        - Помогает избежать проблем с исчезающими/взрывающимися градиентами
        
        Args:
            module: модуль нейронной сети для инициализации
        """
        if isinstance(module, nn.Linear):
            # Xavier uniform инициализация весов
            torch.nn.init.xavier_uniform_(module.weight)
            # Инициализация смещений нулями
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Прямой проход через Actor сеть.
        
        Args:
            x (torch.Tensor): входное состояние (batch_size, state_size)
            
        Returns:
            torch.Tensor: логиты действий (batch_size, action_size)
        """
        # Прямой проход через сеть с ReLU активацией
        x = F.relu(self.fc1(x))    # Первый скрытый слой + ReLU
        x = F.relu(self.fc2(x))    # Второй скрытый слой + ReLU
        x = F.relu(self.fc3(x))    # Третий скрытый слой + ReLU
        return self.action_head(x) # Выходной слой (без активации - логиты)


class SACCritic(nn.Module):
    """
    Critic сеть для SAC алгоритма.
    
    Critic (Критик) - это нейронная сеть, которая оценивает Q-значения действий.
    Она принимает состояние и действие (в one-hot формате) и выдает Q-значение.
    
    Архитектура:
    - Входной слой: (state_size + action_size) -> 256 нейронов
    - Скрытые слои: 256 -> 256 -> 128 нейронов
    - Выходной слой: 128 -> 1 (Q-значение)
    
    Функция активации: ReLU (Rectified Linear Unit)
    Инициализация: Xavier uniform для стабильного обучения
    
    SAC особенности:
    - Double Q-learning: две независимые Q-сети для снижения переоценки
    - Принимает состояние и действие как входные данные
    - Target networks для стабильных целевых значений
    - Soft updates для плавного обновления target networks
    """
    
    def __init__(self, state_size, action_size):
        """
        Инициализация Critic сети.
        
        Args:
            state_size (int): размерность входного состояния
            action_size (int): количество возможных действий
        """
        super(SACCritic, self).__init__()
        
        # Входной размер = состояние + действие (в one-hot формате)
        input_size = state_size + action_size
        
        # Определяем архитектуру сети
        # Полносвязные слои с постепенным уменьшением размерности
        self.fc1 = nn.Linear(input_size, 256)      # Первый скрытый слой
        self.fc2 = nn.Linear(256, 256)             # Второй скрытый слой
        self.fc3 = nn.Linear(256, 128)             # Третий скрытый слой
        self.q_head = nn.Linear(128, 1)            # Выходной слой (Q-значение)
        
        # Инициализируем веса сети для стабильного обучения
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """
        Инициализация весов нейронной сети.
        
        Используется Xavier uniform инициализация:
        - Веса инициализируются из равномерного распределения
        - Дисперсия зависит от размера слоя
        - Помогает избежать проблем с исчезающими/взрывающимися градиентами
        
        Args:
            module: модуль нейронной сети для инициализации
        """
        if isinstance(module, nn.Linear):
            # Xavier uniform инициализация весов
            torch.nn.init.xavier_uniform_(module.weight)
            # Инициализация смещений нулями
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state, action_onehot):
        """
        Прямой проход через Critic сеть.
        
        Args:
            state (torch.Tensor): входное состояние (batch_size, state_size)
            action_onehot (torch.Tensor): действие в one-hot формате (batch_size, action_size)
            
        Returns:
            torch.Tensor: Q-значение (batch_size, 1)
        """
        # Объединяем состояние и действие в один тензор
        x = torch.cat([state, action_onehot], dim=1)
        
        # Прямой проход через сеть с ReLU активацией
        x = F.relu(self.fc1(x))    # Первый скрытый слой + ReLU
        x = F.relu(self.fc2(x))    # Второй скрытый слой + ReLU
        x = F.relu(self.fc3(x))    # Третий скрытый слой + ReLU
        return self.q_head(x)      # Выходной слой (Q-значение без активации)
