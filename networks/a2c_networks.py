"""
Нейронные сети для A2C (Advantage Actor-Critic) алгоритма.

A2C использует две отдельные нейронные сети:
1. Actor (Актор) - предсказывает вероятности действий (политика π(a|s))
2. Critic (Критик) - оценивает ценность состояний (value function V(s))

Архитектура сетей:
- Полносвязные слои с ReLU активацией
- Xavier инициализация весов для стабильного обучения
- Глубокая архитектура для извлечения сложных признаков из состояния

Особенности:
- Actor выдает логиты (сырые оценки) для каждого действия
- Critic выдает скалярную оценку ценности состояния
- Обе сети имеют одинаковую архитектуру до выходного слоя
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class A2CActor(nn.Module):
    """
    Actor сеть для A2C алгоритма.
    
    Actor (Актор) - это нейронная сеть, которая определяет политику агента.
    Она принимает состояние среды и выдает логиты (сырые оценки) для каждого возможного действия.
    
    Архитектура:
    - Входной слой: state_size -> 256 нейронов
    - Скрытые слои: 256 -> 256 -> 128 нейронов
    - Выходной слой: 128 -> action_size (логиты действий)
    
    Функция активации: ReLU (Rectified Linear Unit)
    Инициализация: Xavier uniform для стабильного обучения
    """
    
    def __init__(self, state_size, action_size):
        """
        Инициализация Actor сети.
        
        Args:
            state_size (int): размерность входного состояния
            action_size (int): количество возможных действий
        """
        super(A2CActor, self).__init__()
        
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


class A2CCritic(nn.Module):
    """
    Critic сеть для A2C алгоритма.
    
    Critic (Критик) - это нейронная сеть, которая оценивает ценность состояний.
    Она принимает состояние среды и выдает скалярную оценку того, сколько 
    награды агент может ожидать, начиная с этого состояния.
    
    Архитектура:
    - Входной слой: state_size -> 256 нейронов
    - Скрытые слои: 256 -> 256 -> 128 нейронов
    - Выходной слой: 128 -> 1 (ценность состояния)
    
    Функция активации: ReLU (Rectified Linear Unit)
    Инициализация: Xavier uniform для стабильного обучения
    
    Использование:
    - Оценка V(s) для вычисления advantage: A(s,a) = Q(s,a) - V(s)
    - Целевые значения для обучения: target = r + γV(s')
    """
    
    def __init__(self, state_size):
        """
        Инициализация Critic сети.
        
        Args:
            state_size (int): размерность входного состояния
        """
        super(A2CCritic, self).__init__()
        
        # Определяем архитектуру сети (аналогично Actor)
        # Полносвязные слои с постепенным уменьшением размерности
        self.fc1 = nn.Linear(state_size, 256)      # Первый скрытый слой
        self.fc2 = nn.Linear(256, 256)             # Второй скрытый слой
        self.fc3 = nn.Linear(256, 128)             # Третий скрытый слой
        self.value_head = nn.Linear(128, 1)        # Выходной слой (ценность состояния)
        
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
        Прямой проход через Critic сеть.
        
        Args:
            x (torch.Tensor): входное состояние (batch_size, state_size)
            
        Returns:
            torch.Tensor: оценка ценности состояния (batch_size, 1)
        """
        # Прямой проход через сеть с ReLU активацией
        x = F.relu(self.fc1(x))    # Первый скрытый слой + ReLU
        x = F.relu(self.fc2(x))    # Второй скрытый слой + ReLU
        x = F.relu(self.fc3(x))    # Третий скрытый слой + ReLU
        return self.value_head(x)  # Выходной слой (ценность состояния)
