"""
Буферы опыта (Experience Replay Buffers) для различных алгоритмов RL.

Experience Replay - это техника, которая позволяет алгоритмам обучения с подкреплением
сохранять и повторно использовать прошлый опыт для обучения. Это помогает:
- Разбить корреляции между последовательными переходами
- Повысить стабильность обучения
- Более эффективно использовать собранный опыт

Содержит базовый класс и специализированные буферы для:
- DQN: Experience Replay для off-policy обучения
- PPO: On-policy буфер для хранения эпизодов
- SAC: Off-policy буфер для максимальной энтропии
- A2C: On-policy буфер с дополнительными данными

Особенности реализации:
- Циркулярный буфер для эффективного использования памяти
- Случайная выборка для разбития корреляций
- Специализированные методы push для каждого алгоритма
"""

import random
import numpy as np


class BaseBuffer:
    """
    Базовый класс для буфера опыта (Experience Replay Buffer).
    
    Реализует циркулярный буфер, который:
    - Имеет фиксированную емкость
    - Заменяет старые записи новыми при переполнении
    - Поддерживает случайную выборку для обучения
    
    Принцип работы:
    1. Новые переходы добавляются в позицию self.position
    2. При достижении capacity, старые записи перезаписываются
    3. position циклически перемещается по буферу
    4. Случайная выборка обеспечивает независимость данных
    """
    
    def __init__(self, capacity=100000):
        """
        Инициализация базового буфера.
        
        Args:
            capacity (int): максимальная емкость буфера (количество переходов)
        """
        self.capacity = capacity    # Максимальная емкость буфера
        self.buffer = []           # Список для хранения переходов
        self.position = 0          # Текущая позиция для добавления новых записей
    
    def push(self, **kwargs):
        """
        Добавляет переход (experience) в буфер.
        
        Переход содержит информацию о взаимодействии агента со средой:
        - state: текущее состояние
        - action: выбранное действие
        - reward: полученная награда
        - next_state: следующее состояние
        - done: флаг завершения эпизода
        - дополнительные данные (зависят от алгоритма)
        
        Args:
            **kwargs: словарь с данными перехода
        """
        # Если буфер еще не заполнен, добавляем новую запись
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        # Записываем переход в текущую позицию
        self.buffer[self.position] = kwargs
        # Перемещаем позицию циклически (при достижении capacity возвращаемся к началу)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """
        Случайно выбирает батч переходов из буфера.
        
        Случайная выборка важна для:
        - Разбития корреляций между последовательными переходами
        - Стабилизации обучения нейронных сетей
        - Обеспечения независимости данных в батче
        
        Args:
            batch_size (int): размер батча для выборки
            
        Returns:
            list: список случайно выбранных переходов
        """
        # Если запрашиваемый размер больше доступных данных, используем все доступные
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        
        # Случайная выборка без повторений
        batch = random.sample(self.buffer, batch_size)
        return batch
    
    def __len__(self):
        """
        Возвращает текущее количество переходов в буфере.
        
        Returns:
            int: количество сохраненных переходов
        """
        return len(self.buffer)


class DQNBuffer(BaseBuffer):
    """
    Буфер опыта для DQN (Deep Q-Network) алгоритма.
    
    DQN использует Experience Replay для off-policy обучения:
    - Сохраняет переходы (state, action, reward, next_state, done)
    - Случайная выборка разбивает корреляции между последовательными переходами
    - Позволяет повторно использовать опыт для стабильного обучения
    - Критически важен для стабильности DQN алгоритма
    
    Особенности:
    - Off-policy: может обучаться на опыте других политик
    - Большая емкость буфера для разнообразия опыта
    - Простая структура данных (только основные переходы)
    """
    
    def push(self, state, action, reward, next_state, done):
        """
        Добавляет переход в DQN буфер.
        
        Args:
            state: текущее состояние среды
            action: выбранное действие
            reward: полученная награда
            next_state: следующее состояние
            done: флаг завершения эпизода
        """
        super().push(state=state, action=action, reward=reward, 
                    next_state=next_state, done=done)


class PPOBuffer(BaseBuffer):
    """
    Буфер опыта для PPO (Proximal Policy Optimization) алгоритма.
    
    PPO использует on-policy буфер с дополнительными данными:
    - Сохраняет action_probs для вычисления importance sampling ratio
    - Сохраняет value для вычисления advantages
    - Используется для нескольких эпох обучения на одном батче
    - Очищается после каждого обновления политики
    
    Особенности:
    - On-policy: обучается только на текущей политике
    - Дополнительные данные для clipped surrogate objective
    - Кратковременное хранение (эпизоды очищаются после обучения)
    """
    
    def push(self, state, action, reward, next_state, done, action_probs, value):
        """
        Добавляет переход в PPO буфер.
        
        Args:
            state: текущее состояние среды
            action: выбранное действие
            reward: полученная награда
            next_state: следующее состояние
            done: флаг завершения эпизода
            action_probs: вероятности действий от Actor
            value: оценка ценности состояния от Critic
        """
        super().push(state=state, action=action, reward=reward, 
                    next_state=next_state, done=done, action_probs=action_probs, value=value)


class SACBuffer(BaseBuffer):
    """
    Буфер опыта для SAC (Soft Actor-Critic) алгоритма.
    
    SAC использует off-policy буфер для максимальной энтропии:
    - Сохраняет переходы для off-policy обучения
    - Позволяет эффективное использование опыта
    - Поддерживает double Q-learning с двумя Critic сетями
    - Используется для soft policy updates
    
    Особенности:
    - Off-policy: может обучаться на опыте других политик
    - Максимальная энтропия для лучшего исследования
    - Эффективное использование данных благодаря off-policy обучению
    - Простая структура (аналогично DQN)
    """
    
    def push(self, state, action, reward, next_state, done):
        """
        Добавляет переход в SAC буфер.
        
        Args:
            state: текущее состояние среды
            action: выбранное действие
            reward: полученная награда
            next_state: следующее состояние
            done: флаг завершения эпизода
        """
        super().push(state=state, action=action, reward=reward, 
                    next_state=next_state, done=done)


class A2CBuffer(BaseBuffer):
    """
    Буфер опыта для A2C (Advantage Actor-Critic) алгоритма.
    
    A2C использует on-policy буфер с дополнительными данными:
    - Сохраняет action_probs для вычисления advantage
    - Сохраняет value для вычисления advantage
    - Используется для батчевой обработки on-policy данных
    - Аналогично PPO, но с синхронным обновлением
    
    Особенности:
    - On-policy: обучается только на текущей политике
    - Дополнительные данные для advantage estimation
    - Синхронное обновление Actor и Critic
    - Батчевая обработка для эффективности
    """
    
    def push(self, state, action, reward, next_state, done, action_probs, value):
        """
        Добавляет переход в A2C буфер.
        
        Args:
            state: текущее состояние среды
            action: выбранное действие
            reward: полученная награда
            next_state: следующее состояние
            done: флаг завершения эпизода
            action_probs: вероятности действий от Actor
            value: оценка ценности состояния от Critic
        """
        super().push(state=state, action=action, reward=reward, 
                    next_state=next_state, done=done, action_probs=action_probs, value=value)
