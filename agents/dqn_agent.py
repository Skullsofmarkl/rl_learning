"""
DQN (Deep Q-Network) агент для обучения автономному вождению.

DQN - это алгоритм обучения с подкреплением, который использует нейронную сеть
для аппроксимации Q-функции. Q(s,a) представляет ожидаемую награду при выполнении
действия 'a' в состоянии 's'.

Ключевые особенности DQN:
- Experience Replay Buffer: хранит и случайно выбирает опыт для обучения
- Target Network: стабильная копия основной сети для вычисления целевых значений
- Epsilon-greedy exploration: баланс между исследованием и эксплуатацией
- Off-policy обучение: может обучаться на опыте других политик

Преимущества DQN:
- Стабильное обучение благодаря Experience Replay
- Эффективное использование опыта
- Хорошая производительность на дискретных задачах
- Простота реализации и понимания

Ограничения:
- Только для дискретных действий
- Может страдать от переоценки Q-значений
- Требует много памяти для Experience Replay
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from networks.dqn_networks import DQNNetwork
from buffers.replay_buffers import DQNBuffer


class DQNAgent:
    """
    DQN агент для обучения автономному вождению.
    
    DQN агент использует одну нейронную сеть для оценки Q-значений действий.
    Алгоритм обучения:
    1. Агент взаимодействует со средой, собирая опыт
    2. Опыт сохраняется в Experience Replay Buffer
    3. Случайные батчи опыта используются для обучения
    4. Target Network обеспечивает стабильные целевые значения
    5. Epsilon-greedy стратегия балансирует exploration/exploitation
    """
    
    def __init__(self, state_size, action_size, device='cpu'):
        """
        Инициализация DQN агента.
        
        Args:
            state_size (int): размерность состояния (количество входных признаков)
            action_size (int): количество возможных действий
            device (str): устройство для вычислений ('cpu' или 'cuda')
        """
        # Основные параметры агента
        self.state_size = state_size      # Размерность состояния
        self.action_size = action_size    # Количество действий
        self.device = device              # Устройство для вычислений (CPU/GPU)
        
        # Нейронные сети DQN
        # Основная Q-сеть для обучения
        self.q_network = DQNNetwork(state_size, action_size).to(device)
        # Целевая сеть для стабильных целевых значений
        self.target_network = DQNNetwork(state_size, action_size).to(device)
        # Инициализируем целевую сеть весами основной сети
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Оптимизатор для обучения основной сети
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0003)
        
        # Гиперпараметры DQN алгоритма
        self.gamma = 0.99                 # Коэффициент дисконтирования (важность будущих наград)
        self.epsilon = 1.0                # Начальное значение epsilon для exploration
        self.epsilon_min = 0.01           # Минимальное значение epsilon
        self.epsilon_decay = 0.995        # Скорость уменьшения epsilon
        self.batch_size = 64              # Размер батча для обучения
        self.update_target_every = 100    # Частота обновления целевой сети
        
        # Буфер опыта для Experience Replay
        self.memory = DQNBuffer(50000)
        
        # Счетчики для отслеживания прогресса обучения
        self.steps = 0      # Общее количество шагов
        self.episodes = 0   # Количество завершенных эпизодов
    
    def act(self, state, training=True):
        """
        Выбирает действие используя epsilon-greedy стратегию.
        
        Epsilon-greedy стратегия:
        - С вероятностью epsilon: случайное действие (exploration)
        - С вероятностью (1-epsilon): лучшее действие по Q-значениям (exploitation)
        
        Args:
            state: текущее состояние среды
            training (bool): режим обучения (True) или тестирования (False)
            
        Returns:
            int: выбранное действие
        """
        # Подготовка состояния для нейронной сети
        state_flat = state.flatten() if hasattr(state, 'flatten') else state
        state_tensor = torch.FloatTensor(state_flat).unsqueeze(0).to(self.device)
        
        # Epsilon-greedy стратегия выбора действия
        if training and np.random.random() < self.epsilon:
            # Exploration: случайное действие
            action = np.random.randint(self.action_size)
        else:
            # Exploitation: действие с максимальным Q-значением
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action = q_values.argmax().item()
        
        return action
    
    def remember(self, state, action, reward, next_state, done):
        """
        Сохраняет переход (experience) в Experience Replay Buffer.
        
        Experience Replay позволяет:
        - Разбить корреляции между последовательными переходами
        - Повторно использовать опыт для обучения
        - Повысить стабильность обучения
        
        Args:
            state: текущее состояние среды
            action: выбранное действие
            reward: полученная награда
            next_state: следующее состояние
            done: флаг завершения эпизода
        """
        # Подготавливаем состояния для сохранения (преобразуем в одномерные массивы)
        state_flat = state.flatten() if hasattr(state, 'flatten') else state
        next_state_flat = next_state.flatten() if hasattr(next_state, 'flatten') else next_state
        
        # Сохраняем переход в буфер для последующего обучения
        self.memory.push(state_flat, action, reward, next_state_flat, done)
    
    def replay(self):
        """
        Обновляет Q-сеть используя опыт из Experience Replay Buffer.
        
        DQN алгоритм обучения:
        1. Получаем случайный батч опыта из буфера
        2. Вычисляем текущие Q-значения для выбранных действий
        3. Вычисляем целевые Q-значения используя Target Network
        4. Минимизируем TD-ошибку между текущими и целевыми Q-значениями
        5. Обновляем epsilon для уменьшения exploration
        6. Периодически обновляем Target Network
        
        Целевые Q-значения: target = r + γ * max_a' Q_target(s', a')
        TD-ошибка: loss = MSE(Q_current(s,a), target)
        """
        # Проверяем, достаточно ли опыта в буфере для обучения
        if len(self.memory) < self.batch_size:
            return
        
        # Получаем случайный батч опыта из буфера
        batch = self.memory.sample(self.batch_size)
        
        # Преобразуем данные батча в тензоры PyTorch
        states = torch.FloatTensor(np.array([item['state'] for item in batch])).to(self.device)
        actions = torch.LongTensor(np.array([item['action'] for item in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([item['reward'] for item in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([item['next_state'] for item in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([item['done'] for item in batch])).to(self.device)
        
        # ========== ВЫЧИСЛЕНИЕ Q-ЗНАЧЕНИЙ ==========
        # Получаем Q-значения для выбранных действий из основной сети
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Вычисляем целевые Q-значения используя Target Network
        with torch.no_grad():
            # Максимальные Q-значения для следующих состояний
            next_q_values = self.target_network(next_states).max(1)[0]
            # Целевые Q-значения: r + γ * max Q_target(s', a') (если эпизод не завершен)
            target_q_values = rewards + self.gamma * (1 - dones) * next_q_values
        
        # ========== ОБУЧЕНИЕ СЕТИ ==========
        # Вычисляем TD-ошибку (MSE между текущими и целевыми Q-значениями)
        loss = F.mse_loss(current_q_values.squeeze(1), target_q_values)
        
        # Градиентный спуск
        self.optimizer.zero_grad()  # Обнуляем градиенты
        loss.backward()             # Вычисляем градиенты
        # Обрезаем градиенты для стабильности обучения
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()       # Обновляем веса
        
        # ========== ОБНОВЛЕНИЕ ПАРАМЕТРОВ ==========
        # Уменьшаем epsilon для постепенного перехода от exploration к exploitation
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Периодически обновляем Target Network для стабильности
        if self.steps % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, filename):
        """
        Сохраняет состояние DQN агента в файл.
        
        Сохраняем:
        - Веса основной Q-сети
        - Веса целевой сети
        - Состояние оптимизатора (для продолжения обучения)
        - Текущее значение epsilon
        - Счетчики шагов и эпизодов
        
        Args:
            filename (str): путь к файлу для сохранения
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),        # Веса основной Q-сети
            'target_network_state_dict': self.target_network.state_dict(),  # Веса целевой сети
            'optimizer_state_dict': self.optimizer.state_dict(),        # Состояние оптимизатора
            'epsilon': self.epsilon,                                    # Текущее значение epsilon
            'steps': self.steps,                                        # Общее количество шагов
            'episodes': self.episodes                                   # Количество завершенных эпизодов
        }, filename)
    
    def load(self, filename):
        """
        Загружает состояние DQN агента из файла.
        
        Восстанавливает:
        - Веса нейронных сетей
        - Состояние оптимизатора
        - Значение epsilon
        - Счетчики шагов и эпизодов
        
        Args:
            filename (str): путь к файлу для загрузки
        """
        # Загружаем checkpoint из файла
        checkpoint = torch.load(filename)
        
        # Восстанавливаем веса нейронных сетей
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        
        # Восстанавливаем состояние оптимизатора (для продолжения обучения)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Восстанавливаем параметры обучения
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
