"""
Функции оценки обученных агентов обучения с подкреплением.

Этот модуль предоставляет инструменты для тестирования и оценки производительности
обученных RL агентов в задаче автономного вождения.

Основные возможности:
- Оценка производительности обученных агентов
- Тестирование в режиме inference (без обучения)
- Вычисление метрик производительности
- Сравнение результатов между алгоритмами

Функции модуля:
- evaluate_agent(): основная функция оценки агента

Особенности оценки:
- Тестирование в режиме без обучения (training=False)
- Поддержка различных типов агентов (DQN, PPO, SAC, A2C)
- Автоматическое извлечение действий из кортежей
- Вычисление статистических метрик (среднее, стандартное отклонение)
- Отслеживание успешности эпизодов

Метрики оценки:
- Средняя награда за эпизоды
- Стандартное отклонение наград
- Средняя длина эпизодов
- Процент успешных эпизодов
"""

import numpy as np
from tqdm import tqdm


def evaluate_agent(env, agent, algorithm_name, episodes=20):
    """
    Оценивает производительность обученного агента в тестовых эпизодах.
    
    Эта функция тестирует обученного агента в режиме inference (без обучения)
    для оценки его реальной производительности. Важные особенности:
    
    Режим тестирования:
    - Агент работает в режиме training=False (без обновления весов)
    - Используется детерминированная или жадная политика
    - Нет случайности в выборе действий (кроме inherent stochasticity)
    
    Поддержка различных алгоритмов:
    - DQN, SAC: возвращают только действие
    - PPO, A2C: возвращают кортеж (action, action_probs, value)
    
    Метрики оценки:
    - Средняя награда и стандартное отклонение
    - Средняя длина эпизодов
    - Процент успешных эпизодов (без аварий)
    - Детальные данные по каждому эпизоду
    
    Args:
        env: среда для тестирования (highway-env)
        agent: обученный агент для оценки
        algorithm_name (str): название алгоритма (DQN, PPO, SAC, A2C)
        episodes (int): количество тестовых эпизодов
        
    Returns:
        dict: словарь с метриками оценки, содержащий:
            - mean_reward: средняя награда
            - std_reward: стандартное отклонение наград
            - mean_length: средняя длина эпизодов
            - std_length: стандартное отклонение длин
            - success_rate: процент успешных эпизодов
            - episode_rewards: список наград за эпизоды
            - episode_lengths: список длин эпизодов
            - episode_successes: список успешности эпизодов
    """
    # Инициализация списков для сбора метрик
    episode_rewards = []      # Награды за каждый эпизод
    episode_lengths = []      # Длины эпизодов (количество шагов)
    episode_successes = []    # Успешность эпизодов (1=успех, 0=авария)
    
    print(f"Оценка {algorithm_name} агента на {episodes} эпизодах...")
    
    # Основной цикл оценки
    for episode in tqdm(range(episodes), desc=f"{algorithm_name} Оценка"):
        # Сброс среды для нового эпизода
        state, _ = env.reset()
        total_reward = 0      # Общая награда за эпизод
        steps = 0             # Количество шагов в эпизоде
        
        # Цикл взаимодействия со средой
        while True:
            # Получение действия от агента в режиме inference
            if algorithm_name == "DQN" or algorithm_name == "SAC":
                # DQN и SAC возвращают только действие
                action = agent.act(state, training=False)
            else:
                # PPO и A2C возвращают кортеж (action, action_probs, value)
                action_result = agent.act(state, training=False)
                if isinstance(action_result, tuple) and len(action_result) >= 1:
                    action = action_result[0]  # Извлекаем только действие
                else:
                    action = action_result
            
            # Выполнение действия в среде
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Обновление счетчиков
            total_reward += reward
            steps += 1
            
            # Завершение эпизода при достижении терминального состояния
            if done:
                break
        
        # Сохранение метрик эпизода
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        # Успешность: 1 если эпизод завершился без аварии (не terminated)
        episode_successes.append(1 if not terminated else 0)
    
    # Вычисление статистических метрик
    metrics = {
        "mean_reward": np.mean(episode_rewards),           # Средняя награда
        "std_reward": np.std(episode_rewards),             # Стандартное отклонение наград
        "mean_length": np.mean(episode_lengths),           # Средняя длина эпизодов
        "std_length": np.std(episode_lengths),             # Стандартное отклонение длин
        "success_rate": np.mean(episode_successes),        # Процент успешных эпизодов
        "episode_rewards": episode_rewards,                # Детальные данные
        "episode_lengths": episode_lengths,
        "episode_successes": episode_successes
    }
    
    # Вывод результатов оценки
    print(f"{algorithm_name} результаты:")
    print(f"  Средняя награда: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"  Средняя длина: {metrics['mean_length']:.1f} ± {metrics['std_length']:.1f}")
    print(f"  Успешность: {metrics['success_rate']:.2%}")
    
    return metrics
