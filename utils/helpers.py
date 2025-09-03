"""
Вспомогательные функции для проекта обучения с подкреплением.

Этот модуль содержит утилиты, которые используются в различных частях проекта
для обеспечения воспроизводимости, записи демонстраций и отображения результатов.

Основные функции:
- set_seed(): установка семян для воспроизводимости экспериментов
- record_episode_video(): запись видео лучшего эпизода агента
- display_video(): отображение видео в Jupyter/Colab среде

Особенности:
- Обеспечивает воспроизводимость результатов через фиксированные семена
- Поддерживает запись видео для демонстрации обученных агентов
- Совместим с различными алгоритмами RL (DQN, PPO, SAC, A2C)
- Обрабатывает ошибки и предоставляет информативные сообщения
- Оптимизирован для работы в Google Colab

Используемые библиотеки:
- torch, numpy, random: для установки семян
- imageio: для записи видео
- IPython.display: для отображения в Jupyter
- cv2: для обработки изображений (если необходимо)
"""

import os
import random
import numpy as np
import torch
import imageio
import cv2
from IPython.display import Video, display, HTML


def set_seed(seed=42):
    """
    Устанавливает семена для всех используемых библиотек для обеспечения воспроизводимости.
    
    Воспроизводимость критически важна для научных экспериментов и сравнения алгоритмов.
    Эта функция устанавливает семена для:
    - Python random: для случайных операций в Python
    - NumPy: для случайных операций в NumPy
    - PyTorch: для случайных операций в PyTorch (CPU)
    - PyTorch CUDA: для случайных операций на GPU (если доступно)
    
    Args:
        seed (int): значение семени для всех генераторов случайных чисел
    """
    random.seed(seed)           # Python random
    np.random.seed(seed)        # NumPy random
    torch.manual_seed(seed)     # PyTorch CPU random
    
    # Устанавливаем семена для всех GPU (если доступны)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def record_episode_video(env, agent, algorithm_name, max_steps=1000):
    """
    Записывает видео лучшего эпизода для заданного агента.
    
    Эта функция создает демонстрационное видео, показывающее как обученный агент
    взаимодействует со средой. Видео полезно для:
    - Визуальной оценки качества обучения
    - Демонстрации результатов
    - Анализа поведения агента
    - Сравнения разных алгоритмов
    
    Процесс записи:
    1. Сброс среды и инициализация эпизода
    2. Получение действий от агента (в режиме inference)
    3. Выполнение действий в среде
    4. Запись кадров рендеринга
    5. Сохранение видео в MP4 формате
    
    Args:
        env: среда для записи (должна поддерживать render())
        agent: обученный агент для демонстрации
        algorithm_name (str): название алгоритма (DQN, PPO, SAC, A2C)
        max_steps (int): максимальное количество шагов в эпизоде
    
    Returns:
        str: путь к сохраненному видео файлу или None при ошибке
    """
    print(f"Запись видео лучшего эпизода для {algorithm_name}...")
    
    # Инициализация эпизода
    state, _ = env.reset()
    frames = []          # Список для хранения кадров
    total_reward = 0     # Общая награда за эпизод
    
    # Основной цикл записи
    for step in range(max_steps):
        # Получение действия от агента (в режиме inference)
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
        
        # Запись кадра рендеринга
        try:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        except Exception as e:
            print(f"Предупреждение: не удалось записать кадр {step}: {e}")
            continue
        
        # Обновление счетчиков
        total_reward += reward
        
        # Завершение при достижении терминального состояния
        if done:
            break
    
    # Проверка успешности записи
    if not frames:
        print(f"Не удалось записать ни одного кадра для {algorithm_name}")
        return None
    
    # Сохранение видео в MP4 формате
    video_path = f"best_episode_{algorithm_name.lower()}.mp4"
    try:
        # Используем imageio для создания видео с частотой 10 FPS
        imageio.mimsave(video_path, frames, fps=10)
        print(f"Видео сохранено: {video_path}")
        print(f"Длительность эпизода: {len(frames)} шагов")
        print(f"Общая награда: {total_reward:.2f}")
    except Exception as e:
        print(f"Ошибка при сохранении видео: {e}")
        return None
    
    return video_path


def display_video(video_path, algorithm_name):
    """
    Показывает видео в ячейке Colab.
    
    Args:
        video_path: путь к видео файлу
        algorithm_name: название алгоритма
    """
    try:
        # Проверяем существование файла
        if os.path.exists(video_path):
            print(f"\nВидео лучшего эпизода {algorithm_name}:")
            print(f"Файл: {video_path}")
            
            # Показываем видео в Colab
            video = Video(video_path, embed=True, width=640, height=480)
            display(video)
            
            # Альтернативный способ показа (если Video не работает)
            print(f"\nЕсли видео не отображается, скачайте файл: {video_path}")
            
        else:
            print(f"Видео файл не найден: {video_path}")
            
    except Exception as e:
        print(f"Ошибка при показе видео: {e}")
        print(f"Попробуйте скачать файл вручную: {video_path}")
