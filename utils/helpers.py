"""
Вспомогательные функции для RL проекта.

Содержит:
- set_seed() - установка семян для воспроизводимости
- record_episode_video() - запись видео эпизода
- display_video() - показ видео
"""

import os
import random
import numpy as np
import torch
import imageio
import cv2
from IPython.display import Video, display, HTML


def set_seed(seed=42):
    """Устанавливает семена для воспроизводимости результатов."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def record_episode_video(env, agent, algorithm_name, max_steps=1000):
    """
    Записывает видео лучшего эпизода для заданного агента.
    
    Args:
        env: среда для записи
        agent: обученный агент
        algorithm_name: название алгоритма
        max_steps: максимальное количество шагов
    
    Returns:
        str: путь к сохраненному видео
    """
    print(f"Запись видео лучшего эпизода для {algorithm_name}...")
    
    state, _ = env.reset()
    frames = []
    total_reward = 0
    
    for step in range(max_steps):
        # Получаем действие от агента
        if algorithm_name == "DQN" or algorithm_name == "SAC":
            action = agent.act(state, training=False)
        else:
            # PPO и A2C возвращают кортеж из трех значений
            action_result = agent.act(state, training=False)
            if isinstance(action_result, tuple) and len(action_result) >= 1:
                action = action_result[0]
            else:
                action = action_result
        
        # Выполняем шаг
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Записываем кадр
        try:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        except Exception as e:
            print(f"Предупреждение: не удалось записать кадр {step}: {e}")
            continue
        
        total_reward += reward
        
        if done:
            break
    
    if not frames:
        print(f"Не удалось записать ни одного кадра для {algorithm_name}")
        return None
    
    # Сохраняем видео
    video_path = f"best_episode_{algorithm_name.lower()}.mp4"
    try:
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
