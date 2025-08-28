"""
Модуль настройки среды highway-env.

Содержит функции для создания и конфигурации среды:
- create_highway_env() - основная среда для обучения
- create_video_env() - среда для записи видео
"""

from .highway_env import create_highway_env, create_video_env

__all__ = ['create_highway_env', 'create_video_env']
