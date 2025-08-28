"""
Настройки среды highway-env для обучения RL агентов.

Содержит функции для создания и конфигурации среды:
- create_highway_env() - основная среда для обучения
- create_video_env() - среда для записи видео
"""

import gymnasium as gym
import highway_env


def create_highway_env():
    """
    Создает и настраивает среду highway-env.

    Настройки оптимизированы для обучения:
    - duration: 200 - длинные эпизоды для взаимодействия с трафиком
    - vehicles_count: 25 - больше машин для сложности
    - initial_spacing: 1.5 - плотнее трафик
    - collision_reward: -15 - сильное наказание за аварии
    - simulation_frequency: 15 - частота симуляции
    - policy_frequency: 15 - частота принятия решений
    - action_type: DiscreteMetaAction - дискретные действия для highway-fast-v0
    """
    env = gym.make("highway-fast-v0", render_mode="rgb_array")

    # Настройка параметров среды
    if hasattr(env.unwrapped, "configure"):
        env.unwrapped.configure({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 5, # Видимое кол-во машин
                "absolute": False,   # Относительные координаты
                "order": "sorted",   # Сортировка по расстоянию
                "features": ["presence", "x", "y", "vx", "vy"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20]
                }
            },
            "action": {
                "type": "DiscreteMetaAction"
            },
            "lanes_count": 3,
            "vehicles_count": 25,
            "duration": 200,
            "initial_spacing": 1.5,
            "collision_reward": -15,
            "right_lane_reward": 0.6,
            "high_speed_reward": 0.4,
            "reward_speed_range": [20, 60],
            "normalize_reward": True,
            "offroad_terminal": True,
            "simulation_frequency": 15,
            "policy_frequency": 15
        })

    return env


def create_video_env():
    """
    Создает среду highway-env специально для записи видео.
    Оптимизирована для качественного рендеринга.
    """
    env = gym.make("highway-fast-v0", render_mode="rgb_array")

    # Настройка параметров среды для видео
    if hasattr(env.unwrapped, "configure"):
        env.unwrapped.configure({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 5, # Видимое кол-во машин
                "absolute": False,   # Относительные координаты
                "order": "sorted",   # Сортировка по расстоянию
                "features": ["presence", "x", "y", "vx", "vy"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20]
                }
            },
            "action": {
                "type": "DiscreteMetaAction"
            },
            "lanes_count": 3,
            "vehicles_count": 25,
            "duration": 300,  # Длиннее для лучшего демо
            "initial_spacing": 1.5,
            "collision_reward": -15,
            "right_lane_reward": 0.6,
            "high_speed_reward": 0.4,
            "reward_speed_range": [20, 60],
            "normalize_reward": True,
            "offroad_terminal": True,
            "simulation_frequency": 15,
            "policy_frequency": 15
        })

    return env
