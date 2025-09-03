"""
Основной скрипт для сравнительного анализа алгоритмов обучения с подкреплением.

Этот скрипт является центральной точкой входа для полного сравнительного анализа
четырех основных алгоритмов RL в задаче автономного вождения:

- DQN (Deep Q-Network): off-policy алгоритм с Experience Replay
- PPO (Proximal Policy Optimization): on-policy алгоритм с clipped objective  
- SAC (Soft Actor-Critic): off-policy алгоритм с максимальной энтропией
- A2C (Advantage Actor-Critic): on-policy алгоритм с advantage estimation

Основные возможности:
1. Полный сравнительный анализ всех алгоритмов
2. Быстрый тест для проверки работоспособности
3. Демонстрация отдельного алгоритма
4. Автоматическое сохранение моделей и результатов
5. Создание видео демонстраций
6. Генерация сравнительных графиков и отчетов

Структура анализа:
1. Инициализация среды и агентов
2. Обучение всех алгоритмов
3. Оценка производительности
4. Запись видео демонстраций
5. Сравнительный анализ результатов
6. Сохранение результатов и моделей

Использование:
- run_complete_comparison(): полный анализ всех алгоритмов
- run_quick_test(): быстрый тест (20 эпизодов)
- demonstrate_algorithm('DQN'): демонстрация конкретного алгоритма
"""

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime

# Импорт модулей проекта
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.sac_agent import SACAgent
from agents.a2c_agent import A2CAgent
from environment.highway_env import create_highway_env, create_video_env
from training.training_functions import (
    train_dqn_agent, train_ppo_agent, train_sac_agent, train_a2c_agent
)
from training.evaluation import evaluate_agent
from utils.helpers import set_seed, record_episode_video, display_video
from analysis.comparison import (
    create_comparison_dataframe, create_performance_summary, 
    save_comparison_results
)
from analysis.visualization import plot_comparison_results


def run_complete_comparison(episodes=100, max_steps=1000, save_models=True):
    """
    Запускает полный сравнительный анализ всех 4 алгоритмов RL.
    
    Эта функция выполняет комплексный анализ производительности четырех основных
    алгоритмов обучения с подкреплением в задаче автономного вождения.
    
    Процесс анализа:
    1. Инициализация среды highway-env и всех агентов
    2. Обучение каждого алгоритма на заданном количестве эпизодов
    3. Оценка производительности обученных агентов
    4. Запись видео демонстраций лучших эпизодов
    5. Сравнительный анализ результатов с построением графиков
    6. Сохранение моделей, результатов и видео
    
    Особенности:
    - Автоматическое определение размеров состояния и действий
    - Поддержка как CPU, так и GPU вычислений
    - Обработка ошибок для каждого этапа
    - Детальное логирование прогресса
    - Сохранение всех результатов для последующего анализа
    
    Args:
        episodes (int): количество эпизодов для обучения каждого алгоритма
        max_steps (int): максимальное количество шагов в эпизоде
        save_models (bool): сохранять ли обученные модели в файлы
    
    Returns:
        dict: словарь с результатами сравнения, содержащий:
            - comparison_df: DataFrame с метриками сравнения
            - training_results: результаты обучения всех алгоритмов
            - evaluation_results: результаты оценки всех алгоритмов
            - training_times: время обучения каждого алгоритма
            - agents: словарь обученных агентов
            - video_paths: пути к записанным видео
    """
    
    print("ЗАПУСК ПОЛНОГО СРАВНИТЕЛЬНОГО АНАЛИЗА АЛГОРИТМОВ RL")
    print("=" * 80)
    print(f"Параметры анализа:")
    print(f"  • Эпизоды обучения: {episodes}")
    print(f"  • Максимум шагов: {max_steps}")
    print(f"  • Сохранение моделей: {'Да' if save_models else 'Нет'}")
    print("=" * 80)
    
    # Проверка доступности CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"PyTorch версия: {torch.__version__}")
    print(f"Используемое устройство: {device}")
    if torch.cuda.is_available():
        print(f"CUDA устройство: {torch.cuda.get_device_name(0)}")
    
    # Установка семян
    set_seed(42)
    
    # Создание среды
    print("\nСоздание среды highway-env...")
    env = create_highway_env()
    
    # Получаем размеры состояния и действий
    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    
    if len(state_shape) > 1:
        state_size = np.prod(state_shape)
        print(f"Состояние многомерное: {state_shape} -> преобразуем в {state_size}")
    else:
        state_size = state_shape[0]
    
    if hasattr(env.action_space, 'n'):
        action_size = env.action_space.n
        print(f"Дискретные действия: {action_size}")
    else:
        action_size = action_shape[0]
        print(f"Непрерывные действия: {action_size}")
    
    print(f"Размер состояния: {state_size}")
    print(f"Пространство действий: {env.action_space}")
    
    # Создание агентов
    print("\nСоздание агентов...")
    agents = {
        'DQN': DQNAgent(state_size, action_size, device),
        'PPO': PPOAgent(state_size, action_size, device),
        'SAC': SACAgent(state_size, action_size, device),
        'A2C': A2CAgent(state_size, action_size, device)
    }
    
    print("Все агенты созданы!")
    
    # Обучение всех алгоритмов
    print("\nНАЧАЛО ОБУЧЕНИЯ ВСЕХ АЛГОРИТМОВ")
    print("=" * 80)
    
    training_results = {}
    training_times = {}
    
    for algorithm_name, agent in agents.items():
        print(f"\nОбучение {algorithm_name}...")
        start_time = time.time()
        
        try:
            if algorithm_name == 'DQN':
                result = train_dqn_agent(env, agent, episodes=episodes, max_steps=max_steps)
            elif algorithm_name == 'PPO':
                result = train_ppo_agent(env, agent, episodes=episodes, max_steps=max_steps)
            elif algorithm_name == 'SAC':
                result = train_sac_agent(env, agent, episodes=episodes, max_steps=max_steps)
            elif algorithm_name == 'A2C':
                result = train_a2c_agent(env, agent, episodes=episodes, max_steps=max_steps)
            
            training_results[algorithm_name] = result
            training_time = time.time() - start_time
            training_times[algorithm_name] = training_time
            
            print(f"{algorithm_name} обучен за {training_time:.1f} секунд")
            
            # Сохранение модели
            if save_models:
                os.makedirs("models", exist_ok=True)
                model_path = f"models/{algorithm_name.lower()}_comparison.pth"
                agent.save(model_path)
                print(f"Модель {algorithm_name} сохранена в {model_path}")
            
        except Exception as e:
            print(f"Ошибка при обучении {algorithm_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Оценка всех алгоритмов
    print("\nНАЧАЛО ОЦЕНКИ ВСЕХ АЛГОРИТМОВ")
    print("=" * 80)
    
    evaluation_results = {}
    
    for algorithm_name, agent in agents.items():
        if algorithm_name in training_results:
            print(f"\nОценка {algorithm_name}...")
            try:
                result = evaluate_agent(env, agent, algorithm_name, episodes=20)
                evaluation_results[algorithm_name] = result
                print(f"{algorithm_name} оценен")
            except Exception as e:
                print(f"Ошибка при оценке {algorithm_name}: {e}")
                continue
    
    # Запись видео лучших эпизодов
    print("\nЗАПИСЬ ВИДЕО ЛУЧШИХ ЭПИЗОДОВ")
    print("=" * 80)
    
    video_paths = {}
    
    for algorithm_name, agent in agents.items():
        if algorithm_name in training_results:
            print(f"\nЗапись видео для {algorithm_name}...")
            try:
                video_path = record_episode_video(env, agent, algorithm_name, max_steps)
                video_paths[algorithm_name] = video_path
                print(f"Видео {algorithm_name} записано")
            except Exception as e:
                print(f"Ошибка при записи видео {algorithm_name}: {e}")
                continue
    
    # Анализ и сравнение результатов
    if training_results and evaluation_results:
        print("\nАНАЛИЗ И СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
        print("=" * 80)
        
        # Проверяем, что у нас есть достаточно данных для анализа
        successful_algorithms = [alg for alg in training_results.keys() 
                               if training_results[alg] is not None and 
                               'episode_rewards' in training_results[alg] and
                               len(training_results[alg]['episode_rewards']) > 0]
        
        if len(successful_algorithms) == 0:
            print("ОШИБКА: Ни один алгоритм не был успешно обучен!")
            return None
        
        print(f"Успешно обученные алгоритмы: {successful_algorithms}")
        
        # Создание DataFrame для сравнения
        try:
            comparison_df = create_comparison_dataframe(training_results, evaluation_results)
        except Exception as e:
            print(f"Ошибка при создании DataFrame: {e}")
            comparison_df = None
        
        # Построение графиков
        print("\nПостроение сравнительных графиков...")
        try:
            plot_comparison_results(training_results, evaluation_results)
        except Exception as e:
            print(f"Ошибка при построении графиков: {e}")
            import traceback
            traceback.print_exc()
        
        # Создание сводки производительности
        print("\nСоздание сводки производительности...")
        if comparison_df is not None:
            try:
                performance_summary = create_performance_summary(comparison_df)
            except Exception as e:
                print(f"Ошибка при создании сводки: {e}")
                performance_summary = None
        else:
            print("Не удалось создать сводку - нет данных для сравнения")
            performance_summary = None
        
        # Сохранение результатов
        print("\nСохранение результатов...")
        try:
            save_comparison_results(comparison_df, training_results, evaluation_results)
        except Exception as e:
            print(f"Ошибка при сохранении результатов: {e}")
        
        # Показ видео лучших эпизодов
        print("\nПОКАЗ ВИДЕО ЛУЧШИХ ЭПИЗОДОВ")
        print("=" * 80)
        
        for algorithm_name, video_path in video_paths.items():
            if video_path:
                display_video(video_path, algorithm_name)
        
        # Итоговая сводка
        print("\nИТОГОВАЯ СВОДКА")
        print("=" * 80)
        print("Время обучения:")
        for algorithm, train_time in training_times.items():
            print(f"  • {algorithm}: {train_time:.1f} секунд")
        
        print(f"\nОбщее время анализа: {sum(training_times.values()):.1f} секунд")
        print(f"Количество успешно обученных алгоритмов: {len(training_results)}")
        
        return {
            'comparison_df': comparison_df,
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'training_times': training_times,
            'agents': agents,
            'video_paths': video_paths
        }
    
    else:
        print("Не удалось получить результаты для анализа")
        return None


def run_quick_test(episodes=20, max_steps=500):
    """
    Быстрый тест всех алгоритмов на небольшом количестве эпизодов.
    """
    print("ЗАПУСК БЫСТРОГО ТЕСТА")
    print("=" * 50)
    print(f"Быстрый тест: {episodes} эпизодов, {max_steps} шагов")
    print("=" * 50)
    
    return run_complete_comparison(episodes=episodes, max_steps=max_steps, save_models=False)


def demonstrate_algorithm(algorithm_name, episodes=50, max_steps=1000):
    """
    Демонстрирует работу отдельного алгоритма с записью видео.
    
    Args:
        algorithm_name: название алгоритма ('DQN', 'PPO', 'SAC', 'A2C')
        episodes: количество эпизодов для обучения
        max_steps: максимальное количество шагов в эпизоде
    
    Returns:
        dict: результаты обучения и оценки
    """
    print(f"ДЕМОНСТРАЦИЯ АЛГОРИТМА {algorithm_name}")
    print("=" * 60)
    
    # Проверка доступности CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"PyTorch версия: {torch.__version__}")
    print(f"Используемое устройство: {device}")
    
    # Установка семян
    set_seed(42)
    
    # Создание среды
    print("\nСоздание среды highway-env...")
    env = create_highway_env()
    
    # Получаем размеры состояния и действий
    state_shape = env.observation_space.shape
    if len(state_shape) > 1:
        state_size = np.prod(state_shape)
        print(f"Состояние многомерное: {state_shape} -> преобразуем в {state_size}")
    else:
        state_size = state_shape[0]
        print(f"Состояние одномерное: {state_shape} -> размер {state_size}")
    
    if hasattr(env.action_space, 'n'):
        action_size = env.action_space.n
        print(f"Дискретные действия: {action_size}")
    else:
        action_size = env.action_space.shape[0]
        print(f"Непрерывные действия: {action_size}")
    
    print(f"Размер состояния: {state_size}")
    print(f"Пространство действий: {env.action_space}")
    
    # Проверяем тестовое состояние
    test_state, _ = env.reset()
    print(f"Тестовое состояние: shape={test_state.shape}, ожидаемый размер={state_size}")
    if hasattr(test_state, 'flatten'):
        flat_state = test_state.flatten()
        print(f"Плоское состояние: shape={flat_state.shape}")
    else:
        print(f"Состояние уже плоское: shape={test_state.shape}")
    
    # Создание агента
    print(f"\nСоздание {algorithm_name} агента...")
    print(f"Параметры агента: state_size={state_size}, action_size={action_size}")
    if algorithm_name == 'DQN':
        agent = DQNAgent(state_size, action_size, device)
        train_func = train_dqn_agent
    elif algorithm_name == 'PPO':
        agent = PPOAgent(state_size, action_size, device)
        train_func = train_ppo_agent
    elif algorithm_name == 'SAC':
        agent = SACAgent(state_size, action_size, device)
        train_func = train_sac_agent
    elif algorithm_name == 'A2C':
        agent = A2CAgent(state_size, action_size, device)
        train_func = train_a2c_agent
    else:
        print(f"Неизвестный алгоритм: {algorithm_name}")
        return None
    
    print(f"{algorithm_name} агент создан!")
    
    # Обучение
    print(f"\nНачало обучения {algorithm_name}...")
    start_time = time.time()
    
    try:
        training_result = train_func(env, agent, episodes=episodes, max_steps=max_steps)
        training_time = time.time() - start_time
        print(f"{algorithm_name} обучен за {training_time:.1f} секунд")
        
        # Оценка
        print(f"\nОценка {algorithm_name}...")
        evaluation_result = evaluate_agent(env, agent, algorithm_name, episodes=20)
        
        # Запись видео
        print(f"\nЗапись видео лучшего эпизода {algorithm_name}...")
        video_path = record_episode_video(env, agent, algorithm_name, max_steps)
        
        # Показ видео
        if video_path:
            display_video(video_path, algorithm_name)
        
        # Сохранение модели
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{algorithm_name.lower()}_demo.pth"
        agent.save(model_path)
        print(f"Модель сохранена в {model_path}")
        
        results = {
            'training_result': training_result,
            'evaluation_result': evaluation_result,
            'training_time': training_time,
            'video_path': video_path,
            'model_path': model_path,
            'agent': agent
        }
        
        print(f"\nДемонстрация {algorithm_name} завершена успешно!")
        return results
        
    except Exception as e:
        print(f"Ошибка при демонстрации {algorithm_name}: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        env.close()


if __name__ == "__main__":
    # Настройка стилей для графиков
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ АЛГОРИТМОВ RL ДЛЯ АВТОНОМНОГО ВОЖДЕНИЯ")
    print("=" * 80)
    print("Доступные функции:")
    print("  • run_complete_comparison() - полный анализ всех алгоритмов")
    print("  • run_quick_test() - быстрый тест")
    print("  • demonstrate_algorithm('DQN') - демонстрация DQN")
    print("  • demonstrate_algorithm('PPO') - демонстрация PPO")
    print("  • demonstrate_algorithm('SAC') - демонстрация SAC")
    print("  • demonstrate_algorithm('A2C') - демонстрация A2C")
    print("=" * 80)
    
    # Запуск полного анализа по умолчанию
    print("\nЗапуск полного анализа...")
    results = run_complete_comparison(episodes=100, max_steps=1000, save_models=True)
    
    if results:
        print("\nАнализ завершен успешно!")
    else:
        print("\nАнализ завершен с ошибками.")
