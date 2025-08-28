"""
Функции сравнительного анализа алгоритмов RL.

Содержит функции для:
- Создания DataFrame сравнения
- Анализа производительности
- Сохранения результатов
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime


def create_comparison_dataframe(training_results, evaluation_results):
    """Создает DataFrame для сравнения всех алгоритмов."""
    
    comparison_data = []
    
    for algorithm in ['DQN', 'PPO', 'SAC', 'A2C']:
        try:
            if (algorithm in training_results and algorithm in evaluation_results and
                training_results[algorithm] is not None and evaluation_results[algorithm] is not None):
                
                train_metrics = training_results[algorithm]
                eval_metrics = evaluation_results[algorithm]
                
                # Проверяем наличие необходимых данных
                if ('episode_rewards' not in train_metrics or 
                    'episode_lengths' not in train_metrics or
                    'episode_successes' not in train_metrics):
                    print(f"Пропускаем {algorithm}: недостаточно данных для анализа")
                    continue
                
                # Безопасно получаем последние эпизоды
                rewards = train_metrics['episode_rewards']
                lengths = train_metrics['episode_lengths']
                successes = train_metrics['episode_successes']
                
                if len(rewards) == 0 or len(lengths) == 0 or len(successes) == 0:
                    print(f"Пропускаем {algorithm}: пустые данные")
                    continue
                
                # Берем последние эпизоды, но не больше чем есть
                last_episodes = min(20, len(rewards))
                final_rewards = rewards[-last_episodes:]
                final_lengths = lengths[-last_episodes:]
                final_successes = successes[-last_episodes:]
                
                # Скорость обучения (награда в последних эпизодах)
                learning_speed = np.mean(final_rewards)
                
                # Стабильность (стандартное отклонение наград)
                stability = np.std(final_rewards)
                
                # Успешность в обучении
                training_success_rate = np.mean(final_successes)
                
                # Успешность в оценке (проверяем наличие)
                eval_success_rate = eval_metrics.get('success_rate', 0.0)
                
                # Обобщающая способность (разница между обучением и оценкой)
                generalization = learning_speed - eval_metrics.get('mean_reward', 0.0)
                
                comparison_data.append({
                    'Алгоритм': algorithm,
                    'Средняя награда (обучение)': np.mean(rewards),
                    'Средняя награда (оценка)': eval_metrics.get('mean_reward', 0.0),
                    'Скорость обучения': learning_speed,
                    'Стабильность': stability,
                    'Успешность (обучение)': training_success_rate,
                    'Успешность (оценка)': eval_success_rate,
                    'Средняя длина (обучение)': np.mean(lengths),
                    'Средняя длина (оценка)': eval_metrics.get('mean_length', 0.0),
                    'Обобщающая способность': generalization
                })
                
        except Exception as e:
            print(f"Ошибка при обработке данных для {algorithm}: {e}")
            continue
    
    if not comparison_data:
        print("ОШИБКА: Не удалось создать данные для сравнения!")
        return None
    
    return pd.DataFrame(comparison_data)


def create_performance_summary(comparison_df):
    """Создает сводку производительности всех алгоритмов."""
    
    print("=" * 80)
    print("СВОДКА ПРОИЗВОДИТЕЛЬНОСТИ АЛГОРИТМОВ RL")
    print("=" * 80)
    
    # Сортируем по средней награде в оценке
    sorted_df = comparison_df.sort_values('Средняя награда (оценка)', ascending=False)
    
    print("\nРЕЙТИНГ АЛГОРИТМОВ (по производительности в оценке):")
    for i, (_, row) in enumerate(sorted_df.iterrows()):
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "4️⃣"
        print(f"{medal} {row['Алгоритм']}: {row['Средняя награда (оценка)']:.2f}")
    
    print("\nДЕТАЛЬНЫЕ МЕТРИКИ:")
    print(sorted_df.to_string(index=False, float_format='%.3f'))
    
    # Анализ сильных и слабых сторон
    print("\nАНАЛИЗ СИЛЬНЫХ И СЛАБЫХ СТОРОН:")
    
    best_algorithm = sorted_df.iloc[0]['Алгоритм']
    print(f"\nЛучший алгоритм: {best_algorithm}")
    
    # Находим лучшие показатели по каждой метрике
    metrics_analysis = {
        'Средняя награда (оценка)': 'производительность',
        'Успешность (оценка)': 'надежность',
        'Стабильность': 'стабильность обучения',
        'Обобщающая способность': 'способность к обобщению'
    }
    
    for metric, description in metrics_analysis.items():
        best_idx = comparison_df[metric].idxmax()
        best_value = comparison_df.loc[best_idx, metric]
        best_alg = comparison_df.loc[best_idx, 'Алгоритм']
        print(f"  • Лучший по {description}: {best_alg} ({metric}: {best_value:.3f})")
    
    # Рекомендации
    print("\nРЕКОМЕНДАЦИИ ПО ВЫБОРУ АЛГОРИТМА:")
    
    # Определяем список алгоритмов
    algorithms = ['DQN', 'PPO', 'SAC', 'A2C']
    
    recommendations = {
        'DQN': 'Выбирайте для простых задач с дискретными действиями, когда важна простота реализации',
        'PPO': 'Выбирайте для задач, где важна стабильность обучения и максимальная производительность',
        'SAC': 'Выбирайте для сложных задач, где важна скорость обучения и исследование',
        'A2C': 'Выбирайте для быстрого прототипирования и понимания основ Actor-Critic методов'
    }
    
    for algorithm in algorithms:
        if algorithm in comparison_df['Алгоритм'].values:
            print(f"  • {algorithm}: {recommendations[algorithm]}")
    
    return sorted_df


def save_comparison_results(comparison_df, training_results, evaluation_results):
    """Сохраняет результаты сравнения в файлы."""
    
    # Создаем папку для результатов
    os.makedirs("comparison_results", exist_ok=True)
    
    # Сохраняем DataFrame
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_df.to_csv(f"comparison_results/comparison_{timestamp}.csv", index=False)
    
    # Сохраняем детальные результаты
    results_summary = {
        'timestamp': timestamp,
        'comparison_data': comparison_df.to_dict('records'),
        'training_results': {k: {key: value.tolist() if isinstance(value, np.ndarray) else value 
                                for key, value in v.items()} 
                            for k, v in training_results.items()},
        'evaluation_results': {k: {key: value.tolist() if isinstance(value, np.ndarray) else value 
                                  for key, value in v.items()} 
                              for k, v in evaluation_results.items()}
    }
    
    import json
    with open(f"comparison_results/detailed_results_{timestamp}.json", 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f"\nРезультаты сохранены в папку comparison_results/")
    print(f"  • comparison_{timestamp}.csv - сводная таблица")
    print(f"  • detailed_results_{timestamp}.json - детальные результаты")
