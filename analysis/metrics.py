"""
Функции расчета метрик для RL алгоритмов.

Содержит функции для расчета различных метрик производительности.
"""

import numpy as np


def calculate_metrics(episode_rewards, episode_lengths, episode_successes):
    """
    Рассчитывает основные метрики для алгоритма RL.
    
    Args:
        episode_rewards: список наград за эпизоды
        episode_lengths: список длин эпизодов
        episode_successes: список успешности эпизодов
    
    Returns:
        dict: словарь с метриками
    """
    if not episode_rewards or len(episode_rewards) == 0:
        return None
    
    # Базовые метрики
    total_episodes = len(episode_rewards)
    total_reward = np.sum(episode_rewards)
    total_length = np.sum(episode_lengths) if episode_lengths else 0
    
    # Средние значения
    mean_reward = np.mean(episode_rewards)
    mean_length = np.mean(episode_lengths) if episode_lengths else 0
    mean_success = np.mean(episode_successes) if episode_successes else 0
    
    # Стандартные отклонения
    std_reward = np.std(episode_rewards)
    std_length = np.std(episode_lengths) if episode_lengths else 0
    
    # Минимальные и максимальные значения
    min_reward = np.min(episode_rewards)
    max_reward = np.max(episode_rewards)
    min_length = np.min(episode_lengths) if episode_lengths else 0
    max_length = np.max(episode_lengths) if episode_lengths else 0
    
    # Метрики обучения
    if len(episode_rewards) >= 20:
        # Последние 20 эпизодов
        recent_rewards = episode_rewards[-20:]
        recent_success = episode_successes[-20:] if episode_successes else []
        
        recent_mean_reward = np.mean(recent_rewards)
        recent_mean_success = np.mean(recent_success) if recent_success else 0
        
        # Скорость улучшения (разница между последними и первыми 20 эпизодами)
        if len(episode_rewards) >= 40:
            first_20_rewards = episode_rewards[:20]
            first_20_mean = np.mean(first_20_rewards)
            learning_improvement = recent_mean_reward - first_20_mean
        else:
            learning_improvement = 0
    else:
        recent_mean_reward = mean_reward
        recent_mean_success = mean_success
        learning_improvement = 0
    
    # Стабильность (коэффициент вариации)
    stability = std_reward / (abs(mean_reward) + 1e-8)
    
    # Эффективность (награда на шаг)
    efficiency = mean_reward / (mean_length + 1e-8)
    
    metrics = {
        'total_episodes': total_episodes,
        'total_reward': total_reward,
        'total_length': total_length,
        
        'mean_reward': mean_reward,
        'mean_length': mean_length,
        'mean_success': mean_success,
        
        'std_reward': std_reward,
        'std_length': std_length,
        
        'min_reward': min_reward,
        'max_reward': max_reward,
        'min_length': min_length,
        'max_length': max_length,
        
        'recent_mean_reward': recent_mean_reward,
        'recent_mean_success': recent_mean_success,
        'learning_improvement': learning_improvement,
        
        'stability': stability,
        'efficiency': efficiency
    }
    
    return metrics


def calculate_comparison_metrics(algorithm_metrics):
    """
    Рассчитывает метрики для сравнения алгоритмов.
    
    Args:
        algorithm_metrics: словарь с метриками для каждого алгоритма
    
    Returns:
        dict: словарь с метриками сравнения
    """
    comparison_metrics = {}
    
    for algorithm, metrics in algorithm_metrics.items():
        if metrics is None:
            continue
            
        # Нормализованные метрики (0-1)
        reward_norm = min(1.0, metrics['mean_reward'] / 20)  # Нормализуем по максимальной награде
        success_norm = metrics['mean_success']
        stability_norm = 1 / (1 + metrics['stability'])  # Инвертируем стабильность
        efficiency_norm = min(1.0, metrics['efficiency'] / 0.1)  # Нормализуем эффективность
        
        comparison_metrics[algorithm] = {
            'reward_score': reward_norm,
            'success_score': success_norm,
            'stability_score': stability_norm,
            'efficiency_score': efficiency_norm,
            'overall_score': np.mean([reward_norm, success_norm, stability_norm, efficiency_norm])
        }
    
    return comparison_metrics


def print_metrics_summary(algorithm_metrics):
    """
    Выводит сводку метрик для всех алгоритмов.
    
    Args:
        algorithm_metrics: словарь с метриками для каждого алгоритма
    """
    print("\n" + "="*80)
    print("СВОДКА МЕТРИК ПО АЛГОРИТМАМ")
    print("="*80)
    
    for algorithm, metrics in algorithm_metrics.items():
        if metrics is None:
            print(f"\n{algorithm}: Нет данных")
            continue
            
        print(f"\n{algorithm}:")
        print(f"  Эпизоды: {metrics['total_episodes']}")
        print(f"  Средняя награда: {metrics['mean_reward']:.3f} ± {metrics['std_reward']:.3f}")
        print(f"  Средняя длина: {metrics['mean_length']:.1f} ± {metrics['std_length']:.1f}")
        print(f"  Успешность: {metrics['mean_success']:.2%}")
        print(f"  Стабильность: {metrics['stability']:.3f}")
        print(f"  Эффективность: {metrics['efficiency']:.3f}")
        
        if metrics['learning_improvement'] != 0:
            print(f"  Улучшение: {metrics['learning_improvement']:+.3f}")
        
        print(f"  Последние 20 эпизодов:")
        print(f"    Награда: {metrics['recent_mean_reward']:.3f}")
        print(f"    Успешность: {metrics['recent_mean_success']:.2%}")


def calculate_ranking(comparison_metrics):
    """
    Рассчитывает рейтинг алгоритмов по различным критериям.
    
    Args:
        comparison_metrics: словарь с метриками сравнения
    
    Returns:
        dict: словарь с рейтингами
    """
    if not comparison_metrics:
        return {}
    
    algorithms = list(comparison_metrics.keys())
    
    # Сортируем по каждому критерию
    rankings = {}
    
    # По общей оценке
    overall_scores = [(alg, comparison_metrics[alg]['overall_score']) for alg in algorithms]
    overall_scores.sort(key=lambda x: x[1], reverse=True)
    rankings['overall'] = overall_scores
    
    # По награде
    reward_scores = [(alg, comparison_metrics[alg]['reward_score']) for alg in algorithms]
    reward_scores.sort(key=lambda x: x[1], reverse=True)
    rankings['reward'] = reward_scores
    
    # По успешности
    success_scores = [(alg, comparison_metrics[alg]['success_score']) for alg in algorithms]
    success_scores.sort(key=lambda x: x[1], reverse=True)
    rankings['success'] = success_scores
    
    # По стабильности
    stability_scores = [(alg, comparison_metrics[alg]['stability_score']) for alg in algorithms]
    stability_scores.sort(key=lambda x: x[1], reverse=True)
    rankings['stability'] = stability_scores
    
    # По эффективности
    efficiency_scores = [(alg, comparison_metrics[alg]['efficiency_score']) for alg in algorithms]
    efficiency_scores.sort(key=lambda x: x[1], reverse=True)
    rankings['efficiency'] = efficiency_scores
    
    return rankings


def print_rankings(rankings):
    """
    Выводит рейтинги алгоритмов.
    
    Args:
        rankings: словарь с рейтингами
    """
    print("\n" + "="*80)
    print("РЕЙТИНГИ АЛГОРИТМОВ")
    print("="*80)
    
    for criterion, ranking in rankings.items():
        print(f"\n{criterion.upper()}:")
        for i, (algorithm, score) in enumerate(ranking):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
            print(f"  {medal} {algorithm}: {score:.3f}")
