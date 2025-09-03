"""
Функции визуализации результатов обучения с подкреплением.

Этот модуль предоставляет комплексные инструменты для визуализации и анализа
результатов обучения различных алгоритмов RL в задаче автономного вождения.

Основные возможности:
- Сравнительные графики наград и производительности
- Анализ стабильности обучения через скользящие средние
- Распределение финальных результатов (box plots)
- Сравнение успешности алгоритмов
- Радарные диаграммы характеристик
- Сравнение результатов обучения и оценки

Визуализации помогают:
- Сравнить эффективность разных алгоритмов
- Выявить сильные и слабые стороны каждого подхода
- Понять динамику обучения и стабильность
- Принять обоснованное решение о выборе алгоритма
- Представить результаты в понятном виде

Используемые библиотеки:
- matplotlib: для создания графиков
- seaborn: для улучшенной стилизации
- numpy: для математических операций
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_comparison_results(training_results, evaluation_results):
    """
    Строит комплексные сравнительные графики для всех алгоритмов RL.
    
    Создает 6 различных визуализаций для глубокого анализа результатов:
    1. График наград за эпизоды - показывает динамику обучения
    2. Скользящее среднее наград - сглаживает шум и показывает тренды
    3. Box plot финальных наград - показывает распределение результатов
    4. Сравнение успешности - процент успешных эпизодов
    5. Сравнение обучения vs оценки - проверка обобщающей способности
    6. Радарная диаграмма - многомерное сравнение характеристик
    
    Args:
        training_results (dict): результаты обучения для каждого алгоритма
        evaluation_results (dict): результаты оценки для каждого алгоритма
        
    Returns:
        None: отображает графики через matplotlib
    """
    # Определение алгоритмов и цветов для визуализации
    algorithms = ['DQN', 'PPO', 'SAC', 'A2C']
    colors = ['blue', 'green', 'red', 'purple']
    
    # Создание сетки графиков 2x3
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Сравнительный анализ алгоритмов RL для автономного вождения', fontsize=16)
    
    # 1. График наград за эпизоды
    ax1 = axes[0, 0]
    for i, algorithm in enumerate(algorithms):
        if algorithm in training_results and training_results[algorithm] is not None:
            rewards = training_results[algorithm]['episode_rewards']
            if len(rewards) > 0:
                ax1.plot(rewards, alpha=0.7, color=colors[i % len(colors)], label=algorithm, linewidth=1)
    
    ax1.set_title('Награды за эпизоды')
    ax1.set_xlabel('Эпизод')
    ax1.set_ylabel('Награда')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Скользящее среднее наград
    ax2 = axes[0, 1]
    
    # Определяем размер окна на основе доступных данных
    available_lengths = [len(training_results[alg]['episode_rewards']) 
                         for alg in algorithms 
                         if alg in training_results and training_results[alg] is not None]
    
    if available_lengths:
        window_size = min(20, min(available_lengths))
    else:
        window_size = 0
    
    if window_size > 0:
        for i, algorithm in enumerate(algorithms):
            if algorithm in training_results and training_results[algorithm] is not None:
                rewards = training_results[algorithm]['episode_rewards']
                if len(rewards) >= window_size:
                    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                    ax2.plot(range(window_size-1, len(rewards)), moving_avg, 
                             color=colors[i % len(colors)], label=algorithm, linewidth=2)
        
        ax2.set_title(f'Скользящее среднее наград (окно {window_size})')
    else:
        ax2.set_title('Недостаточно данных для скользящего среднего')
    
    ax2.set_xlabel('Эпизод')
    ax2.set_ylabel('Награда')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Сравнение финальных наград
    ax3 = axes[0, 2]
    final_rewards = []
    labels = []
    
    for algorithm in algorithms:
        if algorithm in training_results and training_results[algorithm] is not None:
            rewards = training_results[algorithm]['episode_rewards']
            if len(rewards) > 0:
                # Берем последние эпизоды, но не больше чем есть
                last_episodes = min(20, len(rewards))
                final_rewards.append(rewards[-last_episodes:])
                labels.append(algorithm)
    
    if final_rewards and all(len(r) > 0 for r in final_rewards):
        try:
            ax3.boxplot(final_rewards, labels=labels)
            ax3.set_title('Распределение финальных наград')
            ax3.set_ylabel('Награда')
            ax3.grid(True, alpha=0.3)
        except Exception as e:
            ax3.text(0.5, 0.5, f'Ошибка построения: {e}', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Ошибка построения графика')
    else:
        ax3.text(0.5, 0.5, 'Недостаточно данных', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Недостаточно данных')
    
    # 4. Сравнение успешности
    ax4 = axes[1, 0]
    success_rates = []
    success_labels = []
    for algorithm in algorithms:
        if algorithm in training_results:
            if 'episode_successes' in training_results[algorithm]:
                success_rate = np.mean(training_results[algorithm]['episode_successes'][-20:])
                success_rates.append(success_rate)
                success_labels.append(algorithm)
    
    if success_rates:
        bars = ax4.bar(success_labels, success_rates, color=colors[:len(success_labels)], alpha=0.7)
        ax4.set_title('Успешность обучения (последние 20 эпизодов)')
        ax4.set_ylabel('Успешность')
        ax4.set_ylim(0, 1)
        
        # Добавляем значения на столбцы
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.2%}', ha='center', va='bottom')
    else:
        ax4.text(0.5, 0.5, 'Нет данных об успешности', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Нет данных об успешности')
    
    # 5. Сравнение обучения vs оценки
    ax5 = axes[1, 1]
    
    train_rewards = []
    eval_rewards = []
    comparison_labels = []
    for algorithm in algorithms:
        if algorithm in training_results and algorithm in evaluation_results:
            train_rewards.append(np.mean(training_results[algorithm]['episode_rewards'][-20:]))
            eval_rewards.append(evaluation_results[algorithm]['mean_reward'])
            comparison_labels.append(algorithm)
    
    if train_rewards and eval_rewards and len(train_rewards) == len(eval_rewards):
        x = np.arange(len(comparison_labels))
        width = 0.35
        
        ax5.bar(x - width/2, train_rewards, width, label='Обучение', alpha=0.7)
        ax5.bar(x + width/2, eval_rewards, width, label='Оценка', alpha=0.7)
        
        ax5.set_title('Сравнение наград: обучение vs оценка')
        ax5.set_ylabel('Награда')
        ax5.set_xticks(x)
        ax5.set_xticklabels(comparison_labels)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'Недостаточно данных для сравнения', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Недостаточно данных для сравнения')
    
    # 6. Радарная диаграмма характеристик
    ax6 = axes[1, 2]
    
    # Создаем данные для радарной диаграммы
    radar_data = []
    radar_labels = []
    
    for algorithm in algorithms:
        if algorithm in training_results and algorithm in evaluation_results:
            try:
                train_metrics = training_results[algorithm]
                eval_metrics = evaluation_results[algorithm]
                
                # Проверяем наличие необходимых данных
                if (len(train_metrics['episode_rewards']) > 0 and 
                    'episode_successes' in train_metrics and 
                    len(train_metrics['episode_successes']) > 0):
                    
                    # Нормализуем метрики (0-1)
                    reward_norm = min(1.0, np.mean(train_metrics['episode_rewards'][-20:]) / 20)
                    success_norm = np.mean(train_metrics['episode_successes'][-20:])
                    stability_norm = 1 / (1 + np.std(train_metrics['episode_rewards'][-20:]))
                    length_norm = min(1.0, np.mean(train_metrics['episode_lengths'][-20:]) / 1000)
                    generalization_norm = 1 / (1 + abs(np.mean(train_metrics['episode_rewards'][-20:]) - eval_metrics['mean_reward']))
                    
                    radar_data.append([reward_norm, success_norm, stability_norm, length_norm, generalization_norm])
                    radar_labels.append(algorithm)
            except Exception as e:
                print(f"Ошибка при обработке данных для {algorithm}: {e}")
                continue
    
    if radar_data and len(radar_data) > 0:
        try:
            categories = ['Награда', 'Успешность', 'Стабильность', 'Длина', 'Обобщение']
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]  # Замыкаем круг
            
            for i, data in enumerate(radar_data):
                data += data[:1]  # Замыкаем круг
                ax6.plot(angles, data, 'o-', linewidth=2, label=radar_labels[i], color=colors[i % len(colors)])
                ax6.fill(angles, data, alpha=0.1, color=colors[i % len(colors)])
            
            ax6.set_xticks(angles[:-1])
            ax6.set_xticklabels(categories)
            ax6.set_ylim(0, 1)
            ax6.set_title('Радарная диаграмма характеристик')
            ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        except Exception as e:
            ax6.text(0.5, 0.5, f'Ошибка построения: {e}', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Ошибка построения радарной диаграммы')
    else:
        ax6.text(0.5, 0.5, 'Недостаточно данных для радарной диаграммы', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Недостаточно данных для радарной диаграммы')
    
    plt.tight_layout()
    plt.show()
