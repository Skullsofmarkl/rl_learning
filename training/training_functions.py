"""
Функции обучения для всех алгоритмов обучения с подкреплением.

Этот модуль содержит унифицированные функции обучения для четырех основных
алгоритмов RL, используемых в проекте автономного вождения:

- DQN (Deep Q-Network): off-policy алгоритм с Experience Replay
- PPO (Proximal Policy Optimization): on-policy алгоритм с clipped objective
- SAC (Soft Actor-Critic): off-policy алгоритм с максимальной энтропией
- A2C (Advantage Actor-Critic): on-policy алгоритм с advantage estimation

Каждая функция обучения:
1. Управляет взаимодействием агента со средой
2. Собирает опыт и сохраняет его в буфер
3. Периодически обновляет нейронные сети агента
4. Отслеживает метрики обучения (награды, длины эпизодов, успешность)
5. Выводит прогресс обучения

Особенности реализации:
- Единообразный интерфейс для всех алгоритмов
- Автоматическое отслеживание прогресса с tqdm
- Гибкие параметры обучения (количество эпизодов, шагов, частота обновлений)
- Сохранение детальной статистики для анализа
"""

import numpy as np
import torch
from tqdm import tqdm


def train_dqn_agent(env, agent, episodes=100, max_steps=1000, update_every=4):
    """
    Обучает DQN (Deep Q-Network) агента.
    
    DQN алгоритм обучения:
    1. Агент взаимодействует со средой, выбирая действия через epsilon-greedy
    2. Переходы (state, action, reward, next_state, done) сохраняются в Experience Replay Buffer
    3. Каждые update_every шагов агент обучается на случайном батче из буфера
    4. Target Network периодически обновляется для стабильности
    5. Epsilon постепенно уменьшается для перехода от exploration к exploitation
    
    Args:
        env: среда для обучения (highway-env)
        agent: DQN агент для обучения
        episodes (int): количество эпизодов обучения
        max_steps (int): максимальное количество шагов в эпизоде
        update_every (int): частота обновления сети (каждые N шагов)
        
    Returns:
        dict: словарь с метриками обучения (награды, длины эпизодов, успешность)
    """
    # Инициализация списков для отслеживания метрик
    episode_rewards = []      # Награды за каждый эпизод
    episode_lengths = []      # Длины эпизодов (количество шагов)
    episode_successes = []    # Успешность эпизодов (1 = успех, 0 = неудача)
    
    print(f"Обучение DQN агента на {episodes} эпизодах...")
    
    # Основной цикл обучения
    for episode in tqdm(range(episodes), desc="DQN Обучение"):
        # Сброс среды и инициализация эпизода
        state, _ = env.reset()
        total_reward = 0      # Общая награда за эпизод
        steps = 0             # Количество шагов в эпизоде
        
        # Цикл взаимодействия со средой
        for step in range(max_steps):
            # Выбор действия агентом (epsilon-greedy)
            action = agent.act(state, training=True)
            
            # Выполнение действия в среде
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Сохранение перехода в Experience Replay Buffer
            agent.remember(state, action, reward, next_state, done)
            
            # Обновление счетчиков
            total_reward += reward
            steps += 1
            agent.steps += 1
            state = next_state
            
            # Обновление сети каждые update_every шагов
            if agent.steps % update_every == 0:
                agent.replay()  # Обучение на случайном батче из буфера
            
            # Завершение эпизода при достижении терминального состояния
            if done:
                break
        
        # Сохранение метрик эпизода
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_successes.append(1 if not terminated else 0)  # 1 = успех, 0 = авария
        agent.episodes += 1
        
        # Вывод прогресса каждые 20 эпизодов
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])  # Средняя награда за последние 20 эпизодов
            print(f"DQN Эпизод {episode + 1}: средняя награда = {avg_reward:.2f}")
    
    # Возврат метрик обучения
    return {
        'episode_rewards': episode_rewards,      # Список наград за все эпизоды
        'episode_lengths': episode_lengths,      # Список длин всех эпизодов
        'episode_successes': episode_successes   # Список успешности всех эпизодов
    }


def train_ppo_agent(env, agent, episodes=100, max_steps=1000, update_every=20):
    """Обучает PPO агента."""
    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    
    print(f"Обучение PPO агента на {episodes} эпизодах...")
    
    for episode in tqdm(range(episodes), desc="PPO Обучение"):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        episode_states = []
        episode_actions = []
        episode_rewards_ep = []
        episode_action_probs = []
        episode_values = []
        episode_dones = []
        
        for step in range(max_steps):
            action, action_probs, value = agent.act(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards_ep.append(reward)
            episode_action_probs.append(action_probs)
            episode_values.append(value)
            episode_dones.append(done)
            
            total_reward += reward
            steps += 1
            agent.steps += 1
            state = next_state
            
            if done:
                break
        
        # Вычисляем GAE и обновляем сети
        if len(episode_states) > 0:
            episode_states = np.array(episode_states)
            episode_actions = np.array(episode_actions)
            episode_rewards_ep = np.array(episode_rewards_ep)
            episode_action_probs = np.array(episode_action_probs)
            episode_values = np.array(episode_values)
            episode_dones = np.array(episode_dones)
            
            # Получаем значение следующего состояния
            next_state_tensor = torch.FloatTensor(next_state.flatten()).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                next_value = agent.critic(next_state_tensor).item()
            
            advantages, returns = agent.compute_gae(
                episode_rewards_ep, episode_values, episode_dones, next_value
            )
            
            # Обновляем сети
            agent.update(episode_states, episode_actions, episode_action_probs, advantages, returns)
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_successes.append(1 if not terminated else 0)
        agent.episodes += 1
        
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"PPO Эпизод {episode + 1}: средняя награда = {avg_reward:.2f}")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_successes': episode_successes
    }


def train_sac_agent(env, agent, episodes=100, max_steps=1000, update_every=4):
    """Обучает SAC агента."""
    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    
    print(f"Обучение SAC агента на {episodes} эпизодах...")
    
    for episode in tqdm(range(episodes), desc="SAC Обучение"):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            action = agent.act(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.remember(state, action, reward, next_state, done)
            
            total_reward += reward
            steps += 1
            agent.steps += 1
            state = next_state
            
            if agent.steps % update_every == 0:
                agent.update(batch_size=64)
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_successes.append(1 if not terminated else 0)
        agent.episodes += 1
        
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"SAC Эпизод {episode + 1}: средняя награда = {avg_reward:.2f}")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_successes': episode_successes
    }


def train_a2c_agent(env, agent, episodes=100, max_steps=1000, update_every=20):
    """Обучает A2C агента."""
    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    
    print(f"Обучение A2C агента на {episodes} эпизодах...")
    
    for episode in tqdm(range(episodes), desc="A2C Обучение"):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            action, action_probs, value = agent.act(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.remember(state, action, reward, next_state, done, action_probs, value)
            
            total_reward += reward
            steps += 1
            agent.steps += 1
            state = next_state
            
            if done:
                break
        
        # Обновляем сети каждые N эпизодов
        if (episode + 1) % update_every == 0:
            agent.update(batch_size=64)
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_successes.append(1 if not terminated else 0)
        agent.episodes += 1
        
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"A2C Эпизод {episode + 1}: средняя награда = {avg_reward:.2f}")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_successes': episode_successes
    }
