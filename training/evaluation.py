"""
Функции оценки обученных RL агентов.

Содержит функции для оценки производительности агентов.
"""

import numpy as np
from tqdm import tqdm


def evaluate_agent(env, agent, algorithm_name, episodes=20):
    """Оценивает обученного агента."""
    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    
    print(f"Оценка {algorithm_name} агента на {episodes} эпизодах...")
    
    for episode in tqdm(range(episodes), desc=f"{algorithm_name} Оценка"):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            if algorithm_name == "DQN" or algorithm_name == "SAC":
                action = agent.act(state, training=False)
            else:
                # PPO и A2C возвращают кортеж из трех значений
                action_result = agent.act(state, training=False)
                if isinstance(action_result, tuple) and len(action_result) >= 1:
                    action = action_result[0]
                else:
                    action = action_result
            
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_successes.append(1 if not terminated else 0)
    
    metrics = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "success_rate": np.mean(episode_successes),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_successes": episode_successes
    }
    
    print(f"{algorithm_name} результаты:")
    print(f"  Средняя награда: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"  Средняя длина: {metrics['mean_length']:.1f} ± {metrics['std_length']:.1f}")
    print(f"  Успешность: {metrics['success_rate']:.2%}")
    
    return metrics
