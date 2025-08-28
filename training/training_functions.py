"""
Функции обучения для всех алгоритмов RL.

Содержит функции обучения для:
- DQN
- PPO
- SAC
- A2C
"""

import numpy as np
import torch
from tqdm import tqdm


def train_dqn_agent(env, agent, episodes=100, max_steps=1000, update_every=4):
    """Обучает DQN агента."""
    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    
    print(f"Обучение DQN агента на {episodes} эпизодах...")
    
    for episode in tqdm(range(episodes), desc="DQN Обучение"):
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
                agent.replay()
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_successes.append(1 if not terminated else 0)
        agent.episodes += 1
        
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"DQN Эпизод {episode + 1}: средняя награда = {avg_reward:.2f}")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_successes': episode_successes
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
