"""
Модуль агентов для обучения с подкреплением.

Содержит реализации четырех алгоритмов RL:
- DQN (Deep Q-Network)
- PPO (Proximal Policy Optimization)  
- SAC (Soft Actor-Critic)
- A2C (Advantage Actor-Critic)
"""

from .dqn_agent import DQNAgent
from .ppo_agent import PPOAgent
from .sac_agent import SACAgent
from .a2c_agent import A2CAgent

__all__ = ['DQNAgent', 'PPOAgent', 'SACAgent', 'A2CAgent']
