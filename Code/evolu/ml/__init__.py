# -*- coding: utf-8 -*-
"""Machine learning module for adaptive algorithm control.

This module provides reinforcement learning and neural network components
for adaptive operator selection and parameter tuning in evolutionary algorithms.

Classes:
    QLearning: Tabular Q-learning for discrete action spaces
    DQN: Deep Q-Network for high-dimensional state spaces
"""

from .q_learning import QLearning
from .dqn import DQN

__all__ = ['QLearning', 'DQN']
