# -*- coding: utf-8 -*-
"""Machine learning utilities for adaptive operator selection.
 
This module provides machine learning components that can be integrated into
evolutionary algorithms for adaptive operator selection and parameter tuning.
Currently includes Q-Learning for reinforcement learning-based adaptation.
"""

from typing import List, Any
import numpy as np
import pandas as pd


class QLearning:
    """Q-Learning reinforcement learning agent for adaptive operator selection.
    
    Q-Learning is a model-free reinforcement learning algorithm that learns
    an optimal action-selection policy through trial and error. In the context
    of evolutionary algorithms, it can be used to adaptively select operators
    (e.g., mutation, crossover) based on their historical performance.
    
    The agent maintains a Q-table that stores Q-values (quality values) for
    state-action pairs. Actions are selected using an epsilon-greedy policy,
    and the Q-table is updated using the Bellman equation.
    
    Attributes:
        epsilon_greedy (float): Epsilon value for epsilon-greedy policy (exploitation rate).
            Higher values favor exploitation over exploration. Defaults to 0.95.
        learning_rate (float): Learning rate (alpha) controlling how much new
            information overrides old Q-values. Defaults to 0.8.
        reward_decay (float): Discount factor (gamma) for future rewards.
            Higher values give more weight to long-term rewards. Defaults to 0.8.
        action_set (List[Any]): List of available actions (e.g., operator types).
        state_set (List[Any]): List of possible states.
        q_table (pd.DataFrame): Q-table storing Q-values for state-action pairs.
            Shape: (number_of_states, number_of_actions).
    
    Example:
        >>> # Define states (e.g., algorithm phases) and actions (e.g., operators)
        >>> states = ["initial", "exploration", "exploitation"]
        >>> actions = ["mutation_op1", "mutation_op2", "crossover_op1"]
        >>> q_agent = QLearning(action_set=actions, state_set=states)
        >>> 
        >>> # Select action for current state
        >>> current_state = 0  # "initial"
        >>> action_idx = q_agent.select_action_and_return_index(current_state)
        >>> 
        >>> # Update Q-table based on reward
        >>> next_state = 1
        >>> reward = 0.5
        >>> q_agent.update_q_table(action_idx, current_state, next_state, reward)
    """
    
    def __init__(self, action_set: List[Any], state_set: List[Any]) -> None:
        """Initialize Q-Learning agent.
        
        Args:
            action_set (List[Any]): List of available actions (e.g., operator types,
                parameter values). Can be any hashable types.
            state_set (List[Any]): List of possible states (e.g., algorithm phases,
                population characteristics). Can be any hashable types.
        
        Note:
            The Q-table is initialized with zeros, meaning all state-action pairs
            start with equal Q-values (tabula rasa learning).
        """
        self.epsilon_greedy: float = 0.95
        self.learning_rate: float = 0.8
        self.reward_decay: float = 0.8
        self.action_set: List[Any] = action_set
        self.state_set: List[Any] = state_set
        self.q_table: pd.DataFrame = pd.DataFrame(
            np.zeros((len(self.state_set), len(self.action_set))),
        )

    def select_action_and_return_index(self, state_index: int) -> int:
        """Select action using epsilon-greedy policy.
        
        With probability epsilon_greedy, selects the action with highest Q-value
        (exploitation). With probability (1 - epsilon_greedy), selects a random action
        (exploration). If multiple actions have the same maximum Q-value, one
        is chosen randomly.
        
        Args:
            state_index (int): Index of the current state in state_set.
        
        Returns:
            int: Index of the selected action in action_set.
        
        Note:
            This is the epsilon-greedy exploration strategy commonly used in
            reinforcement learning to balance exploration and exploitation.
        """
        if np.random.uniform() < self.epsilon_greedy:
            q_values = self.q_table.iloc[state_index, :]
            eligible_actions: List[int] = []
            for i in range(len(self.action_set)):
                if q_values[i] == np.max(q_values):
                    eligible_actions.append(i)
            action_index = np.random.choice(eligible_actions)
        else:
            action_index = np.random.choice(len(self.action_set))
        return action_index

    def update_q_table(self, selected_action_index: int, current_state_index: int, 
                       next_state_index: int, reward_value: float) -> None:
        """Update Q-table using Q-learning formula.
        
        Updates the Q-value for the (state, action) pair using the Bellman equation:
            Q(s,a) = Q(s,a) + α * [R + γ * max(Q(s',a')) - Q(s,a)]
        
        where:
        - α (alpha) = learning_rate
        - R = reward_value
        - γ (gamma) = reward_decay
        
        Args:
            selected_action_index (int): Index of the action that was taken.
            current_state_index (int): Index of the state where action was taken.
            next_state_index (int): Index of the state reached after taking action.
            reward_value (float): Reward received for taking the action.
        
        Note:
            This implements the standard Q-learning update rule for off-policy
            temporal difference learning.
        """
        q_predict = self.q_table.iloc[current_state_index, selected_action_index]
        q_target = reward_value + self.reward_decay * self.q_table.iloc[next_state_index, :].max()
        self.q_table.iloc[current_state_index, selected_action_index] += self.learning_rate * (q_target - q_predict)