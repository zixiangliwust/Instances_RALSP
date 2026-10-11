# -*- coding: utf-8 -*-
"""Deep Q-Network (DQN) implementation for adaptive operator selection.

This module provides a Deep Q-Network reinforcement learning agent that can be
integrated into evolutionary algorithms for adaptive mechanism selection and
parameter tuning. DQN uses neural networks to approximate Q-values, enabling
learning in high-dimensional state spaces.
"""

from typing import List, Tuple, Optional
import numpy as np
import random
from collections import deque


class DQN:
    """Deep Q-Network reinforcement learning agent for adaptive operator selection.
    
    DQN uses a neural network to approximate Q-values for state-action pairs,
    overcoming the limitations of tabular Q-learning in high-dimensional spaces.
    It includes experience replay and target network features to stabilize learning.
    
    Attributes:
        state_dim (int): Dimension of the state space.
        action_dim (int): Number of possible actions.
        learning_rate (float): Learning rate for gradient descent.
        reward_decay (float): Discount factor for future rewards.
        epsilon_greedy (float): Epsilon-greedy value for exploitation (higher = more exploitation).
        epsilon_greedy_min (float): Minimum epsilon-greedy value (for decay).
        epsilon_greedy_decay (float): Decay rate for epsilon-greedy.
        replay_buffer_size (int): Maximum size of replay buffer.
        batch_size (int): Mini-batch size for training.
        memory (deque): Experience replay buffer.
        q_network (np.ndarray): Q-network weights (simplified as linear model).
        target_network (np.ndarray): Target network weights.
    
    Example:
        >>> # Define state and action dimensions
        >>> state_dim = 5  # e.g., [iteration_ratio, diversity, convergence, etc.]
        >>> action_dim = 3  # e.g., [operator1, operator2, operator3]
        >>> dqn = DQN(state_dim, action_dim)
        >>> 
        >>> # Select action
        >>> state = [0.3, 0.5, 0.2, 0.1, 0.8]
        >>> action = dqn.select_action(state)
        >>> 
        >>> # Store experience and train
        >>> next_state = [0.31, 0.48, 0.25, 0.12, 0.78]
        >>> reward = 0.5
        >>> dqn.remember(state, action, reward, next_state, done=False)
        >>> dqn.replay()
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 0.001,
                 reward_decay: float = 0.95,
                 epsilon_greedy: float = 0.9,
                 epsilon_greedy_min: float = 0.01,
                 epsilon_greedy_decay: float = 0.995,
                 replay_buffer_size: int = 10000,
                 batch_size: int = 32) -> None:
        """Initialize DQN agent.
        
        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Number of possible actions.
            learning_rate (float): Learning rate for network updates. Defaults to 0.001.
            reward_decay (float): Discount factor for future rewards. Defaults to 0.95.
            epsilon_greedy (float): Initial epsilon-greedy for exploitation rate. Defaults to 0.9.
            epsilon_greedy_min (float): Minimum epsilon-greedy value. Defaults to 0.01.
            epsilon_greedy_decay (float): Decay rate for epsilon-greedy. Defaults to 0.995.
            replay_buffer_size (int): Maximum size of replay buffer. Defaults to 10000.
            batch_size (int): Mini-batch size for training. Defaults to 32.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.epsilon_greedy = epsilon_greedy
        self.epsilon_greedy_min = epsilon_greedy_min
        self.epsilon_greedy_decay = epsilon_greedy_decay
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        
        # Experience replay buffer
        self.memory = deque(maxlen=self.replay_buffer_size)
        
        # Initialize networks (simplified linear approximation)
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self._update_target_network()
    
    def _build_network(self) -> np.ndarray:
        """Build a simple linear network for Q-value approximation.
        
        In a full implementation, this would create a neural network.
        Here we use a simplified linear model: Q(s,a) = W * s + b
        
        Returns:
            np.ndarray: Network weights matrix of shape (state_dim + 1, action_dim).
        """
        # Simple linear model: [state_dim + 1 (bias), action_dim]
        return np.random.randn(self.state_dim + 1, self.action_dim) * 0.01
    
    def _update_target_network(self) -> None:
        """Copy weights from Q-network to target network."""
        self.target_network = self.q_network.copy()
    
    def _predict(self, state: List[float], network: np.ndarray) -> np.ndarray:
        """Predict Q-values for all actions given a state.
        
        Args:
            state (List[float]): Current state.
            network (np.ndarray): Network to use for prediction.
        
        Returns:
            np.ndarray: Q-values for all actions, shape (action_dim,).
        """
        # Add bias term
        state_with_bias = np.append(state, 1.0)
        q_values = np.dot(state_with_bias, network)
        return q_values
    
    def select_action(self, state: List[float]) -> int:
        """Select action using epsilon-greedy policy.
        
        Args:
            state (List[float]): Current state.
        
        Returns:
            int: Index of selected action.
        """
        # Epsilon-greedy: with probability (1 - epsilon_greedy) explore, else exploit
        if np.random.rand() >= self.epsilon_greedy:
            # Exploration: random action
            return random.randrange(self.action_dim)
        
        # Exploitation: select action with highest Q-value
        q_values = self._predict(state, self.q_network)
        return int(np.argmax(q_values))
    
    def remember(self, 
                 state: List[float], 
                 action: int, 
                 reward: float, 
                 next_state: List[float], 
                 done: bool) -> None:
        """Store experience in replay buffer.
        
        Args:
            state (List[float]): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (List[float]): Next state.
            done (bool): Whether episode is done.
        """
        self.memory.append((state, action, reward, next_state, done))
        
        # Decay epsilon_greedy (matches C++ store_transition behavior)
        if self.epsilon_greedy > self.epsilon_greedy_min:
            self.epsilon_greedy *= self.epsilon_greedy_decay
    
    def replay(self) -> Optional[float]:
        """Train the network using experience replay.
            
        Samples a mini-batch from memory and performs gradient descent
        to update Q-network weights.
            
        Returns:
            Optional[float]: Training loss, or None if not enough samples.
        """
        if len(self.memory) < self.batch_size:
            return None
            
        # Sample mini-batch
        minibatch = random.sample(self.memory, self.batch_size)
            
        total_loss = 0.0
        for state, action, reward, next_state, done in minibatch:
            # Compute target Q-value
            if done:
                target_q = reward
            else:
                next_q_values = self._predict(next_state, self.target_network)
                target_q = reward + self.reward_decay * np.max(next_q_values)
                
            # Current Q-value prediction
            state_with_bias = np.append(state, 1.0)
            current_q_values = self._predict(state, self.q_network)
                
            # Compute loss (TD error)
            td_error = target_q - current_q_values[action]
            total_loss += td_error ** 2
                
            # Gradient descent update following Q-learning loss function
            # Loss = (target - Q(s,a))^2, so gradient = 2 * (Q(s,a) - target) * gradient_w.r.t.Q
            # For linear function approximation: Q(s,a) = w^T * phi(s), so dQ/dw = phi(s)
            # The gradient of loss w.r.t. weights is: -2 * (target - Q(s,a)) * phi(s)
            # Simplified as: td_error * phi(s) where td_error = (target - Q(s,a))
            gradient = state_with_bias * td_error
            # Apply gradient clipping to prevent large updates
            gradient = np.clip(gradient, -1.0, 1.0)
            self.q_network[:, action] -= self.learning_rate * gradient  # Minus because we want to go down the gradient
            
        return total_loss / self.batch_size
    
    def update_target_network(self) -> None:
        """Update target network with current Q-network weights."""
        self._update_target_network()
    
    def get_q_values(self, state: List[float]) -> np.ndarray:
        """Get Q-values for all actions in given state.
        
        Args:
            state (List[float]): Current state.
        
        Returns:
            np.ndarray: Q-values for all actions.
        """
        return self._predict(state, self.q_network)
