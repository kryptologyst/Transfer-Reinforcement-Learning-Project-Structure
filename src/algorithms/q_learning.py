"""Modern Q-Learning agent with transfer learning capabilities."""

import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict, deque
import logging
from abc import ABC, abstractmethod

from ..utils.core import get_device


class BaseAgent(ABC):
    """Abstract base class for RL agents."""
    
    def __init__(self, action_space_size: int, device: Optional[torch.device] = None):
        """Initialize base agent.
        
        Args:
            action_space_size: Number of possible actions.
            device: Device to run computations on.
        """
        self.action_space_size = action_space_size
        self.device = device or get_device()
        self.logger = logging.getLogger(f"transfer_rl.{self.__class__.__name__}")
    
    @abstractmethod
    def select_action(self, state: Any, training: bool = True) -> int:
        """Select action given state.
        
        Args:
            state: Current state.
            training: Whether in training mode.
            
        Returns:
            Selected action.
        """
        pass
    
    @abstractmethod
    def update(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        """Update agent with experience.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether episode is done.
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save agent to file.
        
        Args:
            filepath: Path to save file.
        """
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> None:
        """Load agent from file.
        
        Args:
            filepath: Path to load file from.
        """
        pass


class QLearningAgent(BaseAgent):
    """Modern Q-Learning agent with transfer learning support."""
    
    def __init__(
        self,
        action_space_size: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        exploration_rate: float = 1.0,
        exploration_decay: float = 0.995,
        exploration_min: float = 0.01,
        device: Optional[torch.device] = None,
        use_neural_network: bool = False,
        state_dim: Optional[int] = None,
        hidden_dims: List[int] = [64, 64]
    ):
        """Initialize Q-Learning agent.
        
        Args:
            action_space_size: Number of possible actions.
            learning_rate: Learning rate for Q-value updates.
            discount_factor: Discount factor for future rewards.
            exploration_rate: Initial exploration rate.
            exploration_decay: Exploration rate decay factor.
            exploration_min: Minimum exploration rate.
            device: Device to run computations on.
            use_neural_network: Whether to use neural network Q-function.
            state_dim: State dimension (required for neural network).
            hidden_dims: Hidden layer dimensions for neural network.
        """
        super().__init__(action_space_size, device)
        
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        self.use_neural_network = use_neural_network
        
        if use_neural_network:
            if state_dim is None:
                raise ValueError("state_dim must be provided when using neural network")
            self.q_network = self._build_q_network(state_dim, hidden_dims)
            self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
            self.q_table = None
        else:
            self.q_table = defaultdict(lambda: np.zeros(action_space_size))
            self.q_network = None
            self.optimizer = None
        
        self.training_stats = {
            "episode_rewards": deque(maxlen=1000),
            "episode_lengths": deque(maxlen=1000),
            "q_values": deque(maxlen=1000),
            "exploration_rates": deque(maxlen=1000)
        }
    
    def _build_q_network(self, state_dim: int, hidden_dims: List[int]) -> nn.Module:
        """Build Q-network.
        
        Args:
            state_dim: State dimension.
            hidden_dims: Hidden layer dimensions.
            
        Returns:
            Q-network model.
        """
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, self.action_space_size))
        
        return nn.Sequential(*layers).to(self.device)
    
    def select_action(self, state: Any, training: bool = True) -> int:
        """Select action using epsilon-greedy policy.
        
        Args:
            state: Current state.
            training: Whether in training mode.
            
        Returns:
            Selected action.
        """
        if self.use_neural_network:
            return self._select_action_nn(state, training)
        else:
            return self._select_action_tabular(state, training)
    
    def _select_action_tabular(self, state: Any, training: bool) -> int:
        """Select action using tabular Q-values.
        
        Args:
            state: Current state.
            training: Whether in training mode.
            
        Returns:
            Selected action.
        """
        if training and np.random.random() < self.exploration_rate:
            return np.random.randint(0, self.action_space_size)
        else:
            q_values = self.q_table[state]
            return np.argmax(q_values)
    
    def _select_action_nn(self, state: Any, training: bool) -> int:
        """Select action using neural network Q-values.
        
        Args:
            state: Current state.
            training: Whether in training mode.
            
        Returns:
            Selected action.
        """
        if training and np.random.random() < self.exploration_rate:
            return np.random.randint(0, self.action_space_size)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
    
    def update(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        """Update Q-values with experience.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether episode is done.
        """
        if self.use_neural_network:
            self._update_nn(state, action, reward, next_state, done)
        else:
            self._update_tabular(state, action, reward, next_state, done)
    
    def _update_tabular(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        """Update tabular Q-values.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether episode is done.
        """
        current_q = self.q_table[state][action]
        
        if done:
            target_q = reward
        else:
            next_q_values = self.q_table[next_state]
            target_q = reward + self.discount_factor * np.max(next_q_values)
        
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.q_table[state][action] = new_q
    
    def _update_nn(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        """Update neural network Q-values.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether episode is done.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        current_q_values = self.q_network(state_tensor)
        current_q = current_q_values[0, action]
        
        with torch.no_grad():
            next_q_values = self.q_network(next_state_tensor)
            if done:
                target_q = reward
            else:
                target_q = reward + self.discount_factor * next_q_values.max().item()
        
        loss = nn.MSELoss()(current_q, torch.tensor(target_q, device=self.device))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def decay_exploration(self) -> None:
        """Decay exploration rate."""
        self.exploration_rate = max(
            self.exploration_min,
            self.exploration_rate * self.exploration_decay
        )
    
    def get_q_values(self, state: Any) -> np.ndarray:
        """Get Q-values for a state.
        
        Args:
            state: State to get Q-values for.
            
        Returns:
            Q-values array.
        """
        if self.use_neural_network:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.cpu().numpy().flatten()
        else:
            return self.q_table[state].copy()
    
    def transfer_knowledge(self, source_agent: 'QLearningAgent', transfer_type: str = "full") -> None:
        """Transfer knowledge from source agent.
        
        Args:
            source_agent: Source agent to transfer from.
            transfer_type: Type of transfer ("full", "partial", "weights").
        """
        if transfer_type == "full":
            if self.use_neural_network and source_agent.use_neural_network:
                # Transfer neural network weights
                self.q_network.load_state_dict(source_agent.q_network.state_dict())
            elif not self.use_neural_network and not source_agent.use_neural_network:
                # Transfer Q-table
                self.q_table = source_agent.q_table.copy()
            else:
                self.logger.warning("Cannot transfer between tabular and neural network agents")
        
        elif transfer_type == "partial":
            if not self.use_neural_network and not source_agent.use_neural_network:
                # Partial transfer of Q-table (only common states)
                for state, q_values in source_agent.q_table.items():
                    if state in self.q_table:
                        # Blend Q-values
                        self.q_table[state] = 0.5 * self.q_table[state] + 0.5 * q_values
        
        elif transfer_type == "weights":
            if self.use_neural_network and source_agent.use_neural_network:
                # Transfer only first few layers
                source_state_dict = source_agent.q_network.state_dict()
                target_state_dict = self.q_network.state_dict()
                
                for name, param in source_state_dict.items():
                    if name in target_state_dict and "weight" in name:
                        target_state_dict[name] = param
                
                self.q_network.load_state_dict(target_state_dict)
    
    def save(self, filepath: str) -> None:
        """Save agent to file.
        
        Args:
            filepath: Path to save file.
        """
        save_dict = {
            "action_space_size": self.action_space_size,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "exploration_rate": self.exploration_rate,
            "exploration_decay": self.exploration_decay,
            "exploration_min": self.exploration_min,
            "use_neural_network": self.use_neural_network,
            "training_stats": dict(self.training_stats)
        }
        
        if self.use_neural_network:
            save_dict["q_network_state_dict"] = self.q_network.state_dict()
        else:
            save_dict["q_table"] = dict(self.q_table)
        
        torch.save(save_dict, filepath)
        self.logger.info(f"Agent saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load agent from file.
        
        Args:
            filepath: Path to load file from.
        """
        save_dict = torch.load(filepath, map_location=self.device)
        
        self.action_space_size = save_dict["action_space_size"]
        self.learning_rate = save_dict["learning_rate"]
        self.discount_factor = save_dict["discount_factor"]
        self.exploration_rate = save_dict["exploration_rate"]
        self.exploration_decay = save_dict["exploration_decay"]
        self.exploration_min = save_dict["exploration_min"]
        self.use_neural_network = save_dict["use_neural_network"]
        self.training_stats = save_dict["training_stats"]
        
        if self.use_neural_network:
            self.q_network.load_state_dict(save_dict["q_network_state_dict"])
        else:
            self.q_table = defaultdict(lambda: np.zeros(self.action_space_size), save_dict["q_table"])
        
        self.logger.info(f"Agent loaded from {filepath}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics.
        
        Returns:
            Dictionary with training statistics.
        """
        stats = {}
        
        if self.training_stats["episode_rewards"]:
            stats["avg_reward"] = np.mean(self.training_stats["episode_rewards"])
            stats["std_reward"] = np.std(self.training_stats["episode_rewards"])
        
        if self.training_stats["episode_lengths"]:
            stats["avg_length"] = np.mean(self.training_stats["episode_lengths"])
            stats["std_length"] = np.std(self.training_stats["episode_lengths"])
        
        stats["exploration_rate"] = self.exploration_rate
        
        return stats
