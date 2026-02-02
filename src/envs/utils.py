"""Environment utilities and wrappers for transfer reinforcement learning."""

import gymnasium as gym
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from gymnasium import Env, Wrapper
from gymnasium.spaces import Box, Discrete, Space
import logging


class StateDiscretizer:
    """Discretize continuous state spaces for tabular methods."""
    
    def __init__(self, state_space: Space, n_bins: int = 10):
        """Initialize state discretizer.
        
        Args:
            state_space: Original state space.
            n_bins: Number of bins per dimension.
        """
        self.state_space = state_space
        self.n_bins = n_bins
        
        if isinstance(state_space, Box):
            self.low = state_space.low
            self.high = state_space.high
            self.shape = state_space.shape
        else:
            raise ValueError("State discretizer only supports Box spaces")
    
    def discretize(self, state: np.ndarray) -> Tuple[int, ...]:
        """Discretize continuous state.
        
        Args:
            state: Continuous state vector.
            
        Returns:
            Discretized state tuple.
        """
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        
        discretized = []
        for i in range(self.shape[0]):
            # Normalize to [0, 1]
            normalized = (state[0, i] - self.low[i]) / (self.high[i] - self.low[i])
            # Clip to valid range
            normalized = np.clip(normalized, 0.0, 1.0)
            # Discretize
            bin_idx = int(normalized * (self.n_bins - 1))
            discretized.append(bin_idx)
        
        return tuple(discretized)


class RewardShapingWrapper(Wrapper):
    """Wrapper for reward shaping and normalization."""
    
    def __init__(self, env: Env, reward_scale: float = 1.0, reward_shift: float = 0.0):
        """Initialize reward shaping wrapper.
        
        Args:
            env: Environment to wrap.
            reward_scale: Reward scaling factor.
            reward_shift: Reward shift factor.
        """
        super().__init__(env)
        self.reward_scale = reward_scale
        self.reward_shift = reward_shift
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Step environment with reward shaping.
        
        Args:
            action: Action to take.
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped_reward = self.reward_scale * reward + self.reward_shift
        return obs, shaped_reward, terminated, truncated, info


class EpisodeLogger(Wrapper):
    """Wrapper for logging episode statistics."""
    
    def __init__(self, env: Env, log_every: int = 100):
        """Initialize episode logger.
        
        Args:
            env: Environment to wrap.
            log_every: Log every N episodes.
        """
        super().__init__(env)
        self.log_every = log_every
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.logger = logging.getLogger("transfer_rl.episode_logger")
    
    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Reset environment and log episode statistics.
        
        Args:
            **kwargs: Reset arguments.
            
        Returns:
            Tuple of (observation, info).
        """
        if self.episode_count > 0:
            self.episode_rewards.append(self.current_reward)
            self.episode_lengths.append(self.current_length)
            
            if self.episode_count % self.log_every == 0:
                avg_reward = np.mean(self.episode_rewards[-self.log_every:])
                avg_length = np.mean(self.episode_lengths[-self.log_every:])
                self.logger.info(
                    f"Episode {self.episode_count}: "
                    f"Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.2f}"
                )
        
        obs, info = self.env.reset(**kwargs)
        self.episode_count += 1
        self.current_reward = 0.0
        self.current_length = 0
        return obs, info
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Step environment and track episode statistics.
        
        Args:
            action: Action to take.
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_reward += reward
        self.current_length += 1
        return obs, reward, terminated, truncated, info


class EnvironmentFactory:
    """Factory for creating and configuring environments."""
    
    @staticmethod
    def create_env(
        env_name: str,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        **kwargs
    ) -> Env:
        """Create environment with proper configuration.
        
        Args:
            env_name: Name of the environment.
            seed: Random seed for environment.
            render_mode: Rendering mode.
            **kwargs: Additional environment arguments.
            
        Returns:
            Configured environment.
        """
        env = gym.make(env_name, render_mode=render_mode, **kwargs)
        
        if seed is not None:
            env.reset(seed=seed)
        
        return env
    
    @staticmethod
    def create_wrapped_env(
        env_name: str,
        wrappers: List[Dict[str, Any]],
        seed: Optional[int] = None,
        **kwargs
    ) -> Env:
        """Create environment with multiple wrappers.
        
        Args:
            env_name: Name of the environment.
            wrappers: List of wrapper configurations.
            seed: Random seed for environment.
            **kwargs: Additional environment arguments.
            
        Returns:
            Environment with applied wrappers.
        """
        env = EnvironmentFactory.create_env(env_name, seed=seed, **kwargs)
        
        for wrapper_config in wrappers:
            wrapper_class = wrapper_config["class"]
            wrapper_kwargs = wrapper_config.get("kwargs", {})
            env = wrapper_class(env, **wrapper_kwargs)
        
        return env


def get_env_info(env: Env) -> Dict[str, Any]:
    """Get comprehensive environment information.
    
    Args:
        env: Environment to analyze.
        
    Returns:
        Dictionary with environment information.
    """
    info = {
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "observation_space_type": type(env.observation_space).__name__,
        "action_space_type": type(env.action_space).__name__,
    }
    
    if isinstance(env.observation_space, Box):
        info["observation_dim"] = env.observation_space.shape[0]
        info["observation_low"] = env.observation_space.low
        info["observation_high"] = env.observation_space.high
    
    if isinstance(env.action_space, Discrete):
        info["action_dim"] = env.action_space.n
    elif isinstance(env.action_space, Box):
        info["action_dim"] = env.action_space.shape[0]
        info["action_low"] = env.action_space.low
        info["action_high"] = env.action_space.high
    
    return info
