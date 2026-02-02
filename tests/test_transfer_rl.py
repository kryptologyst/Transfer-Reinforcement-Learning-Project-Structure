"""Test suite for transfer reinforcement learning."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch

from src.utils.core import set_seed, get_device, setup_logging
from src.envs.utils import StateDiscretizer, RewardShapingWrapper, EpisodeLogger
from src.algorithms.q_learning import QLearningAgent


class TestCoreUtils:
    """Test core utility functions."""
    
    def test_set_seed(self):
        """Test seed setting functionality."""
        set_seed(42)
        
        # Test numpy random
        np.random.seed(42)
        val1 = np.random.random()
        np.random.seed(42)
        val2 = np.random.random()
        assert val1 == val2
        
        # Test torch random
        torch.manual_seed(42)
        val1 = torch.rand(1).item()
        torch.manual_seed(42)
        val2 = torch.rand(1).item()
        assert val1 == val2
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ['cpu', 'cuda', 'mps']
    
    def test_setup_logging(self):
        """Test logging setup."""
        logger = setup_logging("INFO")
        assert logger.name == "transfer_rl"
        assert logger.level == 20  # INFO level


class TestStateDiscretizer:
    """Test state discretization functionality."""
    
    def test_discretize_continuous_state(self):
        """Test discretization of continuous states."""
        from gymnasium.spaces import Box
        
        state_space = Box(low=-1.0, high=1.0, shape=(2,))
        discretizer = StateDiscretizer(state_space, n_bins=10)
        
        # Test discretization
        state = np.array([0.0, 0.5])
        discretized = discretizer.discretize(state)
        
        assert isinstance(discretized, tuple)
        assert len(discretized) == 2
        assert all(isinstance(x, int) for x in discretized)
        assert all(0 <= x < 10 for x in discretized)
    
    def test_discretize_edge_cases(self):
        """Test edge cases in discretization."""
        from gymnasium.spaces import Box
        
        state_space = Box(low=-1.0, high=1.0, shape=(2,))
        discretizer = StateDiscretizer(state_space, n_bins=5)
        
        # Test boundary values
        state_low = np.array([-1.0, -1.0])
        state_high = np.array([1.0, 1.0])
        
        discretized_low = discretizer.discretize(state_low)
        discretized_high = discretizer.discretize(state_high)
        
        assert discretized_low == (0, 0)
        assert discretized_high == (4, 4)


class TestRewardShapingWrapper:
    """Test reward shaping wrapper."""
    
    def test_reward_shaping(self):
        """Test reward shaping functionality."""
        # Mock environment
        mock_env = Mock()
        mock_env.step.return_value = (np.array([1, 2]), 1.0, False, False, {})
        
        wrapper = RewardShapingWrapper(mock_env, reward_scale=2.0, reward_shift=1.0)
        
        obs, reward, terminated, truncated, info = wrapper.step(0)
        
        assert reward == 3.0  # 2.0 * 1.0 + 1.0
        mock_env.step.assert_called_once_with(0)


class TestQLearningAgent:
    """Test Q-Learning agent functionality."""
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = QLearningAgent(action_space_size=4)
        
        assert agent.action_space_size == 4
        assert agent.learning_rate == 0.1
        assert agent.discount_factor == 0.99
        assert agent.exploration_rate == 1.0
        assert not agent.use_neural_network
    
    def test_tabular_action_selection(self):
        """Test tabular action selection."""
        agent = QLearningAgent(action_space_size=4, use_neural_network=False)
        
        # Test exploration
        with patch('numpy.random.random', return_value=0.1):
            action = agent.select_action((0, 0), training=True)
            assert 0 <= action < 4
        
        # Test exploitation
        agent.q_table[(0, 0)] = np.array([0.1, 0.9, 0.2, 0.3])
        with patch('numpy.random.random', return_value=0.9):
            action = agent.select_action((0, 0), training=True)
            assert action == 1  # argmax of q_values
    
    def test_tabular_q_update(self):
        """Test tabular Q-value updates."""
        agent = QLearningAgent(action_space_size=4, use_neural_network=False)
        
        # Initial Q-value
        state = (0, 0)
        action = 1
        reward = 1.0
        next_state = (1, 1)
        
        agent.update(state, action, reward, next_state, False)
        
        # Check that Q-value was updated
        assert state in agent.q_table
        assert agent.q_table[state][action] > 0
    
    def test_exploration_decay(self):
        """Test exploration rate decay."""
        agent = QLearningAgent(
            action_space_size=4,
            exploration_rate=1.0,
            exploration_decay=0.9,
            exploration_min=0.1
        )
        
        initial_rate = agent.exploration_rate
        agent.decay_exploration()
        
        assert agent.exploration_rate < initial_rate
        assert agent.exploration_rate >= 0.1
    
    def test_neural_network_agent(self):
        """Test neural network agent."""
        agent = QLearningAgent(
            action_space_size=4,
            use_neural_network=True,
            state_dim=2
        )
        
        assert agent.use_neural_network
        assert agent.q_network is not None
        assert agent.optimizer is not None
    
    def test_knowledge_transfer(self):
        """Test knowledge transfer between agents."""
        source_agent = QLearningAgent(action_space_size=4, use_neural_network=False)
        target_agent = QLearningAgent(action_space_size=4, use_neural_network=False)
        
        # Train source agent
        source_agent.q_table[(0, 0)] = np.array([0.1, 0.9, 0.2, 0.3])
        
        # Transfer knowledge
        target_agent.transfer_knowledge(source_agent, "full")
        
        # Check that knowledge was transferred
        assert (0, 0) in target_agent.q_table
        assert np.array_equal(target_agent.q_table[(0, 0)], source_agent.q_table[(0, 0)])
    
    def test_agent_save_load(self):
        """Test agent save and load functionality."""
        agent = QLearningAgent(action_space_size=4)
        
        # Train agent
        agent.q_table[(0, 0)] = np.array([0.1, 0.9, 0.2, 0.3])
        
        # Save agent
        agent.save("test_agent.pth")
        
        # Create new agent and load
        new_agent = QLearningAgent(action_space_size=4)
        new_agent.load("test_agent.pth")
        
        # Check that knowledge was loaded
        assert (0, 0) in new_agent.q_table
        assert np.array_equal(new_agent.q_table[(0, 0)], agent.q_table[(0, 0)])
        
        # Clean up
        import os
        os.remove("test_agent.pth")


class TestIntegration:
    """Integration tests."""
    
    def test_cartpole_training(self):
        """Test training on CartPole environment."""
        import gymnasium as gym
        
        env = gym.make("CartPole-v1")
        agent = QLearningAgent(action_space_size=env.action_space.n)
        
        # Simple training loop
        for episode in range(10):
            state, _ = env.reset()
            state = tuple(state)
            
            done = False
            while not done:
                action = agent.select_action(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                next_state = tuple(next_state)
                agent.update(state, action, reward, next_state, done)
                state = next_state
        
        # Check that agent learned something
        assert len(agent.q_table) > 0
        env.close()
    
    def test_mountain_car_training(self):
        """Test training on MountainCar environment."""
        import gymnasium as gym
        
        env = gym.make("MountainCar-v0")
        
        # Use state discretizer for continuous state space
        from gymnasium.spaces import Box
        state_space = Box(low=env.observation_space.low, high=env.observation_space.high)
        discretizer = StateDiscretizer(state_space, n_bins=20)
        
        agent = QLearningAgent(action_space_size=env.action_space.n)
        
        # Simple training loop
        for episode in range(10):
            state, _ = env.reset()
            state = discretizer.discretize(state)
            
            done = False
            while not done:
                action = agent.select_action(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                next_state = discretizer.discretize(next_state)
                agent.update(state, action, reward, next_state, done)
                state = next_state
        
        # Check that agent learned something
        assert len(agent.q_table) > 0
        env.close()


if __name__ == "__main__":
    pytest.main([__file__])
