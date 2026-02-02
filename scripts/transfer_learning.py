#!/usr/bin/env python3
"""Transfer learning script for applying knowledge from source to target task."""

import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging
import time

from src.utils.core import set_seed, setup_logging, create_directories
from src.envs.utils import EnvironmentFactory, StateDiscretizer, EpisodeLogger, RewardShapingWrapper
from src.algorithms.q_learning import QLearningAgent


class TransferLearningExperiment:
    """Experiment runner for transfer learning."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize experiment with configuration.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.logger = setup_logging(
            config["logging"]["level"],
            config["logging"]["log_file"]
        )
        
        # Set random seed
        set_seed(config["env"]["seed"])
        
        # Create output directories
        create_directories(".")
        
        # Initialize environment and agent
        self.env = self._create_environment()
        self.agent = self._create_agent()
        
        # Initialize state discretizer for continuous environments
        if hasattr(self.env.observation_space, 'shape'):
            self.state_discretizer = StateDiscretizer(
                self.env.observation_space, 
                n_bins=20
            )
        else:
            self.state_discretizer = None
        
        # Experiment statistics
        self.experiment_stats = {
            "episode_rewards": [],
            "episode_lengths": [],
            "eval_rewards": [],
            "eval_lengths": [],
            "exploration_rates": [],
            "transfer_efficiency": []
        }
    
    def _create_environment(self):
        """Create and configure environment."""
        env_name = self.config["env"]["name"]
        seed = self.config["env"]["seed"]
        
        # Create base environment
        env = EnvironmentFactory.create_env(env_name, seed=seed)
        
        # Add wrappers
        wrappers = []
        
        # Add episode logger
        wrappers.append({
            "class": EpisodeLogger,
            "kwargs": {"log_every": self.config["training"]["log_frequency"]}
        })
        
        # Add reward shaping if needed
        if env_name == "MountainCar-v0":
            wrappers.append({
                "class": RewardShapingWrapper,
                "kwargs": {"reward_scale": 1.0, "reward_shift": 0.0}
            })
        
        # Apply wrappers
        for wrapper_config in wrappers:
            wrapper_class = wrapper_config["class"]
            wrapper_kwargs = wrapper_config.get("kwargs", {})
            env = wrapper_class(env, **wrapper_kwargs)
        
        return env
    
    def _create_agent(self):
        """Create agent based on configuration."""
        agent_config = self.config["agent"]
        
        # Determine action space size
        action_space_size = self.env.action_space.n
        
        # Determine state dimension for neural network
        state_dim = None
        if agent_config.get("use_neural_network", False):
            if hasattr(self.env.observation_space, 'shape'):
                state_dim = self.env.observation_space.shape[0]
            else:
                raise ValueError("Neural network requires continuous state space")
        
        agent = QLearningAgent(
            action_space_size=action_space_size,
            learning_rate=agent_config["learning_rate"],
            discount_factor=agent_config["discount_factor"],
            exploration_rate=agent_config["exploration_rate"],
            exploration_decay=agent_config["exploration_decay"],
            exploration_min=agent_config["exploration_min"],
            use_neural_network=agent_config.get("use_neural_network", False),
            state_dim=state_dim
        )
        
        return agent
    
    def _discretize_state(self, state: np.ndarray) -> Tuple[int, ...]:
        """Discretize continuous state for tabular methods.
        
        Args:
            state: Continuous state.
            
        Returns:
            Discretized state tuple.
        """
        if self.state_discretizer is not None:
            return self.state_discretizer.discretize(state)
        return tuple(state) if isinstance(state, (list, tuple)) else (state,)
    
    def train_episode(self) -> Tuple[float, int]:
        """Train agent for one episode.
        
        Returns:
            Tuple of (total_reward, episode_length).
        """
        state, _ = self.env.reset()
        state = self._discretize_state(state)
        
        total_reward = 0.0
        episode_length = 0
        done = False
        
        while not done:
            action = self.agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            next_state = self._discretize_state(next_state)
            self.agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            episode_length += 1
        
        # Decay exploration rate
        self.agent.decay_exploration()
        
        return total_reward, episode_length
    
    def evaluate(self, n_episodes: int = 100) -> Tuple[float, float]:
        """Evaluate agent performance.
        
        Args:
            n_episodes: Number of episodes to evaluate.
            
        Returns:
            Tuple of (average_reward, average_length).
        """
        eval_rewards = []
        eval_lengths = []
        
        for _ in range(n_episodes):
            state, _ = self.env.reset()
            state = self._discretize_state(state)
            
            total_reward = 0.0
            episode_length = 0
            done = False
            
            while not done:
                action = self.agent.select_action(state, training=False)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                state = self._discretize_state(next_state)
                total_reward += reward
                episode_length += 1
            
            eval_rewards.append(total_reward)
            eval_lengths.append(episode_length)
        
        return np.mean(eval_rewards), np.mean(eval_lengths)
    
    def run_transfer_experiment(self):
        """Run transfer learning experiment."""
        self.logger.info("Starting transfer learning experiment...")
        self.logger.info(f"Target Environment: {self.config['env']['name']}")
        self.logger.info(f"Agent: {self.config['agent']['type']}")
        
        # Load source model if transfer is enabled
        if self.config["transfer"]["enabled"]:
            source_model_path = self.config["transfer"]["source_model_path"]
            if Path(source_model_path).exists():
                # Create temporary source agent to load weights
                source_agent = QLearningAgent(
                    action_space_size=self.env.action_space.n,
                    use_neural_network=self.config["agent"].get("use_neural_network", False)
                )
                source_agent.load(source_model_path)
                
                # Transfer knowledge
                self.agent.transfer_knowledge(
                    source_agent, 
                    self.config["transfer"]["transfer_type"]
                )
                self.logger.info(f"Transferred knowledge from {source_model_path}")
            else:
                self.logger.warning(f"Source model not found: {source_model_path}")
        
        start_time = time.time()
        
        # Track performance milestones for transfer efficiency
        milestones = [100, 200, 500, 1000]  # Episodes to check performance
        milestone_rewards = []
        
        for episode in range(self.config["training"]["total_episodes"]):
            # Train one episode
            reward, length = self.train_episode()
            
            # Store statistics
            self.experiment_stats["episode_rewards"].append(reward)
            self.experiment_stats["episode_lengths"].append(length)
            self.experiment_stats["exploration_rates"].append(self.agent.exploration_rate)
            
            # Check milestones
            if (episode + 1) in milestones:
                eval_reward, eval_length = self.evaluate(self.config["training"]["eval_episodes"])
                milestone_rewards.append(eval_reward)
                self.logger.info(f"Milestone {episode + 1}: Eval Reward: {eval_reward:.2f}")
            
            # Evaluation
            if (episode + 1) % self.config["training"]["eval_frequency"] == 0:
                eval_reward, eval_length = self.evaluate(self.config["training"]["eval_episodes"])
                self.experiment_stats["eval_rewards"].append(eval_reward)
                self.experiment_stats["eval_lengths"].append(eval_length)
                
                self.logger.info(
                    f"Episode {episode + 1}: "
                    f"Train Reward: {reward:.2f}, "
                    f"Eval Reward: {eval_reward:.2f}, "
                    f"Exploration: {self.agent.exploration_rate:.3f}"
                )
            
            # Save model
            if (episode + 1) % self.config["training"]["save_frequency"] == 0:
                self.agent.save(self.config["output"]["model_path"])
                self.logger.info(f"Model saved at episode {episode + 1}")
        
        training_time = time.time() - start_time
        self.logger.info(f"Transfer learning completed in {training_time:.2f} seconds")
        
        # Final evaluation
        final_eval_reward, final_eval_length = self.evaluate(self.config["training"]["eval_episodes"])
        self.logger.info(f"Final evaluation: Reward: {final_eval_reward:.2f}, Length: {final_eval_length:.2f}")
        
        # Calculate transfer efficiency
        if milestone_rewards:
            self.experiment_stats["transfer_efficiency"] = milestone_rewards
            self.logger.info(f"Transfer efficiency milestones: {milestone_rewards}")
        
        # Save final model
        self.agent.save(self.config["output"]["model_path"])
        
        # Generate plots
        self._generate_plots()
    
    def _generate_plots(self):
        """Generate experiment plots."""
        plots_path = Path(self.config["output"]["plots_path"])
        plots_path.mkdir(parents=True, exist_ok=True)
        
        # Training rewards
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.plot(self.experiment_stats["episode_rewards"])
        plt.title("Training Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True)
        
        plt.subplot(2, 3, 2)
        plt.plot(self.experiment_stats["episode_lengths"])
        plt.title("Episode Lengths")
        plt.xlabel("Episode")
        plt.ylabel("Length")
        plt.grid(True)
        
        plt.subplot(2, 3, 3)
        if self.experiment_stats["eval_rewards"]:
            eval_episodes = np.arange(
                self.config["training"]["eval_frequency"],
                len(self.experiment_stats["episode_rewards"]) + 1,
                self.config["training"]["eval_frequency"]
            )
            plt.plot(eval_episodes, self.experiment_stats["eval_rewards"])
            plt.title("Evaluation Rewards")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.grid(True)
        
        plt.subplot(2, 3, 4)
        plt.plot(self.experiment_stats["exploration_rates"])
        plt.title("Exploration Rate")
        plt.xlabel("Episode")
        plt.ylabel("Exploration Rate")
        plt.grid(True)
        
        plt.subplot(2, 3, 5)
        if self.experiment_stats["transfer_efficiency"]:
            milestones = [100, 200, 500, 1000]
            plt.plot(milestones, self.experiment_stats["transfer_efficiency"], 'o-')
            plt.title("Transfer Efficiency")
            plt.xlabel("Episode")
            plt.ylabel("Average Reward")
            plt.grid(True)
        
        plt.subplot(2, 3, 6)
        # Rolling average of rewards
        window = 50
        if len(self.experiment_stats["episode_rewards"]) >= window:
            rolling_avg = np.convolve(
                self.experiment_stats["episode_rewards"], 
                np.ones(window)/window, 
                mode='valid'
            )
            plt.plot(range(window-1, len(self.experiment_stats["episode_rewards"])), rolling_avg)
            plt.title(f"Rolling Average Rewards (window={window})")
            plt.xlabel("Episode")
            plt.ylabel("Average Reward")
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(plots_path / "transfer_learning_curves.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        self.logger.info(f"Plots saved to {plots_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run transfer learning experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--source-model", type=str, help="Path to source model for transfer")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override source model path if provided
    if args.source_model:
        config["transfer"]["source_model_path"] = args.source_model
    
    # Create experiment runner
    experiment = TransferLearningExperiment(config)
    
    # Run transfer experiment
    experiment.run_transfer_experiment()


if __name__ == "__main__":
    main()
