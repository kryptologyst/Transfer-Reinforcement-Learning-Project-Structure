#!/usr/bin/env python3
"""Evaluation script for transfer learning experiments."""

import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging
from scipy import stats

from src.utils.core import set_seed, setup_logging, create_directories
from src.envs.utils import EnvironmentFactory, StateDiscretizer
from src.algorithms.q_learning import QLearningAgent


class TransferLearningEvaluator:
    """Evaluator for transfer learning experiments."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize evaluator with configuration.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.logger = setup_logging(
            config["logging"]["level"],
            config["logging"]["log_file"]
        )
        
        # Create output directories
        create_directories(".")
        
        # Initialize environment
        self.env = self._create_environment()
        
        # Initialize state discretizer for continuous environments
        if hasattr(self.env.observation_space, 'shape'):
            self.state_discretizer = StateDiscretizer(
                self.env.observation_space, 
                n_bins=20
            )
        else:
            self.state_discretizer = None
    
    def _create_environment(self):
        """Create and configure environment."""
        env_name = self.config["env"]["name"]
        seed = self.config["env"]["seed"]
        
        # Create base environment
        env = EnvironmentFactory.create_env(env_name, seed=seed)
        
        return env
    
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
    
    def evaluate_agent(self, agent: QLearningAgent, n_episodes: int, n_seeds: int = 1) -> Dict[str, Any]:
        """Evaluate agent performance across multiple seeds.
        
        Args:
            agent: Agent to evaluate.
            n_episodes: Number of episodes per seed.
            n_seeds: Number of random seeds.
            
        Returns:
            Dictionary with evaluation results.
        """
        all_rewards = []
        all_lengths = []
        success_rates = []
        
        for seed in range(n_seeds):
            set_seed(seed)
            seed_rewards = []
            seed_lengths = []
            successes = 0
            
            for _ in range(n_episodes):
                state, _ = self.env.reset()
                state = self._discretize_state(state)
                
                total_reward = 0.0
                episode_length = 0
                done = False
                
                while not done:
                    action = agent.select_action(state, training=False)
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    
                    state = self._discretize_state(next_state)
                    total_reward += reward
                    episode_length += 1
                
                seed_rewards.append(total_reward)
                seed_lengths.append(episode_length)
                
                # Define success criteria based on environment
                if self.config["env"]["name"] == "MountainCar-v0":
                    # Success if reached the goal (reward > -200)
                    if total_reward > -200:
                        successes += 1
                elif self.config["env"]["name"] == "CartPole-v1":
                    # Success if episode lasted full length
                    if episode_length >= 500:
                        successes += 1
            
            all_rewards.extend(seed_rewards)
            all_lengths.extend(seed_lengths)
            success_rates.append(successes / n_episodes)
        
        # Calculate statistics
        mean_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        mean_length = np.mean(all_lengths)
        std_length = np.std(all_lengths)
        mean_success_rate = np.mean(success_rates)
        std_success_rate = np.std(success_rates)
        
        # Confidence intervals
        confidence_level = self.config["evaluation"]["confidence_interval"]
        ci_reward = stats.t.interval(
            confidence_level, 
            len(all_rewards) - 1, 
            loc=mean_reward, 
            scale=stats.sem(all_rewards)
        )
        ci_length = stats.t.interval(
            confidence_level, 
            len(all_lengths) - 1, 
            loc=mean_length, 
            scale=stats.sem(all_lengths)
        )
        
        return {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "ci_reward": ci_reward,
            "mean_length": mean_length,
            "std_length": std_length,
            "ci_length": ci_length,
            "mean_success_rate": mean_success_rate,
            "std_success_rate": std_success_rate,
            "all_rewards": all_rewards,
            "all_lengths": all_lengths,
            "success_rates": success_rates
        }
    
    def compare_agents(self, baseline_agent: QLearningAgent, transfer_agent: QLearningAgent) -> Dict[str, Any]:
        """Compare baseline and transfer agents.
        
        Args:
            baseline_agent: Baseline agent (trained from scratch).
            transfer_agent: Transfer agent (with transferred knowledge).
            
        Returns:
            Dictionary with comparison results.
        """
        self.logger.info("Evaluating baseline agent...")
        baseline_results = self.evaluate_agent(
            baseline_agent, 
            self.config["evaluation"]["n_eval_episodes"],
            self.config["evaluation"]["n_seeds"]
        )
        
        self.logger.info("Evaluating transfer agent...")
        transfer_results = self.evaluate_agent(
            transfer_agent, 
            self.config["evaluation"]["n_eval_episodes"],
            self.config["evaluation"]["n_seeds"]
        )
        
        # Calculate transfer efficiency
        transfer_efficiency = transfer_results["mean_reward"] / baseline_results["mean_reward"]
        
        # Statistical significance test
        _, p_value = stats.ttest_ind(
            baseline_results["all_rewards"], 
            transfer_results["all_rewards"]
        )
        
        comparison_results = {
            "baseline": baseline_results,
            "transfer": transfer_results,
            "transfer_efficiency": transfer_efficiency,
            "p_value": p_value,
            "significant": p_value < 0.05
        }
        
        return comparison_results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate evaluation report.
        
        Args:
            results: Evaluation results.
            
        Returns:
            Report string.
        """
        report = f"""
# Transfer Learning Evaluation Report

## Environment: {self.config['env']['name']}

## Evaluation Settings
- Episodes per seed: {self.config['evaluation']['n_eval_episodes']}
- Number of seeds: {self.config['evaluation']['n_seeds']}
- Confidence interval: {self.config['evaluation']['confidence_interval']}

## Results

### Baseline Agent (Trained from Scratch)
- Average Reward: {results['baseline']['mean_reward']:.2f} ± {results['baseline']['std_reward']:.2f}
- 95% CI: [{results['baseline']['ci_reward'][0]:.2f}, {results['baseline']['ci_reward'][1]:.2f}]
- Average Length: {results['baseline']['mean_length']:.2f} ± {results['baseline']['std_length']:.2f}
- Success Rate: {results['baseline']['mean_success_rate']:.2f} ± {results['baseline']['std_success_rate']:.2f}

### Transfer Agent (With Transferred Knowledge)
- Average Reward: {results['transfer']['mean_reward']:.2f} ± {results['transfer']['std_reward']:.2f}
- 95% CI: [{results['transfer']['ci_reward'][0]:.2f}, {results['transfer']['ci_reward'][1]:.2f}]
- Average Length: {results['transfer']['mean_length']:.2f} ± {results['transfer']['std_length']:.2f}
- Success Rate: {results['transfer']['mean_success_rate']:.2f} ± {results['transfer']['std_success_rate']:.2f}

### Transfer Efficiency
- Transfer Efficiency: {results['transfer_efficiency']:.2f}
- Statistical Significance: {'Yes' if results['significant'] else 'No'} (p = {results['p_value']:.4f})

## Interpretation
"""
        
        if results['transfer_efficiency'] > 1.0:
            report += "Transfer learning improved performance over baseline training.\n"
        elif results['transfer_efficiency'] > 0.8:
            report += "Transfer learning provided modest improvement over baseline training.\n"
        else:
            report += "Transfer learning did not improve performance over baseline training.\n"
        
        if results['significant']:
            report += "The difference between baseline and transfer agents is statistically significant.\n"
        else:
            report += "The difference between baseline and transfer agents is not statistically significant.\n"
        
        return report
    
    def generate_plots(self, results: Dict[str, Any]):
        """Generate evaluation plots.
        
        Args:
            results: Evaluation results.
        """
        plots_path = Path(self.config["output"]["plots_path"])
        plots_path.mkdir(parents=True, exist_ok=True)
        
        # Reward comparison
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.hist(results['baseline']['all_rewards'], alpha=0.7, label='Baseline', bins=20)
        plt.hist(results['transfer']['all_rewards'], alpha=0.7, label='Transfer', bins=20)
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.title('Reward Distribution')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 3, 2)
        plt.hist(results['baseline']['all_lengths'], alpha=0.7, label='Baseline', bins=20)
        plt.hist(results['transfer']['all_lengths'], alpha=0.7, label='Transfer', bins=20)
        plt.xlabel('Episode Length')
        plt.ylabel('Frequency')
        plt.title('Episode Length Distribution')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 3, 3)
        categories = ['Baseline', 'Transfer']
        means = [results['baseline']['mean_reward'], results['transfer']['mean_reward']]
        stds = [results['baseline']['std_reward'], results['transfer']['std_reward']]
        plt.bar(categories, means, yerr=stds, capsize=5)
        plt.ylabel('Average Reward')
        plt.title('Average Reward Comparison')
        plt.grid(True)
        
        plt.subplot(2, 3, 4)
        success_rates = [results['baseline']['mean_success_rate'], results['transfer']['mean_success_rate']]
        success_stds = [results['baseline']['std_success_rate'], results['transfer']['std_success_rate']]
        plt.bar(categories, success_rates, yerr=success_stds, capsize=5)
        plt.ylabel('Success Rate')
        plt.title('Success Rate Comparison')
        plt.grid(True)
        
        plt.subplot(2, 3, 5)
        plt.bar(['Transfer Efficiency'], [results['transfer_efficiency']])
        plt.ylabel('Transfer Efficiency')
        plt.title('Transfer Efficiency')
        plt.grid(True)
        
        plt.subplot(2, 3, 6)
        # Box plot comparison
        data = [results['baseline']['all_rewards'], results['transfer']['all_rewards']]
        plt.boxplot(data, labels=['Baseline', 'Transfer'])
        plt.ylabel('Reward')
        plt.title('Reward Box Plot Comparison')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(plots_path / "evaluation_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        self.logger.info(f"Evaluation plots saved to {plots_path}")
    
    def run_evaluation(self):
        """Run complete evaluation."""
        self.logger.info("Starting evaluation...")
        
        # Load agents
        baseline_path = self.config["comparison"]["baseline_model_path"]
        transfer_path = self.config["comparison"]["transfer_model_path"]
        
        if not Path(baseline_path).exists():
            self.logger.error(f"Baseline model not found: {baseline_path}")
            return
        
        if not Path(transfer_path).exists():
            self.logger.error(f"Transfer model not found: {transfer_path}")
            return
        
        # Create agents
        baseline_agent = QLearningAgent(
            action_space_size=self.env.action_space.n,
            use_neural_network=False  # Assuming tabular for now
        )
        transfer_agent = QLearningAgent(
            action_space_size=self.env.action_space.n,
            use_neural_network=False
        )
        
        # Load models
        baseline_agent.load(baseline_path)
        transfer_agent.load(transfer_path)
        
        # Run comparison
        results = self.compare_agents(baseline_agent, transfer_agent)
        
        # Generate report
        report = self.generate_report(results)
        
        # Save results
        results_path = Path(self.config["output"]["results_path"])
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save report
        report_path = Path(self.config["output"]["report_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Generate plots
        self.generate_plots(results)
        
        # Print summary
        self.logger.info("Evaluation completed!")
        self.logger.info(f"Transfer Efficiency: {results['transfer_efficiency']:.2f}")
        self.logger.info(f"Statistical Significance: {'Yes' if results['significant'] else 'No'}")
        self.logger.info(f"Results saved to {results_path}")
        self.logger.info(f"Report saved to {report_path}")


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
    parser = argparse.ArgumentParser(description="Evaluate transfer learning performance")
    parser.add_argument("--config", type=str, required=True, help="Path to evaluation configuration file")
    parser.add_argument("--baseline-model", type=str, help="Path to baseline model")
    parser.add_argument("--transfer-model", type=str, help="Path to transfer model")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override model paths if provided
    if args.baseline_model:
        config["comparison"]["baseline_model_path"] = args.baseline_model
    if args.transfer_model:
        config["comparison"]["transfer_model_path"] = args.transfer_model
    
    # Create evaluator and run evaluation
    evaluator = TransferLearningEvaluator(config)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
