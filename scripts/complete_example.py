#!/usr/bin/env python3
"""Complete example demonstrating transfer reinforcement learning workflow."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import time

from src.utils.core import set_seed, setup_logging, create_directories
from src.envs.utils import EnvironmentFactory, StateDiscretizer, EpisodeLogger, RewardShapingWrapper
from src.algorithms.q_learning import QLearningAgent


def run_complete_example():
    """Run complete transfer learning example."""
    
    # Setup
    set_seed(42)
    logger = setup_logging("INFO")
    create_directories(".")
    
    logger.info("Starting Transfer Reinforcement Learning Example")
    logger.info("=" * 60)
    
    # Step 1: Train source agent on CartPole-v1
    logger.info("Step 1: Training source agent on CartPole-v1")
    
    source_env = EnvironmentFactory.create_env("CartPole-v1", seed=42)
    source_agent = QLearningAgent(
        action_space_size=source_env.action_space.n,
        learning_rate=0.1,
        discount_factor=0.99,
        exploration_rate=1.0,
        exploration_decay=0.995,
        exploration_min=0.01
    )
    
    # Add episode logger
    source_env = EpisodeLogger(source_env, log_every=100)
    
    # Train source agent
    source_rewards = []
    source_lengths = []
    
    for episode in range(1000):
        state, _ = source_env.reset()
        state = tuple(state)  # Convert to hashable tuple
        
        total_reward = 0.0
        episode_length = 0
        done = False
        
        while not done:
            action = source_agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = source_env.step(action)
            done = terminated or truncated
            
            next_state = tuple(next_state)
            source_agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            episode_length += 1
        
        source_agent.decay_exploration()
        source_rewards.append(total_reward)
        source_lengths.append(episode_length)
    
    logger.info(f"Source training completed. Final exploration rate: {source_agent.exploration_rate:.3f}")
    logger.info(f"Average reward (last 100 episodes): {np.mean(source_rewards[-100:]):.2f}")
    
    # Save source agent
    source_agent.save("checkpoints/source_agent.pth")
    logger.info("Source agent saved to checkpoints/source_agent.pth")
    
    # Step 2: Train baseline agent on MountainCar-v0 (from scratch)
    logger.info("\nStep 2: Training baseline agent on MountainCar-v0 (from scratch)")
    
    target_env = EnvironmentFactory.create_env("MountainCar-v0", seed=42)
    
    # Add wrappers
    target_env = EpisodeLogger(target_env, log_every=100)
    target_env = RewardShapingWrapper(target_env, reward_scale=1.0, reward_shift=0.0)
    
    # State discretizer for continuous state space
    state_discretizer = StateDiscretizer(target_env.observation_space, n_bins=20)
    
    baseline_agent = QLearningAgent(
        action_space_size=target_env.action_space.n,
        learning_rate=0.1,
        discount_factor=0.99,
        exploration_rate=1.0,
        exploration_decay=0.995,
        exploration_min=0.01
    )
    
    # Train baseline agent
    baseline_rewards = []
    baseline_lengths = []
    
    for episode in range(1000):
        state, _ = target_env.reset()
        state = state_discretizer.discretize(state)
        
        total_reward = 0.0
        episode_length = 0
        done = False
        
        while not done:
            action = baseline_agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = target_env.step(action)
            done = terminated or truncated
            
            next_state = state_discretizer.discretize(next_state)
            baseline_agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            episode_length += 1
        
        baseline_agent.decay_exploration()
        baseline_rewards.append(total_reward)
        baseline_lengths.append(episode_length)
    
    logger.info(f"Baseline training completed. Final exploration rate: {baseline_agent.exploration_rate:.3f}")
    logger.info(f"Average reward (last 100 episodes): {np.mean(baseline_rewards[-100:]):.2f}")
    
    # Save baseline agent
    baseline_agent.save("checkpoints/baseline_agent.pth")
    logger.info("Baseline agent saved to checkpoints/baseline_agent.pth")
    
    # Step 3: Train transfer agent on MountainCar-v0 (with transferred knowledge)
    logger.info("\nStep 3: Training transfer agent on MountainCar-v0 (with transferred knowledge)")
    
    transfer_agent = QLearningAgent(
        action_space_size=target_env.action_space.n,
        learning_rate=0.1,
        discount_factor=0.99,
        exploration_rate=0.5,  # Lower initial exploration due to transfer
        exploration_decay=0.995,
        exploration_min=0.01
    )
    
    # Transfer knowledge from source agent
    transfer_agent.transfer_knowledge(source_agent, "full")
    logger.info("Knowledge transferred from source agent")
    
    # Train transfer agent
    transfer_rewards = []
    transfer_lengths = []
    
    for episode in range(1000):
        state, _ = target_env.reset()
        state = state_discretizer.discretize(state)
        
        total_reward = 0.0
        episode_length = 0
        done = False
        
        while not done:
            action = transfer_agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = target_env.step(action)
            done = terminated or truncated
            
            next_state = state_discretizer.discretize(next_state)
            transfer_agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            episode_length += 1
        
        transfer_agent.decay_exploration()
        transfer_rewards.append(total_reward)
        transfer_lengths.append(episode_length)
    
    logger.info(f"Transfer training completed. Final exploration rate: {transfer_agent.exploration_rate:.3f}")
    logger.info(f"Average reward (last 100 episodes): {np.mean(transfer_rewards[-100:]):.2f}")
    
    # Save transfer agent
    transfer_agent.save("checkpoints/transfer_agent.pth")
    logger.info("Transfer agent saved to checkpoints/transfer_agent.pth")
    
    # Step 4: Evaluation
    logger.info("\nStep 4: Evaluating agents")
    
    def evaluate_agent(agent, n_episodes=100):
        """Evaluate agent performance."""
        eval_rewards = []
        eval_lengths = []
        successes = 0
        
        for _ in range(n_episodes):
            state, _ = target_env.reset()
            state = state_discretizer.discretize(state)
            
            total_reward = 0.0
            episode_length = 0
            done = False
            
            while not done:
                action = agent.select_action(state, training=False)
                next_state, reward, terminated, truncated, _ = target_env.step(action)
                done = terminated or truncated
                
                state = state_discretizer.discretize(next_state)
                total_reward += reward
                episode_length += 1
            
            eval_rewards.append(total_reward)
            eval_lengths.append(episode_length)
            
            # Success criteria for MountainCar-v0
            if total_reward > -200:
                successes += 1
        
        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'success_rate': successes / n_episodes,
            'rewards': eval_rewards,
            'lengths': eval_lengths
        }
    
    # Evaluate both agents
    baseline_eval = evaluate_agent(baseline_agent)
    transfer_eval = evaluate_agent(transfer_agent)
    
    # Calculate transfer efficiency
    transfer_efficiency = transfer_eval['mean_reward'] / baseline_eval['mean_reward']
    
    logger.info("Evaluation Results:")
    logger.info(f"Baseline Agent:")
    logger.info(f"  Average Reward: {baseline_eval['mean_reward']:.2f} ± {baseline_eval['std_reward']:.2f}")
    logger.info(f"  Success Rate: {baseline_eval['success_rate']:.2%}")
    
    logger.info(f"Transfer Agent:")
    logger.info(f"  Average Reward: {transfer_eval['mean_reward']:.2f} ± {transfer_eval['std_reward']:.2f}")
    logger.info(f"  Success Rate: {transfer_eval['success_rate']:.2%}")
    
    logger.info(f"Transfer Efficiency: {transfer_efficiency:.2f}")
    
    if transfer_efficiency > 1.0:
        logger.info("Transfer learning improved performance!")
    elif transfer_efficiency > 0.8:
        logger.info("Transfer learning provided modest improvement.")
    else:
        logger.info("Transfer learning did not improve performance.")
    
    # Step 5: Generate visualizations
    logger.info("\nStep 5: Generating visualizations")
    
    # Create plots directory
    plots_dir = Path("assets/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Training curves comparison
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(source_rewards)
    plt.title("Source Agent (CartPole-v1)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    
    plt.subplot(2, 3, 2)
    plt.plot(baseline_rewards, label='Baseline', alpha=0.7)
    plt.plot(transfer_rewards, label='Transfer', alpha=0.7)
    plt.title("Target Agent (MountainCar-v0)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 3)
    # Rolling averages
    window = 50
    baseline_rolling = np.convolve(baseline_rewards, np.ones(window)/window, mode='valid')
    transfer_rolling = np.convolve(transfer_rewards, np.ones(window)/window, mode='valid')
    
    plt.plot(range(window-1, len(baseline_rewards)), baseline_rolling, label='Baseline')
    plt.plot(range(window-1, len(transfer_rewards)), transfer_rolling, label='Transfer')
    plt.title(f"Rolling Average Rewards (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 4)
    # Reward distributions
    plt.hist(baseline_eval['rewards'], alpha=0.7, label='Baseline', bins=20)
    plt.hist(transfer_eval['rewards'], alpha=0.7, label='Transfer', bins=20)
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.title("Evaluation Reward Distribution")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 5)
    # Success rates
    categories = ['Baseline', 'Transfer']
    success_rates = [baseline_eval['success_rate'], transfer_eval['success_rate']]
    plt.bar(categories, success_rates)
    plt.ylabel("Success Rate")
    plt.title("Success Rate Comparison")
    plt.grid(True)
    
    plt.subplot(2, 3, 6)
    # Transfer efficiency
    plt.bar(['Transfer Efficiency'], [transfer_efficiency])
    plt.ylabel("Transfer Efficiency")
    plt.title("Transfer Efficiency")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "complete_example_results.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Plots saved to {plots_dir}")
    
    # Step 6: Generate summary report
    logger.info("\nStep 6: Generating summary report")
    
    report = f"""
# Transfer Reinforcement Learning Example Results

## Experiment Overview
- Source Task: CartPole-v1 (pole balancing)
- Target Task: MountainCar-v0 (car climbing)
- Training Episodes: 1000 per agent
- Evaluation Episodes: 100 per agent

## Results Summary

### Source Agent (CartPole-v1)
- Final Average Reward: {np.mean(source_rewards[-100:]):.2f}
- Final Exploration Rate: {source_agent.exploration_rate:.3f}

### Baseline Agent (MountainCar-v0, from scratch)
- Training Average Reward: {np.mean(baseline_rewards[-100:]):.2f}
- Evaluation Average Reward: {baseline_eval['mean_reward']:.2f} ± {baseline_eval['std_reward']:.2f}
- Success Rate: {baseline_eval['success_rate']:.2%}

### Transfer Agent (MountainCar-v0, with transfer)
- Training Average Reward: {np.mean(transfer_rewards[-100:]):.2f}
- Evaluation Average Reward: {transfer_eval['mean_reward']:.2f} ± {transfer_eval['std_reward']:.2f}
- Success Rate: {transfer_eval['success_rate']:.2%}

### Transfer Efficiency
- Transfer Efficiency: {transfer_efficiency:.2f}
- Performance Improvement: {((transfer_efficiency - 1.0) * 100):.1f}%

## Interpretation
"""
    
    if transfer_efficiency > 1.0:
        report += "Transfer learning successfully improved performance on the target task.\n"
    elif transfer_efficiency > 0.8:
        report += "Transfer learning provided modest improvement on the target task.\n"
    else:
        report += "Transfer learning did not improve performance on the target task.\n"
    
    report += f"""
## Key Insights
1. The source task (CartPole-v1) was successfully learned with an average reward of {np.mean(source_rewards[-100:]):.2f}
2. The baseline agent achieved {baseline_eval['success_rate']:.2%} success rate on MountainCar-v0
3. The transfer agent achieved {transfer_eval['success_rate']:.2%} success rate on MountainCar-v0
4. Transfer efficiency of {transfer_efficiency:.2f} indicates {'positive' if transfer_efficiency > 1.0 else 'negative'} transfer

## Files Generated
- Source agent: checkpoints/source_agent.pth
- Baseline agent: checkpoints/baseline_agent.pth
- Transfer agent: checkpoints/transfer_agent.pth
- Results plot: assets/plots/complete_example_results.png
"""
    
    # Save report
    report_path = Path("assets/reports/example_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Report saved to {report_path}")
    
    # Cleanup
    source_env.close()
    target_env.close()
    
    logger.info("\n" + "=" * 60)
    logger.info("Transfer Reinforcement Learning Example Completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_complete_example()
