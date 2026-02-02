"""Streamlit demo application for transfer reinforcement learning."""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path
import json
import yaml
from typing import Dict, Any, List, Tuple
import time

# Add src to path for imports
import sys
sys.path.append('src')

from src.utils.core import set_seed, get_device
from src.envs.utils import EnvironmentFactory, StateDiscretizer
from src.algorithms.q_learning import QLearningAgent


# Page configuration
st.set_page_config(
    page_title="Transfer Reinforcement Learning Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Safety disclaimer
st.markdown("""
<div class="warning-box">
    <h4>⚠️ Safety Disclaimer</h4>
    <p><strong>This demo is for research and educational purposes only.</strong> 
    It is NOT intended for production control of real-world systems. 
    The algorithms and implementations are experimental and should not be used 
    in safety-critical applications without extensive validation and testing.</p>
</div>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">Transfer Reinforcement Learning Demo</h1>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("Configuration")

# Environment selection
env_options = {
    "CartPole-v1": {
        "description": "Balance a pole on a cart",
        "source_task": True,
        "max_steps": 500
    },
    "MountainCar-v0": {
        "description": "Drive a car up a mountain",
        "source_task": False,
        "max_steps": 200
    }
}

selected_env = st.sidebar.selectbox(
    "Select Environment",
    list(env_options.keys()),
    help="Choose the environment to work with"
)

env_info = env_options[selected_env]
st.sidebar.write(f"**Description:** {env_info['description']}")
st.sidebar.write(f"**Max Steps:** {env_info['max_steps']}")

# Agent configuration
st.sidebar.subheader("Agent Settings")
learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
discount_factor = st.sidebar.slider("Discount Factor", 0.9, 0.999, 0.99, 0.001)
exploration_rate = st.sidebar.slider("Initial Exploration Rate", 0.1, 1.0, 0.5, 0.05)
exploration_decay = st.sidebar.slider("Exploration Decay", 0.99, 0.999, 0.995, 0.001)

# Transfer learning options
st.sidebar.subheader("Transfer Learning")
enable_transfer = st.sidebar.checkbox("Enable Transfer Learning", value=False)
transfer_type = st.sidebar.selectbox(
    "Transfer Type",
    ["full", "partial", "weights"],
    disabled=not enable_transfer
)

# Training parameters
st.sidebar.subheader("Training Parameters")
n_episodes = st.sidebar.slider("Number of Episodes", 100, 2000, 500, 50)
eval_frequency = st.sidebar.slider("Evaluation Frequency", 10, 100, 50, 10)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["Training", "Evaluation", "Visualization", "About"])

with tab1:
    st.header("Training")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Training Progress")
        
        if st.button("Start Training", type="primary"):
            # Initialize training
            with st.spinner("Initializing training..."):
                # Set random seed
                set_seed(42)
                
                # Create environment
                env = EnvironmentFactory.create_env(selected_env, seed=42)
                
                # Create agent
                agent = QLearningAgent(
                    action_space_size=env.action_space.n,
                    learning_rate=learning_rate,
                    discount_factor=discount_factor,
                    exploration_rate=exploration_rate,
                    exploration_decay=exploration_decay,
                    use_neural_network=False
                )
                
                # State discretizer for continuous environments
                if hasattr(env.observation_space, 'shape'):
                    state_discretizer = StateDiscretizer(env.observation_space, n_bins=20)
                else:
                    state_discretizer = None
                
                def discretize_state(state):
                    if state_discretizer is not None:
                        return state_discretizer.discretize(state)
                    return tuple(state) if isinstance(state, (list, tuple)) else (state,)
                
                # Training loop
                progress_bar = st.progress(0)
                status_text = st.empty()
                chart_placeholder = st.empty()
                
                episode_rewards = []
                episode_lengths = []
                exploration_rates = []
                
                for episode in range(n_episodes):
                    # Train one episode
                    state, _ = env.reset()
                    state = discretize_state(state)
                    
                    total_reward = 0.0
                    episode_length = 0
                    done = False
                    
                    while not done:
                        action = agent.select_action(state, training=True)
                        next_state, reward, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated
                        
                        next_state = discretize_state(next_state)
                        agent.update(state, action, reward, next_state, done)
                        
                        state = next_state
                        total_reward += reward
                        episode_length += 1
                    
                    # Decay exploration
                    agent.decay_exploration()
                    
                    # Store statistics
                    episode_rewards.append(total_reward)
                    episode_lengths.append(episode_length)
                    exploration_rates.append(agent.exploration_rate)
                    
                    # Update progress
                    progress = (episode + 1) / n_episodes
                    progress_bar.progress(progress)
                    status_text.text(f"Episode {episode + 1}/{n_episodes} - Reward: {total_reward:.2f}")
                    
                    # Update chart every 10 episodes
                    if (episode + 1) % 10 == 0:
                        df = pd.DataFrame({
                            'Episode': range(1, episode + 2),
                            'Reward': episode_rewards,
                            'Length': episode_lengths,
                            'Exploration Rate': exploration_rates
                        })
                        
                        fig = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=('Episode Rewards', 'Episode Lengths', 'Exploration Rate', 'Reward Distribution'),
                            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                                   [{"secondary_y": False}, {"secondary_y": False}]]
                        )
                        
                        fig.add_trace(
                            go.Scatter(x=df['Episode'], y=df['Reward'], name='Reward'),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(x=df['Episode'], y=df['Length'], name='Length'),
                            row=1, col=2
                        )
                        
                        fig.add_trace(
                            go.Scatter(x=df['Episode'], y=df['Exploration Rate'], name='Exploration Rate'),
                            row=2, col=1
                        )
                        
                        fig.add_trace(
                            go.Histogram(x=df['Reward'], name='Reward Distribution'),
                            row=2, col=2
                        )
                        
                        fig.update_layout(height=600, showlegend=False)
                        chart_placeholder.plotly_chart(fig, use_container_width=True)
                
                st.success("Training completed!")
                
                # Store results in session state
                st.session_state.training_results = {
                    'episode_rewards': episode_rewards,
                    'episode_lengths': episode_lengths,
                    'exploration_rates': exploration_rates,
                    'agent': agent,
                    'env_name': selected_env
                }
    
    with col2:
        st.subheader("Training Statistics")
        
        if 'training_results' in st.session_state:
            results = st.session_state.training_results
            
            # Calculate statistics
            avg_reward = np.mean(results['episode_rewards'])
            std_reward = np.std(results['episode_rewards'])
            avg_length = np.mean(results['episode_lengths'])
            final_exploration = results['exploration_rates'][-1]
            
            st.metric("Average Reward", f"{avg_reward:.2f} ± {std_reward:.2f}")
            st.metric("Average Length", f"{avg_length:.2f}")
            st.metric("Final Exploration Rate", f"{final_exploration:.3f}")
            
            # Success rate
            if selected_env == "CartPole-v1":
                success_rate = np.mean([l >= 500 for l in results['episode_lengths']])
                st.metric("Success Rate", f"{success_rate:.2%}")
            elif selected_env == "MountainCar-v0":
                success_rate = np.mean([r > -200 for r in results['episode_rewards']])
                st.metric("Success Rate", f"{success_rate:.2%}")

with tab2:
    st.header("Evaluation")
    
    if 'training_results' in st.session_state:
        st.subheader("Agent Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Run Evaluation", type="primary"):
                with st.spinner("Running evaluation..."):
                    agent = st.session_state.training_results['agent']
                    env_name = st.session_state.training_results['env_name']
                    
                    # Create evaluation environment
                    eval_env = EnvironmentFactory.create_env(env_name, seed=42)
                    
                    # State discretizer
                    if hasattr(eval_env.observation_space, 'shape'):
                        state_discretizer = StateDiscretizer(eval_env.observation_space, n_bins=20)
                    else:
                        state_discretizer = None
                    
                    def discretize_state(state):
                        if state_discretizer is not None:
                            return state_discretizer.discretize(state)
                        return tuple(state) if isinstance(state, (list, tuple)) else (state,)
                    
                    # Run evaluation episodes
                    eval_rewards = []
                    eval_lengths = []
                    
                    for _ in range(100):  # 100 evaluation episodes
                        state, _ = eval_env.reset()
                        state = discretize_state(state)
                        
                        total_reward = 0.0
                        episode_length = 0
                        done = False
                        
                        while not done:
                            action = agent.select_action(state, training=False)
                            next_state, reward, terminated, truncated, _ = eval_env.step(action)
                            done = terminated or truncated
                            
                            state = discretize_state(next_state)
                            total_reward += reward
                            episode_length += 1
                        
                        eval_rewards.append(total_reward)
                        eval_lengths.append(episode_length)
                    
                    # Store evaluation results
                    st.session_state.eval_results = {
                        'rewards': eval_rewards,
                        'lengths': eval_lengths
                    }
        
        with col2:
            if 'eval_results' in st.session_state:
                eval_results = st.session_state.eval_results
                
                avg_eval_reward = np.mean(eval_results['rewards'])
                std_eval_reward = np.std(eval_results['rewards'])
                avg_eval_length = np.mean(eval_results['lengths'])
                
                st.metric("Evaluation Reward", f"{avg_eval_reward:.2f} ± {std_eval_reward:.2f}")
                st.metric("Evaluation Length", f"{avg_eval_length:.2f}")
                
                # Success rate
                if env_name == "CartPole-v1":
                    success_rate = np.mean([l >= 500 for l in eval_results['lengths']])
                elif env_name == "MountainCar-v0":
                    success_rate = np.mean([r > -200 for r in eval_results['rewards']])
                
                st.metric("Success Rate", f"{success_rate:.2%}")
    
    else:
        st.info("Please train an agent first to enable evaluation.")

with tab3:
    st.header("Visualization")
    
    if 'training_results' in st.session_state:
        results = st.session_state.training_results
        
        # Create comprehensive visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Rewards', 'Episode Lengths', 'Exploration Rate', 'Reward Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        episodes = range(1, len(results['episode_rewards']) + 1)
        
        # Training rewards
        fig.add_trace(
            go.Scatter(x=episodes, y=results['episode_rewards'], name='Reward', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Episode lengths
        fig.add_trace(
            go.Scatter(x=episodes, y=results['episode_lengths'], name='Length', line=dict(color='green')),
            row=1, col=2
        )
        
        # Exploration rate
        fig.add_trace(
            go.Scatter(x=episodes, y=results['exploration_rates'], name='Exploration Rate', line=dict(color='red')),
            row=2, col=1
        )
        
        # Reward distribution
        fig.add_trace(
            go.Histogram(x=results['episode_rewards'], name='Reward Distribution', nbinsx=30),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False, title_text="Training Analysis")
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional analysis
        st.subheader("Performance Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Rolling average
            window = 50
            if len(results['episode_rewards']) >= window:
                rolling_avg = np.convolve(results['episode_rewards'], np.ones(window)/window, mode='valid')
                fig_rolling = go.Figure()
                fig_rolling.add_trace(go.Scatter(
                    x=list(range(window-1, len(results['episode_rewards']))),
                    y=rolling_avg,
                    name=f'Rolling Average (window={window})'
                ))
                fig_rolling.update_layout(title="Rolling Average Rewards", height=400)
                st.plotly_chart(fig_rolling, use_container_width=True)
        
        with col2:
            # Learning curve
            fig_learning = go.Figure()
            fig_learning.add_trace(go.Scatter(
                x=episodes,
                y=results['episode_rewards'],
                mode='markers',
                marker=dict(size=3, opacity=0.6),
                name='Episode Rewards'
            ))
            fig_learning.update_layout(title="Learning Curve", height=400)
            st.plotly_chart(fig_learning, use_container_width=True)
        
        with col3:
            # Statistics summary
            st.subheader("Summary Statistics")
            st.write(f"**Total Episodes:** {len(results['episode_rewards'])}")
            st.write(f"**Average Reward:** {np.mean(results['episode_rewards']):.2f}")
            st.write(f"**Best Reward:** {np.max(results['episode_rewards']):.2f}")
            st.write(f"**Worst Reward:** {np.min(results['episode_rewards']):.2f}")
            st.write(f"**Final Exploration:** {results['exploration_rates'][-1]:.3f}")
    
    else:
        st.info("Please train an agent first to see visualizations.")

with tab4:
    st.header("About Transfer Reinforcement Learning")
    
    st.markdown("""
    ## What is Transfer Reinforcement Learning?
    
    Transfer Reinforcement Learning (TRL) is a subfield of machine learning that focuses on 
    transferring knowledge learned from one task (source task) to improve the learning 
    process in a different but related task (target task). The goal is to help the agent 
    generalize knowledge across tasks without starting from scratch each time.
    
    ## Key Concepts
    
    ### 1. Source Task
    - The task from which knowledge is transferred
    - Usually well-studied and has sufficient training data
    - Example: CartPole-v1 (balancing a pole)
    
    ### 2. Target Task
    - The task where transferred knowledge is applied
    - Usually more complex or has limited training data
    - Example: MountainCar-v0 (driving up a mountain)
    
    ### 3. Transfer Methods
    - **Full Transfer**: Complete Q-table or neural network weights
    - **Partial Transfer**: Only common states or layers
    - **Weight Transfer**: Transfer only specific network layers
    
    ## Benefits of Transfer Learning
    
    1. **Faster Learning**: Reduced time to reach good performance
    2. **Sample Efficiency**: Fewer training episodes needed
    3. **Better Generalization**: Improved performance on related tasks
    4. **Knowledge Reuse**: Leverage existing learned policies
    
    ## Applications
    
    - **Robotics**: Transfer manipulation skills between different robots
    - **Gaming**: Transfer strategies between similar games
    - **Autonomous Vehicles**: Transfer driving policies across environments
    - **Recommendation Systems**: Transfer user preferences across domains
    
    ## Technical Implementation
    
    This demo implements Q-Learning with transfer capabilities:
    
    - **Tabular Q-Learning**: For discrete state spaces
    - **Neural Q-Networks**: For continuous state spaces
    - **State Discretization**: Convert continuous states to discrete
    - **Transfer Mechanisms**: Multiple transfer strategies
    
    ## Research Areas
    
    - **Domain Adaptation**: Adapting to different environments
    - **Multi-task Learning**: Learning multiple tasks simultaneously
    - **Meta-Learning**: Learning to learn efficiently
    - **Curriculum Learning**: Progressive task difficulty
    
    ## Limitations and Challenges
    
    - **Negative Transfer**: When transfer hurts performance
    - **Domain Mismatch**: When source and target tasks are too different
    - **Catastrophic Forgetting**: Losing previously learned knowledge
    - **Transfer Evaluation**: Measuring transfer effectiveness
    
    ## Further Reading
    
    - [Transfer Learning in Reinforcement Learning](https://arxiv.org/abs/1904.10090)
    - [A Survey of Transfer Learning](https://arxiv.org/abs/1411.1792)
    - [Deep Reinforcement Learning: An Overview](https://arxiv.org/abs/1701.07274)
    """)
    
    st.subheader("Demo Features")
    
    st.markdown("""
    This demo provides:
    
    - **Interactive Training**: Train agents with customizable parameters
    - **Real-time Visualization**: Watch learning progress in real-time
    - **Transfer Learning**: Compare performance with and without transfer
    - **Comprehensive Evaluation**: Statistical analysis of agent performance
    - **Educational Content**: Learn about transfer learning concepts
    
    ## How to Use This Demo
    
    1. **Configure Parameters**: Adjust agent and training settings in the sidebar
    2. **Start Training**: Click "Start Training" to begin the learning process
    3. **Monitor Progress**: Watch real-time updates of training metrics
    4. **Evaluate Performance**: Run evaluation to assess agent capabilities
    5. **Analyze Results**: Use visualizations to understand learning patterns
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Transfer Reinforcement Learning Demo | Educational and Research Purposes Only</p>
    <p>Built with Streamlit, PyTorch, and Gymnasium</p>
</div>
""", unsafe_allow_html=True)
