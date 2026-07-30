import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(rewards, window=20):
    """Plots raw episode rewards along with a rolling mean curve."""
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.4, label='Episode Reward', color='skyblue')
    
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window - 1, len(rewards)), moving_avg, color='navy', label=f'{window}-Episode Moving Avg')
        
    plt.axhline(y=500, color='r', linestyle='--', label='Target Reward (500)')
    plt.title('CartPole-v1 RL Agent Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()