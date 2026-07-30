import matplotlib.pyplot as plt
import numpy as np

def plot_training_results(rewards, window=50):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, label='Raw Episode Reward', color='gray')
    
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window - 1, len(rewards)), moving_avg, color='darkgreen', label=f'{window}-Episode Average')
        
    plt.axhline(y=200, color='r', linestyle='--', label='Solved Benchmark (+200)')
    plt.title('LunarLander RL Agent Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()