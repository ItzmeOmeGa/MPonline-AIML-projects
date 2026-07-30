import numpy as np
import torch
import gymnasium as gym
from agent import DQNAgent
from utils import plot_training_results

def train():
    # Supports 'LunarLander-v3' or 'LunarLander-v2'
    env = gym.make('LunarLander-v3')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size=state_size, action_size=action_size, lr=5e-4)

    max_episodes = 800
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01

    rewards_history = []
    
    print("Starting Lunar Lander Agent Training...")

    for episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.memory.push(state, action, reward, next_state, done)
            agent.learn()
            
            state = next_state
            episode_reward += reward

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_history.append(episode_reward)

        avg_reward = np.mean(rewards_history[-100:])
        
        print(f"Episode [{episode:03d}/{max_episodes}] | "
              f"Reward: {episode_reward:6.1f} | "
              f"100-Ep Moving Avg: {avg_reward:6.1f} | "
              f"Epsilon: {epsilon:.3f}")

        # LunarLander is officially considered solved when average score >= 200
        if avg_reward >= 200 and episode >= 100:
            print(f"\nEnvironment Solved in {episode} Episodes! Average Score: {avg_reward:.2f}")
            break

    # Save trained model weights
    torch.save(agent.q_network.state_dict(), 'lunar_lander_dqn.pth')
    print("Saved model checkpoint to 'lunar_lander_dqn.pth'")
    
    env.close()
    plot_training_results(rewards_history)

if __name__ == '__main__':
    train()