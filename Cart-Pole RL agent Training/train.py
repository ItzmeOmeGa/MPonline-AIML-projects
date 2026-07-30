import numpy as np
import torch
import gymnasium as gym
from agent import DQNAgent
from utils import plot_rewards

def train():
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size=state_size, action_size=action_size, lr=1e-3, gamma=0.99)

    num_episodes = 500
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01

    rewards_history = []
    
    print("Starting CartPole Agent Training...")
    
    for episode in range(1, num_episodes + 1):
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

        # Decaying exploration rate
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_history.append(episode_reward)

        # Update Target Network Periodically
        if episode % agent.target_update_freq == 0:
            agent.update_target_network()

        # Evaluate progress
        avg_reward = np.mean(rewards_history[-20:])
        print(f"Episode [{episode:03d}/{num_episodes}] | Total Reward: {episode_reward:3.0f} | "
              f"20-Ep Moving Avg: {avg_reward:5.1f} | Epsilon: {epsilon:.3f}")

        # Stop training early if solved (Avg Reward >= 490 over 20 consecutive episodes)
        if avg_reward >= 490:
            print(f"\nEnvironment Solved in {episode} episodes!")
            break

    # Save trained checkpoint
    torch.save(agent.q_network.state_dict(), 'cartpole_dqn.pth')
    print("Model saved to cartpole_dqn.pth")
    
    env.close()
    plot_rewards(rewards_history)

if __name__ == '__main__':
    train()