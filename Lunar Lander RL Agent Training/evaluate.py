import sys
import time
import torch
import gymnasium as gym
from model import QNetwork

def evaluate(model_path='lunar_lander_dqn.pth', episodes=5):
    env = gym.make('LunarLander-v3', render_mode='human')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model weights
    q_network = QNetwork(state_size, action_size).to(device)
    q_network.load_state_dict(torch.load(model_path, map_location=device))
    q_network.eval()

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action = torch.argmax(q_network(state_tensor), dim=1).item()
                
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            time.sleep(0.01)

        print(f"Evaluation Episode {ep}: Score = {total_reward:.2f}")

    env.close()

if __name__ == '__main__':
    model_file = sys.argv[1] if len(sys.argv) > 1 else 'lunar_lander_dqn.pth'
    evaluate(model_file)