import random
import torch
import torch.nn as nn
import torch.optim as optim
from model import QNetwork
from memory import ReplayBuffer

class DQNAgent:
    def __init__(self, state_size=4, action_size=2, lr=1e-3, gamma=0.99, 
                 buffer_size=10000, batch_size=64, target_update_freq=10):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Networks: Main Q-Network and Target Network
        self.q_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = ReplayBuffer(capacity=buffer_size)
        self.criterion = nn.MSELoss()

    def select_action(self, state, epsilon=0.0):
        """Select action using Epsilon-Greedy policy."""
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(state)
        self.q_network.train()
        
        return torch.argmax(action_values, dim=1).item()

    def learn(self):
        """Train main network using sample batch from Replay Memory."""
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states, actions = states.to(self.device), actions.to(self.device)
        rewards, next_states = rewards.to(self.device), next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q-values: Q(s, a)
        q_values = self.q_network(states).gather(1, actions)

        # Target Q-values: r + gamma * max_a Q_target(s', a) * (1 - done)
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

        # Compute Loss and Optimize
        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        """Copy weights from Main Network to Target Network."""
        self.target_network.load_state_dict(self.q_network.state_dict())