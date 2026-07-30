import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_size=4, action_size=2, hidden_size=64):
        super(QNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, state):
        return self.network(state)