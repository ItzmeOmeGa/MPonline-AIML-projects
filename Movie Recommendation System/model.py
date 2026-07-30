import torch
import torch.nn as nn

class CollaborativeFilteringNN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=50):
        super(CollaborativeFilteringNN, self).__init__()
        
        # User & Item Latent Factor Embeddings
        self.user_embed = nn.Embedding(num_users, embedding_dim)
        self.item_embed = nn.Embedding(num_items, embedding_dim)
        
        # User & Item Biases
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        
        # Dense interaction layers
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        # Global bias initializer
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_ids, item_ids):
        # Embeddings lookups
        u_emb = self.user_embed(user_ids)
        i_emb = self.item_embed(item_ids)
        
        u_b = self.user_bias(user_ids).squeeze(1)
        i_b = self.item_bias(item_ids).squeeze(1)
        
        # Concatenate latent vectors
        x = torch.cat([u_emb, i_emb], dim=1)
        out = self.fc_layers(x).squeeze(1)
        
        # Rating = Deep Interaction + User Bias + Item Bias
        prediction = out + u_b + i_b
        return prediction