import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def evaluate_metrics(model, test_loader, device):
    """Calculates Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE)."""
    model.eval()
    mse_loss = torch.nn.MSELoss(reduction='sum')
    mae_loss = torch.nn.L1Loss(reduction='sum')
    
    total_mse, total_mae, count = 0.0, 0.0, 0
    
    with torch.no_grad():
        for u, i, r in test_loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            preds = model(u, i)
            
            total_mse += mse_loss(preds, r).item()
            total_mae += mae_loss(preds, r).item()
            count += len(r)
            
    rmse = np.sqrt(total_mse / count)
    mae = total_mae / count
    return rmse, mae

def get_content_based_recommendations(preferred_genres, df_movies, genre_cols, top_n=5):
    """Fallback engine using Genre-Cosine similarity for Cold-Start scenarios."""
    user_vector = np.array([1 if genre in preferred_genres else 0 for genre in genre_cols]).reshape(1, -1)
    movie_vectors = df_movies[genre_cols].values
    
    similarities = cosine_similarity(user_vector, movie_vectors)[0]
    top_indices = similarities.argsort()[::-1][:top_n]
    
    return df_movies.iloc[top_indices][["item_id", "title"] + genre_cols]