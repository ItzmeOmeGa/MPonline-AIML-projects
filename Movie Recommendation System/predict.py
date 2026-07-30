import sys
import torch
import pandas as pd
from model import CollaborativeFilteringNN
from utils import get_content_based_recommendations

def recommend_for_user(raw_user_id, top_n=5, model_path='best_recommender.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Checkpoint & Metadata
    checkpoint = torch.load(model_path, map_location=device)
    metadata = checkpoint['metadata']
    
    user2idx = metadata['user2idx']
    item2idx = metadata['item2idx']
    idx2item = metadata['idx2item']
    df_movies = metadata['df_movies']
    
    # Check if user exists (Collaborative) or is unknown (Cold-start Content-Based)
    if raw_user_id not in user2idx:
        print(f"\nUser ID {raw_user_id} not found in training dataset (Cold-Start).")
        print("Falling back to Content-Based Genre Filtering for ['Action', 'Sci-Fi']...")
        recs = get_content_based_recommendations(['Action', 'Sci-Fi'], df_movies, metadata['genre_cols'], top_n)
        print("\n--- Top Recommendations ---")
        print(recs[["item_id", "title"]].to_string(index=False))
        return

    user_idx = user2idx[raw_user_id]
    
    # Initialize and load model
    model = CollaborativeFilteringNN(metadata['num_users'], metadata['num_items'], embedding_dim=64).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # Score all movies for this user
    all_item_indices = torch.tensor(list(idx2item.keys()), dtype=torch.long).to(device)
    user_indices = torch.tensor([user_idx] * len(all_item_indices), dtype=torch.long).to(device)

    with torch.no_grad():
        predicted_ratings = model(user_indices, all_item_indices)

    # Rank top N
    top_scores, top_item_indices = torch.topk(predicted_ratings, top_n)
    
    results = []
    for score, item_idx in zip(top_scores.cpu().numpy(), top_item_indices.cpu().numpy()):
        raw_item_id = idx2item[item_idx]
        movie_title = df_movies[df_movies["item_id"] == raw_item_id]["title"].values[0]
        results.append({"Movie ID": raw_item_id, "Title": movie_title, "Predicted Rating": round(float(score), 2)})

    print(f"\n--- Top {top_n} Movie Recommendations for User {raw_user_id} ---")
    print(pd.DataFrame(results).to_string(index=False))

if __name__ == '__main__':
    user_id_input = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    recommend_for_user(user_id_input)