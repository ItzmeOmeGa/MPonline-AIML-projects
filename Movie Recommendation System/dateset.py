import os
import zipfile
import urllib.request
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = "./data"

def download_and_extract_movielens():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    zip_path = os.path.join(DATA_DIR, "ml-100k.zip")
    extracted_folder = os.path.join(DATA_DIR, "ml-100k")
    
    if not os.path.exists(extracted_folder):
        print("Downloading MovieLens-100k dataset...")
        urllib.request.urlretrieve(DATA_URL, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("Dataset extracted successfully.")

class MovieLensDataset(Dataset):
    def __init__(self, user_ids, item_ids, ratings):
        self.user_ids = torch.tensor(user_ids, dtype=torch.long)
        self.item_ids = torch.tensor(item_ids, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]

def prepare_dataloaders(batch_size=256, test_size=0.2):
    download_and_extract_movielens()
    
    # Load Ratings
    ratings_path = os.path.join(DATA_DIR, "ml-100k", "u.data")
    df_ratings = pd.read_csv(
        ratings_path, sep="\t", names=["user_id", "item_id", "rating", "timestamp"]
    )
    
    # Load Movie Metadata (Items)
    items_path = os.path.join(DATA_DIR, "ml-100k", "u.item")
    genre_cols = [
        "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
        "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
    movie_cols = ["item_id", "title", "release_date", "video_release_date", "IMDb_URL"] + genre_cols
    df_movies = pd.read_csv(items_path, sep="|", names=movie_cols, encoding="latin-1")

    # Encode raw IDs into contiguous range [0, N-1]
    user2idx = {id_: idx for idx, id_ in enumerate(df_ratings["user_id"].unique())}
    item2idx = {id_: idx for idx, id_ in enumerate(df_movies["item_id"].unique())}
    idx2item = {idx: id_ for id_, idx in item2idx.items()}

    # Map ratings
    df_ratings = df_ratings[df_ratings["item_id"].isin(item2idx.keys())].copy()
    df_ratings["user_idx"] = df_ratings["user_id"].map(user2idx)
    df_ratings["item_idx"] = df_ratings["item_id"].map(item2idx)

    train_df, test_df = train_test_split(df_ratings, test_size=test_size, random_state=42)

    train_ds = MovieLensDataset(train_df["user_idx"].values, train_df["item_idx"].values, train_df["rating"].values)
    test_ds = MovieLensDataset(test_df["user_idx"].values, test_df["item_idx"].values, test_df["rating"].values)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    metadata = {
        "num_users": len(user2idx),
        "num_items": len(item2idx),
        "user2idx": user2idx,
        "item2idx": item2idx,
        "idx2item": idx2item,
        "df_movies": df_movies,
        "genre_cols": genre_cols
    }

    return train_loader, test_loader, metadata