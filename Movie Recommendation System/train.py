import torch
import torch.nn as nn
import torch.optim as optim
from dataset import prepare_dataloaders
from model import CollaborativeFilteringNN
from utils import evaluate_metrics

def main():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Data
    train_loader, test_loader, metadata = prepare_dataloaders(batch_size=256)
    
    num_users = metadata["num_users"]
    num_items = metadata["num_items"]
    print(f"Dataset Stats: {num_users} Users | {num_items} Movies")

    # Initialize Model, Loss, and Optimizer
    model = CollaborativeFilteringNN(num_users=num_users, num_items=num_items, embedding_dim=64).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    epochs = 12
    best_rmse = float('inf')

    print("Beginning Recommendation Model Training...")
    for epoch in range(epochs):
        model.train()
        running_loss, total_count = 0.0, 0
        
        for u, i, r in train_loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            
            optimizer.zero_grad()
            preds = model(u, i)
            loss = criterion(preds, r)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * len(r)
            total_count += len(r)

        train_mse = running_loss / total_count
        val_rmse, val_mae = evaluate_metrics(model, test_loader, device)
        scheduler.step()

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save({
                'model_state': model.state_dict(),
                'metadata': metadata
            }, 'best_recommender.pth')

        print(f"Epoch [{epoch+1:02d}/{epochs}] | "
              f"Train Loss (MSE): {train_mse:.4f} | "
              f"Val RMSE: {val_rmse:.4f} | Val MAE: {val_mae:.4f}")

    print(f"\nTraining Complete. Best Validation RMSE: {best_rmse:.4f}")

if __name__ == '__main__':
    main()