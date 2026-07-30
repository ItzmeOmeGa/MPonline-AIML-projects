import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataset import get_lfw_dataloaders
from model import FaceRecognitionCNN
from utils import plot_training_history, evaluate_model

def main():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load LFW Data
    train_loader, test_loader, target_names, _ = get_lfw_dataloaders(min_faces_per_person=70, batch_size=32)
    num_classes = len(target_names)

    # Initialize Network, Loss, Optimizer, and LR Scheduler
    model = FaceRecognitionCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    epochs = 25
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')

    print("Beginning Model Training...")
    for epoch in range(epochs):
        # Training Step
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_loss = running_loss / total
        train_acc = 100. * correct / total
        
        # Validation Step
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = 100. * val_correct / val_total
        
        scheduler.step(epoch_val_loss)
        
        # Save Best Model Weights
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'target_names': target_names
            }, 'best_lfw_model.pth')
            
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        print(f"Epoch [{epoch+1:02d}/{epochs}] | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")

    # Load best state and run full evaluation
    checkpoint = torch.load('best_lfw_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    plot_training_history(history)
    evaluate_model(model, test_loader, target_names, device)

if __name__ == '__main__':
    main()