import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_mri_dataloaders
from model import CancerDetectionCNN
from utils import plot_learning_curves, evaluate_mri_model

def main():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset path (Ensure directory contains subfolders like: data/tumor, data/no_tumor)
    DATA_DIR = './data/mri_scans' 

    if not os.path.exists(DATA_DIR):
        print(f"Error: Path '{DATA_DIR}' not found. Please create folder structure with MRI image categories.")
        return

    # Load Data Loaders
    train_loader, val_loader, class_names, _ = get_mri_dataloaders(DATA_DIR, batch_size=32)
    num_classes = len(class_names)
    print(f"Loaded Classes: {class_names}")

    # Model Initialization
    model = CancerDetectionCNN(num_classes=num_classes).to(device)
    
    # Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    epochs = 20
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')

    print("Beginning Training...")
    for epoch in range(epochs):
        # --- Training ---
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
        
        # --- Validation ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
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
        
        # Checkpoint Saving
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save({
                'model_state': model.state_dict(),
                'classes': class_names
            }, 'best_mri_cancer_model.pth')
            
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        print(f"Epoch [{epoch+1:02d}/{epochs}] | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")

    # Load Best Model Weights and Plot Analysis
    checkpoint = torch.load('best_mri_cancer_model.pth')
    model.load_state_dict(checkpoint['model_state'])
    plot_learning_curves(history)
    evaluate_mri_model(model, val_loader, class_names, device)

if __name__ == '__main__':
    main()