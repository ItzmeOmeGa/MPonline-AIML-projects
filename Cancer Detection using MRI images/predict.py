import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import torch
import numpy as np

def plot_learning_curves(history):
    """Plots training and validation loss & accuracy."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0].plot(history['val_loss'], label='Val Loss', color='red')
    axes[0].set_title('Loss Curves')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    axes[1].plot(history['train_acc'], label='Train Acc', color='blue')
    axes[1].plot(history['val_acc'], label='Val Acc', color='red')
    axes[1].set_title('Accuracy Curves')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

def evaluate_mri_model(model, val_loader, class_names, device):
    """Evaluates Precision, Sensitivity/Recall, Specificity, and Confusion Matrix."""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            
    print("\n--- Clinical Classification Report ---")
    print(classification_report(all_targets, all_preds, target_names=class_names))
    
    # Calculate ROC-AUC for binary or multiclass
    try:
        if len(class_names) == 2:
            auc = roc_auc_score(all_targets, [p[1] for p in all_probs])
            print(f"ROC-AUC Score: {auc:.4f}")
        else:
            auc = roc_auc_score(all_targets, all_probs, multi_class='ovr')
            print(f"Multiclass ROC-AUC Score (OVR): {auc:.4f}")
    except Exception as e:
        print(f"ROC-AUC calculation skipped: {e}")

    # Plot Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('Ground Truth')
    plt.title('MRI Cancer Detection Confusion Matrix')
    plt.show()