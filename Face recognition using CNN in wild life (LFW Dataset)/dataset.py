import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from PIL import Image

class LFWDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        
        # Convert NumPy array to PIL Image for torchvision transforms
        img = Image.fromarray((img * 255).astype('uint8'))
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

def get_lfw_dataloaders(min_faces_per_person=70, batch_size=32, test_size=0.2):
    """
    Downloads and prepares LFW dataset with data augmentation.
    """
    print("Loading LFW Dataset...")
    # Fetch LFW faces in RGB color mode
    lfw_people = fetch_lfw_people(min_faces_per_person=min_faces_per_person, color=True, resize=0.7)
    
    X = lfw_people.images  # Shape: (N, Height, Width, Channels)
    y = lfw_people.target
    target_names = lfw_people.target_names
    num_classes = len(target_names)
    
    print(f"Dataset Loaded: {X.shape[0]} samples across {num_classes} people.")
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    
    # Augmentations for Face Recognition
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = LFWDataset(X_train, y_train, transform=transform_train)
    test_dataset = LFWDataset(X_test, y_test, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, target_names, transform_test