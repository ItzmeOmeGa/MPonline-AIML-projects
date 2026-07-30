import sys
import torch
from PIL import Image
from dataset import CLASSES, get_dataloaders
from model import CIFAR10CNN

def predict_image(image_path, model_path='best_cifar10_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get test transformation from dataset helper
    _, _, transform_test = get_dataloaders()
    
    # Load Model Structure & Weights
    model = CIFAR10CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Process Image
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform_test(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    class_name = CLASSES[predicted_idx.item()]
    print(f"Prediction: {class_name} ({confidence.item() * 100:.2f}% confidence)")
    return class_name, confidence.item()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        predict_image(img_path)
    else:
        print("Usage: python predict.py <path_to_image>")