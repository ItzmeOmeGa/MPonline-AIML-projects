import sys
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import FaceRecognitionCNN

def predict_face(image_path, checkpoint_path='best_lfw_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Weights and Target Class Names
    checkpoint = torch.load(checkpoint_path, map_location=device)
    target_names = checkpoint['target_names']
    
    model = FaceRecognitionCNN(num_classes=len(target_names)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Matching Normalization
    transform = transforms.Compose([
        transforms.Resize((87, 65)),  # Fits LFW native dimension scaling
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    person_name = target_names[predicted_idx.item()]
    print(f"Identified Person: {person_name} ({confidence.item() * 100:.2f}% confidence)")
    return person_name, confidence.item()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        predict_face(image_path)
    else:
        print("Usage: python predict.py <path_to_face_image>")