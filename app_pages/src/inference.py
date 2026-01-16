"""
Utilities for loading the SakuraScan model and making predictions.
"""

from pathlib import Path
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image



def load_model(model_path: Path, device: torch.device):
    """
    Load the trained ResNet18 model and its metadata.
    """
    
    checkpoint = torch.load(model_path, map_location=device)
    
    class_names = checkpoint['class_names']
    image_size = checkpoint['image_size']
    
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, class_names, image_size


def predict_image(model, image, class_names, device, image_size):
    """
    Predict the class of a single PIL image
    """
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
])
    
    tensor = transform(image.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = probs.max(1)
        
    return class_names[pred_idx.item()], conf.item()