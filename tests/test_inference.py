from PIL import Image
from pathlib import Path
import torch
from app_pages.src.inference import load_model, predict_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
project_root = Path('.')
model_path = project_root / 'app_pages' / 'src' / 'models' / 'sakuramodel_resnet18.pth'

model, class_names, image_size = load_model(model_path, device)

test_image_dir = project_root / 'data' / 'source_images' / 'healthy'
test_image_path = next(test_image_dir.glob('*.*'))
image = Image.open(test_image_path)
pred_class, conf = predict_image(model, image, class_names, device, image_size)
print(pred_class, conf)