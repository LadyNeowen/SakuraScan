"""
Smoke test for the Streamlit inference page (2_Leaf_Inference.py).
Verifies that the model and metadata can be loaded without errors.
"""

import importlib.util
import sys
from pathlib import Path

def main() -> None:
    """
    Load the inference page module and run the model loader
    """
    
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        
    page_path = project_root / 'app_pages' / '2_Leaf_Inference.py'
    
    if not page_path.exists():
        raise FileNotFoundError(f'Inference page not found: {page_path}')
    
    spec = importlib.util.spec_from_file_location('leaf_page', page_path)
    leaf_page = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(leaf_page)
    
    model, class_names, image_size, device = leaf_page.load_app_model()
    
    print('Model loaded successfully.')
    print(f'Number of classes: {len(class_names)}')
    print(f'Image size: {image_size}')
    print(f'Device: {device}')
    
if __name__ == '__main__':
    main()
  
    