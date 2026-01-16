"""
Streamlit page for exploring the Sakura dataset.
"""

from pathlib import Path
import random
from PIL import Image
import streamlit as st

VALID_EXTENSIONS = ['.jpg', '.jpeg', '.png']

def list_images(folder: Path):
    """
    Return all valid image paths in the given folder.
    """
    
    if not folder.exists():
        return []
    return [p for p in folder.iterdir() if p.suffix.lower() in VALID_EXTENSIONS]


def run() -> None:
    """
    Render the dataset exploration page.
    """
    
    st.title('SakuraScan - Explore Dataset')
    st.write('Browse through sample images from the Sakura dataset.')
    
    project_root = Path(__file__).resolve().parents[1]
    base = project_root / 'data' / 'source_images'
    
    dirs = {
        'Healthy Leaves': base / 'healthy',
        'Powdery Mildew': base / 'powdery_mildew',
    }
    
    
    # Counts images
    counts = {cls: len(list_images(path)) for cls, path in dirs.items()}
    st.subheader('Images per class')
    for cls, n in counts.items():
        st.write(f'- **{cls}**: {n} images')
    
    if sum(counts.values()) == 0:
        st.error('No images found in the dataset folders.')
        return
    
    # Select class and show example
    st.subheader('Sample Images')
    selected = st.selectbox('Class', list(dirs.keys()))
    images = list_images(dirs[selected])
    if not images:
        st.warning(f'No images found for class {selected}')
        return
    
    mode = st.radio('Mode', ['Random', 'Pick index'], horizontal=True)
    
    if mode == 'Random':
        img_path = random.choice(images)
    else:
        idx = st.slider('Image index', 0, len(images) - 1)
        img_path = images[idx]
        
    image = Image.open(img_path).convert('RGB')
    st.image(image, caption=f'{selected} - {img_path.name}', use_column_width=True)
    

if __name__ == '__main__':
    run()
    