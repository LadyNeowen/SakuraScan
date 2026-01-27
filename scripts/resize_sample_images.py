"""
Resize and optimize sample images for Heroku deployment.

This script modifies images in-place under data_sample/source_images.
It resizes imahges so the longest side is MAX_SIZE px and saves JPEGs
with optimization to reduce file size.
"""

from pathlib import Path
from PIL import Image

MAX_SIZE = 300  # Max size for the longest side
JPEG_QUALITY = 75  # Quality for JPEG compression
VALID_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff']


def resize_in_place(image_path: Path) -> None:
    """
    Resize and optimize the image at the given path in-place.
    """
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        
        
        w, h = img.size
        max_side = max(w, h)
        
        if max_side > MAX_SIZE:
            new_size = (int(w * scale), int(h * scale))
            img = img.resize(new_size, Image.LANCZOS)
            
            img.save(
                image_path.with_suffix('.jpg'),
                format='JPEG',
                quality=JPEG_QUALITY,
                optimize=True,
                progressive=True,
            )
            
    if image_path.suffix.lower() != '.jpg':
        image_path.unlink(missing_ok=True)
        
def main() -> None:
    """
    Resize all images under data_sample/source_images.
    """
    root = Path('data_sample') / 'source_images'
    if not root.exists():
        raise SystemExit('data_sample/source_images not found.')
    
    image_paths = [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]

    for p in image_paths:
        resize_in_place(p)
        
    print(f'Processed {len(image_paths)} images.')
    
if __name__ == '__main__':
    main()    