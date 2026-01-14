from pathlib import Path
from PIL import Image
from typing import List, Tuple


class SakuraDataset:
    '''
    A simple dataset loader for the SakuraScan project.

    This class loads images from the given base directory.
    It assumes that the folder structure contains class subfolders,
    such as 'healthy' and 'powdery_mildew'.

    The dataset stores a list of (image_path, label_index) pairs.
    '''

    def __init__(self, base_dir: str = '../Data/source_images') -> None:
        '''
        Initialize the dataset by scanning all images and building a sample list.

        Args:
            base_dir: Path to the directory containing class subfolders.
        '''
        self.base_dir = Path(base_dir)

        if not self.base_dir.exists():
            raise FileNotFoundError(f'Base directory not found: {self.base_dir}')

        # Discover class folders (e.g. healthy, powdery_mildew)
        self.classes = sorted([folder.name for folder in self.base_dir.iterdir() if folder.is_dir()])

        if not self.classes:
            raise ValueError('No class folders found inside the base directory.')

        # Build list of (image_path, label_index)
        self.samples: List[Tuple[Path, int]] = []

        for index, class_name in enumerate(self.classes):
            class_folder = self.base_dir / class_name
            image_files = list(class_folder.rglob('*.*'))

            for img_path in image_files:
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((img_path, index))

        if not self.samples:
            raise ValueError('No images found in dataset.')

    def __len__(self) -> int:
        '''
        Return the number of samples in the dataset.
        '''
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        '''
        Load and return an image and its label index.

        Args:
            index: The sample index to load.

        Returns:
            A tuple (image, label_index).
        '''
        img_path, label_index = self.samples[index]
        image = Image.open(img_path).convert('RGB')
        return image, label_index
