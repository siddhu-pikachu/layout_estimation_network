import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class Pix3DDataset(Dataset):
    """
    Dataset class for the Pix3D dataset focused on layout estimation.

    The Pix3D dataset contains images of furniture with 3D annotations.
    For layout estimation, we use the camera parameters and room layout
    information provided in the dataset.
    """

    def __init__(self, root_dir, split='train', transform=None, img_size=256):
        """
        Initialize the Pix3D dataset.

        Args:
            root_dir (str): Directory containing the Pix3D dataset
            split (str): Dataset split ('train', 'val', or 'test')
            transform (callable, optional): Optional transform to be applied on an image
            img_size (int): Size to resize images to
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size

        # Load dataset annotations
        self.annotations = self._load_annotations()

        # Filter annotations based on split
        if split == 'train':
            self.annotations = self.annotations[:int(len(self.annotations) * 0.8)]
        elif split == 'val':
            self.annotations = self.annotations[int(len(self.annotations) * 0.8):int(len(self.annotations) * 0.9)]
        elif split == 'test':
            self.annotations = self.annotations[int(len(self.annotations) * 0.9):]

        # Set up image transformations
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def _load_annotations(self):
        """
        Load annotations from the Pix3D dataset.

        Returns:
            list: List of dictionaries containing image paths, layout, and camera information
        """
        # In a real implementation, this would load the actual Pix3D metadata
        # For now, we'll create a placeholder that you can update with real data
        annotation_file = os.path.join(self.root_dir, 'pix3d.json')

        # Check if the annotation file exists
        if not os.path.exists(annotation_file):
            print(f"Warning: Annotation file {annotation_file} not found. Creating dummy data.")
            # Create dummy data for development
            return self._create_dummy_annotations(100)

        # Load annotations from file
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)

        # Process annotations to extract relevant information
        processed_annotations = []
        for ann in annotations:
            if 'img' not in ann or 'room_layout' not in ann:
                continue

            item = {
                'img_path': os.path.join(self.root_dir, ann['img']),
                'layout_centroid': np.array(ann.get('room_layout', {}).get('centroid', [0, 0, 0])),
                'layout_size': np.array(ann.get('room_layout', {}).get('size', [1, 1, 1])),
                'layout_orientation': np.array([ann.get('room_layout', {}).get('orientation', 0)]),
                'camera_pitch': np.array([ann.get('camera', {}).get('pitch', 0)]),
                'camera_roll': np.array([ann.get('camera', {}).get('roll', 0)])
            }
            processed_annotations.append(item)

        return processed_annotations

    def _create_dummy_annotations(self, num_samples):
        """
        Create dummy annotations for development.

        Args:
            num_samples (int): Number of dummy samples to create

        Returns:
            list: List of dictionaries with dummy annotations
        """
        dummy_annotations = []
        for i in range(num_samples):
            item = {
                'img_path': os.path.join(self.root_dir, f'dummy_image_{i}.jpg'),
                'layout_centroid': np.array([0.0, 0.0, 0.0]),
                'layout_size': np.array([3.0, 2.5, 4.0]),
                'layout_orientation': np.array([0.0]),
                'camera_pitch': np.array([0.3]),
                'camera_roll': np.array([0.0])
            }
            dummy_annotations.append(item)
        return dummy_annotations

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample

        Returns:
            tuple: (image, layout_params, camera_params)
                - image (torch.Tensor): Image tensor
                - layout_params (torch.Tensor): Layout parameters
                - camera_params (torch.Tensor): Camera parameters
        """
        annotation = self.annotations[idx]

        # Load image
        img_path = annotation['img_path']

        # For development with dummy data, create a blank image if file doesn't exist
        if not os.path.exists(img_path):
            image = Image.new('RGB', (self.img_size, self.img_size), color=(128, 128, 128))
        else:
            image = Image.open(img_path).convert('RGB')

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Prepare layout parameters
        layout_centroid = torch.tensor(annotation['layout_centroid'], dtype=torch.float32)
        layout_size = torch.tensor(annotation['layout_size'], dtype=torch.float32)
        layout_orientation = torch.tensor(annotation['layout_orientation'], dtype=torch.float32)

        # Combine layout parameters
        layout_params = torch.cat([layout_centroid, layout_size, layout_orientation], dim=0)

        # Prepare camera parameters
        camera_pitch = torch.tensor(annotation['camera_pitch'], dtype=torch.float32)
        camera_roll = torch.tensor(annotation['camera_roll'], dtype=torch.float32)

        # Combine camera parameters
        camera_params = torch.cat([camera_pitch, camera_roll], dim=0)

        return image, layout_params, camera_params


def get_dataloader(config, split='train'):
    """
    Create a dataloader for the specified split.

    Args:
        config (dict): Configuration parameters
        split (str): Dataset split ('train', 'val', or 'test')

    Returns:
        DataLoader: PyTorch dataloader
    """
    # Create dataset
    dataset = Pix3DDataset(
        root_dir=config['dataset']['root_dir'],
        split=split,
        img_size=config['dataset']['img_size']
    )

    # Create dataloader
    batch_size = config['train']['batch_size'] if split == 'train' else 1
    shuffle = True if split == 'train' else False
    num_workers = config['train']['num_workers']

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return dataloader