import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2


class Pix3DLayoutDataset(Dataset):
    """
    Enhanced Pix3D dataset for layout estimation, aligned with Total3D approach.
    """

    def __init__(self, root_dir, split='train', transform=None, img_size=256, use_augmentation=True):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.use_augmentation = use_augmentation and split == 'train'

        # Load annotations
        self.annotations = self._load_annotations()

        # Split the data
        self._split_data()

        # Setup transformations
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        # Data augmentation transforms
        self.aug_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _load_annotations(self):
        """Load and process Pix3D annotations with focus on layout information"""
        annotation_file = os.path.join(self.root_dir, 'pix3d.json')

        if not os.path.exists(annotation_file):
            print(f"Warning: Annotation file {annotation_file} not found. Using dummy data.")
            return self._create_dummy_annotations(200)  # More samples for better training

        with open(annotation_file, 'r') as f:
            raw_annotations = json.load(f)

        processed_annotations = []
        for ann in raw_annotations:
            # Filter invalid annotations
            if not self._is_valid_annotation(ann):
                continue

            # Process layout annotation
            layout_info = self._process_layout_info(ann)
            if layout_info:
                processed_annotations.append(layout_info)

        return processed_annotations

    def _is_valid_annotation(self, ann):
        """Check if annotation has all required fields"""
        required_fields = ['img', 'room_layout', 'camera']
        if not all(field in ann for field in required_fields):
            return False

        # Check if image exists
        img_path = os.path.join(self.root_dir, ann['img'])
        if not os.path.exists(img_path):
            return False

        return True

    def _process_layout_info(self, ann):
        """Process layout and camera information from annotation"""
        room_layout = ann.get('room_layout', {})
        camera = ann.get('camera', {})

        # Extract 3D room layout parameters
        try:
            centroid = np.array(room_layout.get('centroid', [0, 0, 0]), dtype=np.float32)
            size = np.array(room_layout.get('size', [1, 1, 1]), dtype=np.float32)
            orientation = np.array([room_layout.get('orientation', 0)], dtype=np.float32)

            # Extract camera parameters
            pitch = np.array([camera.get('pitch', 0)], dtype=np.float32)
            roll = np.array([camera.get('roll', 0)], dtype=np.float32)

            # Ensure size is positive
            size = np.maximum(size, 0.1)  # Avoid zero or negative sizes

            return {
                'img_path': os.path.join(self.root_dir, ann['img']),
                'layout_centroid': centroid,
                'layout_size': size,
                'layout_orientation': orientation,
                'camera_pitch': pitch,
                'camera_roll': roll
            }
        except (ValueError, TypeError):
            return None

    def _split_data(self):
        """Split data into train/val/test sets based on split parameter"""
        # Shuffle data with fixed seed for reproducibility
        np.random.seed(42)
        indices = np.random.permutation(len(self.annotations))

        # Calculate split indices
        train_end = int(len(indices) * 0.8)
        val_end = int(len(indices) * 0.9)

        # Select appropriate indices
        if self.split == 'train':
            self.indices = indices[:train_end]
        elif self.split == 'val':
            self.indices = indices[train_end:val_end]
        else:  # test
            self.indices = indices[val_end:]

        # Filter annotations
        self.annotations = [self.annotations[i] for i in self.indices]

    def _create_dummy_annotations(self, num_samples):
        """Create realistic dummy data to test the model"""
        dummy_annotations = []
        np.random.seed(42)  # For reproducibility

        for i in range(num_samples):
            # Create realistic room layout parameters
            centroid = np.random.uniform(-1.0, 1.0, 3).astype(np.float32)
            size = np.random.uniform(1.0, 5.0, 3).astype(np.float32)
            orientation = np.array([np.random.uniform(0, 2 * np.pi)], dtype=np.float32)

            # Create realistic camera parameters
            pitch = np.array([np.random.uniform(-0.5, 0.5)], dtype=np.float32)
            roll = np.array([np.random.uniform(-0.3, 0.3)], dtype=np.float32)

            item = {
                'img_path': os.path.join(self.root_dir, f'dummy_image_{i}.jpg'),
                'layout_centroid': centroid,
                'layout_size': size,
                'layout_orientation': orientation,
                'camera_pitch': pitch,
                'camera_roll': roll
            }
            dummy_annotations.append(item)
        return dummy_annotations

    def __len__(self):
        """Return number of samples"""
        return len(self.annotations)

    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        annotation = self.annotations[idx]

        # Load image
        img_path = annotation['img_path']
        if not os.path.exists(img_path):
            # Create dummy image with room layout visualization
            image = self._create_dummy_layout_image(annotation)
        else:
            image = Image.open(img_path).convert('RGB')

        # Apply transformations
        if self.use_augmentation and np.random.random() < 0.5:
            image_tensor = self.aug_transform(image)
        else:
            image_tensor = self.transform(image)

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

        return image_tensor, layout_params, camera_params

    def _create_dummy_layout_image(self, annotation):
        """Create a visualization of the room layout for dummy data"""
        width, height = self.img_size, self.img_size
        img = np.ones((height, width, 3), dtype=np.uint8) * 200  # Light gray background

        # Extract layout parameters
        centroid = annotation['layout_centroid']
        size = annotation['layout_size']
        orientation = annotation['layout_orientation'][0]

        # Draw a simple top-down view of the room
        center_x, center_y = width // 2, height // 2
        room_width = int(size[0] * 30)
        room_depth = int(size[2] * 30)

        # Calculate room corners with orientation
        cos_theta = np.cos(orientation)
        sin_theta = np.sin(orientation)

        corners = [
            [-room_width // 2, -room_depth // 2],
            [room_width // 2, -room_depth // 2],
            [room_width // 2, room_depth // 2],
            [-room_width // 2, room_depth // 2]
        ]

        # Rotate corners
        rotated_corners = []
        for x, y in corners:
            x_rot = x * cos_theta - y * sin_theta
            y_rot = x * sin_theta + y * cos_theta
            rotated_corners.append([int(center_x + x_rot), int(center_y + y_rot)])

        # Draw room boundaries
        cv2.polylines(img, [np.array(rotated_corners)], True, (0, 0, 255), 2)

        # Draw coordinate axes
        cv2.line(img, (center_x, center_y), (center_x + 50, center_y), (255, 0, 0), 2)  # X-axis
        cv2.line(img, (center_x, center_y), (center_x, center_y - 50), (0, 255, 0), 2)  # Y-axis

        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f"Size: {size[0]:.1f}x{size[1]:.1f}x{size[2]:.1f}",
                    (10, 30), font, 0.5, (0, 0, 0), 1)
        cv2.putText(img, f"Orient: {orientation:.2f}",
                    (10, 60), font, 0.5, (0, 0, 0), 1)

        return Image.fromarray(img)


def get_data_loaders(config):
    """Create data loaders for train, validation, and test sets"""
    train_dataset = Pix3DLayoutDataset(
        root_dir=config['dataset']['root_dir'],
        split='train',
        img_size=config['dataset']['img_size'],
        use_augmentation=config['dataset'].get('data_augmentation', True)
    )

    val_dataset = Pix3DLayoutDataset(
        root_dir=config['dataset']['root_dir'],
        split='val',
        img_size=config['dataset']['img_size'],
        use_augmentation=False
    )

    test_dataset = Pix3DLayoutDataset(
        root_dir=config['dataset']['root_dir'],
        split='test',
        img_size=config['dataset']['img_size'],
        use_augmentation=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['train']['num_workers'],
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['test'].get('batch_size', 1),
        shuffle=False,
        num_workers=config['train']['num_workers']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['test'].get('batch_size', 1),
        shuffle=False,
        num_workers=config['train']['num_workers']
    )

    return train_loader, val_loader, test_loader