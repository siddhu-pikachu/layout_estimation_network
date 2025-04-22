import os
import yaml
import torch
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import traceback
import sys

# Import project modules
from models.layout_estimation import LayoutEstimationNetwork


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Debug Layout Estimation Network')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (overrides config)')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='outputs/debug',
                        help='Directory to save outputs')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cuda or cpu)')
    return parser.parse_args()


def debug_demo():
    try:
        args = parse_args()
        print(f"Arguments parsed successfully: {args}")

        # Load config
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Config loaded successfully from {args.config}")

        # Setup device
        device = torch.device(args.device)
        print(f"Using device: {device}")

        # Initialize model - FIXED: removed dropout parameter
        print("Initializing model...")
        model = LayoutEstimationNetwork(
            backbone=config['model']['backbone'],
            pretrained=False
            # removed dropout parameter
        )
        model = model.to(device)
        print("Model initialized successfully")

        # Load checkpoint
        checkpoint_path = args.checkpoint if args.checkpoint else config['test']['checkpoint']
        print(f"Attempting to load checkpoint from: {checkpoint_path}")

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Checkpoint loaded successfully")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            print("Running with randomly initialized weights")

        # Preprocess image
        print(f"Loading image from: {args.image}")
        if not os.path.exists(args.image):
            print(f"Error: Image not found at {args.image}")
            return

        img_size = config['dataset']['img_size']
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        image = Image.open(args.image).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        image_tensor = image_tensor.to(device)
        print("Image preprocessed successfully")

        # Run inference
        print("Running inference...")
        model.eval()
        with torch.no_grad():
            layout_pred, camera_pred = model(image_tensor)
            print("Forward pass completed successfully")

            # Extract layout components
            pred_centroid, pred_size, pred_orientation = model.get_layout_components(layout_pred)
            print("Layout components extracted successfully")

        # Convert to numpy
        pred_centroid = pred_centroid.cpu().numpy()[0]
        pred_size = pred_size.cpu().numpy()[0]
        pred_orientation = pred_orientation.cpu().numpy()[0][0]

        # Print predictions
        print("\nPredicted Layout Parameters:")
        print(f"Centroid (x, y, z): {pred_centroid}")
        print(f"Size (width, height, depth): {pred_size}")
        print(f"Orientation: {pred_orientation} radians")

        print("\nPredicted Camera Parameters:")
        print(f"Pitch: {camera_pred[0, 0].item()} radians")
        print(f"Roll: {camera_pred[0, 1].item()} radians")

        print("Debug run completed successfully")

    except Exception as e:
        print(f"Exception occurred: {e}")
        print("Traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    debug_demo()