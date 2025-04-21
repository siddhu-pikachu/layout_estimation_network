import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

# Import project modules
from models.layout_estimation import LayoutEstimationNetwork
from data.dataset import get_dataloader
from utils.visualization import visualize_layout


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Test Layout Estimation Network')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (overrides config)')
    parser.add_argument('--device', type=str, default='mps' if torch.backends.mps.is_available() else 'cpu',
                        help='Device to use (mps, cuda, or cpu)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save outputs (overrides config)')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_metrics(pred_layout, gt_layout, pred_camera, gt_camera):
    """
    Compute evaluation metrics for layout estimation.

    Args:
        pred_layout (torch.Tensor): Predicted layout parameters
        gt_layout (torch.Tensor): Ground truth layout parameters
        pred_camera (torch.Tensor): Predicted camera parameters
        gt_camera (torch.Tensor): Ground truth camera parameters

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Convert to numpy for easier manipulation
    pred_layout = pred_layout.cpu().numpy()
    gt_layout = gt_layout.cpu().numpy()
    pred_camera = pred_camera.cpu().numpy()
    gt_camera = gt_camera.cpu().numpy()

    # Layout parameters
    pred_centroid = pred_layout[:, :3]
    gt_centroid = gt_layout[:, :3]
    pred_size = pred_layout[:, 3:6]
    gt_size = gt_layout[:, 3:6]
    pred_orientation = pred_layout[:, 6]
    gt_orientation = gt_layout[:, 6]

    # Camera parameters
    pred_pitch = pred_camera[:, 0]
    gt_pitch = gt_camera[:, 0]
    pred_roll = pred_camera[:, 1]
    gt_roll = gt_camera[:, 1]

    # Compute error metrics
    centroid_error = np.mean(np.sqrt(np.sum((pred_centroid - gt_centroid) ** 2, axis=1)))
    size_error = np.mean(np.sqrt(np.sum((pred_size - gt_size) ** 2, axis=1)))
    orientation_error = np.mean(np.abs(pred_orientation - gt_orientation))
    pitch_error = np.mean(np.abs(pred_pitch - gt_pitch))
    roll_error = np.mean(np.abs(pred_roll - gt_roll))

    # Compute 3D IoU between layout boxes
    iou_list = []
    for i in range(pred_centroid.shape[0]):
        iou = compute_3d_iou(
            pred_centroid[i], pred_size[i], pred_orientation[i],
            gt_centroid[i], gt_size[i], gt_orientation[i]
        )
        iou_list.append(iou)
    layout_iou = np.mean(iou_list)

    # Return metrics
    metrics = {
        'centroid_error': centroid_error,
        'size_error': size_error,
        'orientation_error': orientation_error,
        'pitch_error': pitch_error,
        'roll_error': roll_error,
        'layout_iou': layout_iou
    }

    return metrics


def compute_3d_iou(pred_centroid, pred_size, pred_orientation,
                   gt_centroid, gt_size, gt_orientation):
    """
    Compute 3D IoU between two layout boxes.

    Simplified version for axis-aligned boxes (ignoring orientation for now).
    In a full implementation, orientation would be considered.

    Args:
        pred_centroid (np.ndarray): Predicted box centroid [x, y, z]
        pred_size (np.ndarray): Predicted box size [width, height, depth]
        pred_orientation (float): Predicted box orientation angle
        gt_centroid (np.ndarray): Ground truth box centroid [x, y, z]
        gt_size (np.ndarray): Ground truth box size [width, height, depth]
        gt_orientation (float): Ground truth box orientation angle

    Returns:
        float: 3D IoU value (0-1)
    """
    # For simplicity, we're ignoring orientation and assuming axis-aligned boxes
    # In a full implementation, we would transform vertices with orientation

    # Calculate min and max bounds for both boxes
    pred_min = pred_centroid - pred_size / 2
    pred_max = pred_centroid + pred_size / 2

    gt_min = gt_centroid - gt_size / 2
    gt_max = gt_centroid + gt_size / 2

    # Calculate intersection bounds
    inter_min = np.maximum(pred_min, gt_min)
    inter_max = np.minimum(pred_max, gt_max)

    # Calculate intersection dimensions
    inter_dims = np.maximum(0, inter_max - inter_min)

    # Calculate volumes
    inter_volume = np.prod(inter_dims)
    pred_volume = np.prod(pred_size)
    gt_volume = np.prod(gt_size)

    # Calculate union volume
    union_volume = pred_volume + gt_volume - inter_volume

    # Calculate IoU
    iou = inter_volume / union_volume if union_volume > 0 else 0

    return iou


def test(config, device, checkpoint_path=None, output_dir=None):
    """
    Test the Layout Estimation Network.

    Args:
        config (dict): Configuration parameters
        device (torch.device): Device to use
        checkpoint_path (str, optional): Path to model checkpoint
        output_dir (str, optional): Directory to save outputs
    """
    # Setup output directory
    if output_dir is None:
        output_dir = config['test']['output_dir']
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize model
    model = LayoutEstimationNetwork(
        backbone=config['model']['backbone'],
        pretrained=False,  # No need for pretrained weights when loading checkpoint
        dropout=0.0  # No dropout during testing
    )
    model = model.to(device)

    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = config['test']['checkpoint']

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Get test dataloader
    test_loader = get_dataloader(config, split='test')

    # Evaluate model
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for batch_idx, (images, layout_gt, camera_gt) in enumerate(tqdm(test_loader, desc="Testing")):
            # Move data to device
            images = images.to(device)
            layout_gt = layout_gt.to(device)
            camera_gt = camera_gt.to(device)

            # Forward pass
            layout_pred, camera_pred = model(images)

            # Compute metrics
            metrics = evaluate_metrics(layout_pred, layout_gt, camera_pred, camera_gt)
            all_metrics.append(metrics)

            # Visualize results (for the first 10 samples)
            if batch_idx < 10 and config['test']['visualization']:
                for i in range(images.size(0)):
                    # Extract parameters for visualization
                    pred_centroid, pred_size, pred_orientation = model.get_layout_components(layout_pred[i:i + 1])
                    gt_centroid, gt_size, gt_orientation = model.get_layout_components(layout_gt[i:i + 1])

                    # Convert to numpy
                    pred_centroid = pred_centroid.cpu().numpy()[0]
                    pred_size = pred_size.cpu().numpy()[0]
                    pred_orientation = pred_orientation.cpu().numpy()[0][0]

                    gt_centroid = gt_centroid.cpu().numpy()[0]
                    gt_size = gt_size.cpu().numpy()[0]
                    gt_orientation = gt_orientation.cpu().numpy()[0][0]

                    # Visualize
                    fig = visualize_layout(
                        images[i].cpu(),
                        pred_centroid, pred_size, pred_orientation,
                        gt_centroid, gt_size, gt_orientation
                    )

                    # Save visualization
                    fig_path = output_path / f"sample_{batch_idx}_{i}.png"
                    fig.savefig(fig_path)
                    plt.close(fig)

    # Compute average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    # Print results
    print("\nTest Results:")
    print("-" * 50)
    for key, value in avg_metrics.items():
        print(f"{key}: {value:.4f}")
    print("-" * 50)

    # Save results to file
    results_path = output_path / "test_results.txt"
    with open(results_path, 'w') as f:
        f.write("Test Results:\n")
        f.write("-" * 50 + "\n")
        for key, value in avg_metrics.items():
            f.write(f"{key}: {value:.4f}\n")
        f.write("-" * 50 + "\n")

    print(f"Results saved to: {results_path}")

    return avg_metrics


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)

    print(f"Using device: {device}")
    test(config, device, args.checkpoint, args.output_dir)