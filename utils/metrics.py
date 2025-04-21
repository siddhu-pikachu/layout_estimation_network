import numpy as np
import torch


def compute_3d_iou(pred_centroid, pred_size, pred_orientation,
                   gt_centroid, gt_size, gt_orientation):
    """
    Compute 3D IoU between two layout boxes.

    Args:
        pred_centroid (np.ndarray or torch.Tensor): Predicted box centroid [x, y, z]
        pred_size (np.ndarray or torch.Tensor): Predicted box size [width, height, depth]
        pred_orientation (float or torch.Tensor): Predicted box orientation angle
        gt_centroid (np.ndarray or torch.Tensor): Ground truth box centroid [x, y, z]
        gt_size (np.ndarray or torch.Tensor): Ground truth box size [width, height, depth]
        gt_orientation (float or torch.Tensor): Ground truth box orientation angle

    Returns:
        float: 3D IoU value (0-1)
    """
    # Convert to numpy if tensors
    if isinstance(pred_centroid, torch.Tensor):
        pred_centroid = pred_centroid.detach().cpu().numpy()
    if isinstance(pred_size, torch.Tensor):
        pred_size = pred_size.detach().cpu().numpy()
    if isinstance(pred_orientation, torch.Tensor):
        pred_orientation = pred_orientation.detach().cpu().numpy()
    if isinstance(gt_centroid, torch.Tensor):
        gt_centroid = gt_centroid.detach().cpu().numpy()
    if isinstance(gt_size, torch.Tensor):
        gt_size = gt_size.detach().cpu().numpy()
    if isinstance(gt_orientation, torch.Tensor):
        gt_orientation = gt_orientation.detach().cpu().numpy()

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


def compute_layout_accuracy(pred_layout, gt_layout, threshold=0.5):
    """
    Compute the layout estimation accuracy based on IoU threshold.

    Args:
        pred_layout (torch.Tensor): Predicted layout parameters batch [B, 7]
        gt_layout (torch.Tensor): Ground truth layout parameters batch [B, 7]
        threshold (float): IoU threshold for considering a prediction correct

    Returns:
        float: Accuracy (percentage of layouts with IoU > threshold)
    """
    batch_size = pred_layout.size(0)
    correct = 0

    for i in range(batch_size):
        # Extract layout components
        pred_centroid = pred_layout[i, :3]
        pred_size = pred_layout[i, 3:6]
        pred_orientation = pred_layout[i, 6]

        gt_centroid = gt_layout[i, :3]
        gt_size = gt_layout[i, 3:6]
        gt_orientation = gt_layout[i, 6]

        # Compute IoU
        iou = compute_3d_iou(
            pred_centroid, pred_size, pred_orientation,
            gt_centroid, gt_size, gt_orientation
        )

        # Check if prediction is correct
        if iou > threshold:
            correct += 1

    accuracy = correct / batch_size
    return accuracy


def compute_distance_error(pred_centroid, gt_centroid):
    """
    Compute the Euclidean distance error between predicted and ground truth centroids.

    Args:
        pred_centroid (torch.Tensor): Predicted centroid [B, 3]
        gt_centroid (torch.Tensor): Ground truth centroid [B, 3]

    Returns:
        torch.Tensor: Average distance error
    """
    return torch.sqrt(torch.sum((pred_centroid - gt_centroid) ** 2, dim=1)).mean()


def compute_orientation_error(pred_orientation, gt_orientation):
    """
    Compute the orientation error in degrees.

    Args:
        pred_orientation (torch.Tensor): Predicted orientation [B, 1]
        gt_orientation (torch.Tensor): Ground truth orientation [B, 1]

    Returns:
        torch.Tensor: Average orientation error in degrees
    """
    # Convert to degrees
    pred_deg = pred_orientation * 180.0 / np.pi
    gt_deg = gt_orientation * 180.0 / np.pi

    # Calculate absolute error
    error = torch.abs(pred_deg - gt_deg)

    # Handle angle wrapping (e.g., 350° vs 10°)
    error = torch.min(error, 360.0 - error)

    return error.mean()