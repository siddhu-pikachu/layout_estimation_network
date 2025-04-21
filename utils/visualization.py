import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torchvision.transforms as transforms


def visualize_layout(image, pred_centroid, pred_size, pred_orientation,
                     gt_centroid, gt_size, gt_orientation):
    """
    Visualize layout estimation results.

    Args:
        image (torch.Tensor): Input image tensor [C, H, W]
        pred_centroid (np.ndarray): Predicted centroid [x, y, z]
        pred_size (np.ndarray): Predicted size [width, height, depth]
        pred_orientation (float): Predicted orientation angle
        gt_centroid (np.ndarray): Ground truth centroid [x, y, z]
        gt_size (np.ndarray): Ground truth size [width, height, depth]
        gt_orientation (float): Ground truth orientation angle

    Returns:
        matplotlib.figure.Figure: Figure with layout visualization
    """
    # Create figure with two subplots
    fig = plt.figure(figsize=(16, 8))

    # Display input image
    ax1 = fig.add_subplot(1, 2, 1)

    # Convert tensor to numpy and denormalize
    img_np = image.permute(1, 2, 0).numpy()
    denorm = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    img_np = denorm(image).permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)

    ax1.imshow(img_np)
    ax1.set_title('Input Image')
    ax1.axis('off')

    # Display 3D layout
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    # Plot predicted layout box
    plot_3d_box(ax2, pred_centroid, pred_size, pred_orientation, color='blue', label='Predicted')

    # Plot ground truth layout box
    plot_3d_box(ax2, gt_centroid, gt_size, gt_orientation, color='green', label='Ground Truth')

    # Set plot properties
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')  # Z is typically depth in camera-centric coordinates
    ax2.set_zlabel('Y')  # Y is typically height in camera-centric coordinates

    # Set equal aspect ratio
    set_axes_equal(ax2)

    # Set title and legend
    ax2.set_title('3D Layout Prediction')
    ax2.legend()

    # Add metrics to the plot
    centroid_error = np.sqrt(np.sum((pred_centroid - gt_centroid) ** 2))
    size_error = np.sqrt(np.sum((pred_size - gt_size) ** 2))
    orientation_error = np.abs(pred_orientation - gt_orientation)

    metrics_text = f"Centroid Error: {centroid_error:.4f}\n" \
                   f"Size Error: {size_error:.4f}\n" \
                   f"Orientation Error: {orientation_error:.4f}"

    fig.text(0.5, 0.01, metrics_text, ha='center', fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()

    return fig


def plot_3d_box(ax, centroid, size, orientation, color='blue', label=None):
    """
    Plot a 3D box on the given matplotlib 3D axis.

    Args:
        ax (matplotlib.axes.Axes): Matplotlib 3D axis
        centroid (np.ndarray): Box centroid [x, y, z]
        size (np.ndarray): Box size [width, height, depth]
        orientation (float): Box orientation angle in radians
        color (str): Color for the box
        label (str, optional): Label for the legend
    """
    # For simplicity, we'll ignore orientation for now and plot axis-aligned box
    # In a full implementation, we would apply rotation based on orientation

    # Calculate the corners of the box
    x, y, z = centroid
    w, h, d = size

    # Box corners (assuming axis-aligned for simplicity)
    corners = np.array([
        [x - w / 2, y - h / 2, z - d / 2],  # 0: left, bottom, back
        [x + w / 2, y - h / 2, z - d / 2],  # 1: right, bottom, back
        [x + w / 2, y + h / 2, z - d / 2],  # 2: right, top, back
        [x - w / 2, y + h / 2, z - d / 2],  # 3: left, top, back
        [x - w / 2, y - h / 2, z + d / 2],  # 4: left, bottom, front
        [x + w / 2, y - h / 2, z + d / 2],  # 5: right, bottom, front
        [x + w / 2, y + h / 2, z + d / 2],  # 6: right, top, front
        [x - w / 2, y + h / 2, z + d / 2]  # 7: left, top, front
    ])

    # Define edges as pairs of corner indices
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # back face
        (4, 5), (5, 6), (6, 7), (7, 4),  # front face
        (0, 4), (1, 5), (2, 6), (3, 7)  # connecting edges
    ]

    # Plot edges
    for edge in edges:
        i, j = edge
        ax.plot([corners[i, 0], corners[j, 0]],
                [corners[i, 2], corners[j, 2]],  # Z is Y in the plot
                [corners[i, 1], corners[j, 1]],  # Y is Z in the plot
                color=color)

    # Plot centroid
    ax.scatter(centroid[0], centroid[2], centroid[1], color=color, marker='o',
               s=50, label=label)


def set_axes_equal(ax):
    """
    Set equal scaling for 3D axes to ensure the box isn't distorted.

    Args:
        ax (matplotlib.axes.Axes): Matplotlib 3D axis
    """
    # Get limits for each axis
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    # Calculate ranges for each axis
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    # Find the maximum range
    max_range = max(x_range, y_range, z_range)

    # Calculate new limits
    x_mid = (x_limits[1] + x_limits[0]) / 2
    y_mid = (y_limits[1] + y_limits[0]) / 2
    z_mid = (z_limits[1] + z_limits[0]) / 2

    ax.set_xlim3d([x_mid - max_range / 2, x_mid + max_range / 2])
    ax.set_ylim3d([y_mid - max_range / 2, y_mid + max_range / 2])
    ax.set_zlim3d([z_mid - max_range / 2, z_mid + max_range / 2])