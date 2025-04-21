import os
import yaml
import torch
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from mpl_toolkits.mplot3d import Axes3D

# Import project modules
from models.layout_estimation import LayoutEstimationNetwork
from utils.visualization import visualize_layout, plot_3d_box, set_axes_equal


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Demo for Layout Estimation Network')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (overrides config)')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='outputs/demo',
                        help='Directory to save outputs')
    parser.add_argument('--device', type=str, default='mps' if torch.backends.mps.is_available() else 'cpu',
                        help='Device to use (mps, cuda, or cpu)')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def preprocess_image(image_path, img_size=256):
    """
    Preprocess image for model input.

    Args:
        image_path (str): Path to input image
        img_size (int): Size to resize image to

    Returns:
        tuple: (image_tensor, original_image)
            - image_tensor (torch.Tensor): Preprocessed image tensor [1, C, H, W]
            - original_image (PIL.Image): Original image
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    return image_tensor, image


def run_demo(config, device, checkpoint_path, image_path, output_dir):
    """
    Run demo on a single image.

    Args:
        config (dict): Configuration parameters
        device (torch.device): Device to use
        checkpoint_path (str): Path to model checkpoint
        image_path (str): Path to input image
        output_dir (str): Directory to save outputs
    """
    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize model
    model = LayoutEstimationNetwork(
        backbone=config['model']['backbone'],
        pretrained=False,  # No need for pretrained weights when loading checkpoint
        dropout=0.0  # No dropout during inference
    )
    model = model.to(device)

    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = config['test']['checkpoint']

    print(f"Loading checkpoint from: {checkpoint_path}")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Checkpoint loaded successfully")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Running with randomly initialized weights")

    # Preprocess image
    img_size = config['dataset']['img_size']
    image_tensor, original_image = preprocess_image(image_path, img_size)
    image_tensor = image_tensor.to(device)

    # Run inference
    model.eval()
    with torch.no_grad():
        layout_pred, camera_pred = model(image_tensor)

    # Extract layout components
    pred_centroid, pred_size, pred_orientation = model.get_layout_components(layout_pred)

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

    # Visualize results
    # Create figure
    fig = plt.figure(figsize=(12, 6))

    # Display input image
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(np.array(original_image))
    ax1.set_title('Input Image')
    ax1.axis('off')

    # Display 3D layout
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    # Plot predicted layout box
    plot_3d_box(ax2, pred_centroid, pred_size, pred_orientation, color='blue', label='Predicted Layout')

    # Set plot properties
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')  # Z is typically depth in camera-centric coordinates
    ax2.set_zlabel('Y')  # Y is typically height in camera-centric coordinates

    # Set equal aspect ratio
    set_axes_equal(ax2)

    # Set title and legend
    ax2.set_title('3D Layout Prediction')
    ax2.legend()

    # Add camera parameters to the plot
    camera_text = f"Camera Pitch: {camera_pred[0, 0].item():.4f} rad\n" \
                  f"Camera Roll: {camera_pred[0, 1].item():.4f} rad"

    fig.text(0.5, 0.01, camera_text, ha='center', fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save visualization
    output_file = output_path / f"{Path(image_path).stem}_layout.png"
    plt.savefig(output_file)
    print(f"\nVisualization saved to: {output_file}")

    # Generate 3D visualization from different viewpoints
    views = [
        (30, 30),  # (elevation, azimuth)
        (0, 0),  # Front view
        (0, 90),  # Side view
        (90, 0)  # Top view
    ]

    for i, (elev, azim) in enumerate(views):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot layout
        plot_3d_box(ax, pred_centroid, pred_size, pred_orientation, color='blue', label='Layout')

        # Add camera position
        # In a simple approximation, we place camera at origin looking towards -Z
        ax.scatter(0, 0, 0, color='red', marker='^', s=100, label='Camera')

        # Set viewpoint
        ax.view_init(elev=elev, azim=azim)

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        ax.set_title(f'View {i + 1}: Elevation={elev}, Azimuth={azim}')

        # Set equal aspect ratio
        set_axes_equal(ax)

        # Add legend
        ax.legend()

        # Save figure
        view_file = output_path / f"{Path(image_path).stem}_view_{i + 1}.png"
        plt.savefig(view_file)
        plt.close(fig)

    print(f"Additional viewpoints saved to: {output_path}")

    # Generate a 3D model file (simple OBJ format)
    obj_file = output_path / f"{Path(image_path).stem}_layout.obj"

    # Create vertices of the box
    x, y, z = pred_centroid
    w, h, d = pred_size

    # Box corners (assuming axis-aligned for simplicity)
    vertices = np.array([
        [x - w / 2, y - h / 2, z - d / 2],  # 0: left, bottom, back
        [x + w / 2, y - h / 2, z - d / 2],  # 1: right, bottom, back
        [x + w / 2, y + h / 2, z - d / 2],  # 2: right, top, back
        [x - w / 2, y + h / 2, z - d / 2],  # 3: left, top, back
        [x - w / 2, y - h / 2, z + d / 2],  # 4: left, bottom, front
        [x + w / 2, y - h / 2, z + d / 2],  # 5: right, bottom, front
        [x + w / 2, y + h / 2, z + d / 2],  # 6: right, top, front
        [x - w / 2, y + h / 2, z + d / 2]  # 7: left, top, front
    ])

    # Define faces as vertex indices (1-indexed for OBJ format)
    faces = [
        [1, 2, 3, 4],  # back face
        [5, 6, 7, 8],  # front face
        [1, 5, 8, 4],  # left face
        [2, 6, 7, 3],  # right face
        [4, 3, 7, 8],  # top face
        [1, 2, 6, 5]  # bottom face
    ]

    # Write OBJ file
    with open(obj_file, 'w') as f:
        f.write("# 3D layout model generated by Layout Estimation Network\n")

        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # Write faces
        for face in faces:
            f.write(f"f {' '.join(str(i) for i in face)}\n")

    print(f"3D model saved to: {obj_file}")

    # Create a simple HTML visualization report
    html_file = output_path / f"{Path(image_path).stem}_report.html"

    with open(html_file, 'w') as f:
        f.write("<!DOCTYPE html>\n")
        f.write("<html>\n")
        f.write("<head>\n")
        f.write("    <title>Layout Estimation Results</title>\n")
        f.write("    <style>\n")
        f.write("        body { font-family: Arial, sans-serif; margin: 20px; }\n")
        f.write("        .image-container { display: flex; flex-wrap: wrap; }\n")
        f.write("        .image-item { margin: 10px; text-align: center; }\n")
        f.write("        table { border-collapse: collapse; width: 100%; }\n")
        f.write("        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n")
        f.write("        th { background-color: #f2f2f2; }\n")
        f.write("    </style>\n")
        f.write("</head>\n")
        f.write("<body>\n")
        f.write("    <h1>Layout Estimation Results</h1>\n")

        # Input image
        f.write("    <h2>Input Image</h2>\n")
        rel_path = os.path.relpath(image_path, output_path)
        f.write(f"    <img src='{rel_path}' style='max-width: 500px;'><br>\n")

        # Layout parameters
        f.write("    <h2>Estimated Layout Parameters</h2>\n")
        f.write("    <table>\n")
        f.write("        <tr><th>Parameter</th><th>Value</th></tr>\n")
        f.write(
            f"        <tr><td>Centroid (x, y, z)</td><td>{pred_centroid[0]:.4f}, {pred_centroid[1]:.4f}, {pred_centroid[2]:.4f}</td></tr>\n")
        f.write(
            f"        <tr><td>Size (width, height, depth)</td><td>{pred_size[0]:.4f}, {pred_size[1]:.4f}, {pred_size[2]:.4f}</td></tr>\n")
        f.write(f"        <tr><td>Orientation</td><td>{pred_orientation:.4f} radians</td></tr>\n")
        f.write(f"        <tr><td>Camera Pitch</td><td>{camera_pred[0, 0].item():.4f} radians</td></tr>\n")
        f.write(f"        <tr><td>Camera Roll</td><td>{camera_pred[0, 1].item():.4f} radians</td></tr>\n")
        f.write("    </table>\n")

        # Visualizations
        f.write("    <h2>Visualizations</h2>\n")
        f.write("    <div class='image-container'>\n")

        # Main visualization
        main_vis = f"{Path(image_path).stem}_layout.png"
        f.write("        <div class='image-item'>\n")
        f.write(f"            <img src='{main_vis}' style='width: 500px;'><br>\n")
        f.write("            <p>3D Layout Estimation</p>\n")
        f.write("        </div>\n")

        # Different viewpoints
        for i in range(len(views)):
            view_file = f"{Path(image_path).stem}_view_{i + 1}.png"
            f.write("        <div class='image-item'>\n")
            f.write(f"            <img src='{view_file}' style='width: 300px;'><br>\n")
            f.write(f"            <p>View {i + 1}</p>\n")
            f.write("        </div>\n")

        f.write("    </div>\n")

        # 3D model download link
        obj_filename = f"{Path(image_path).stem}_layout.obj"
        f.write("    <h2>3D Model</h2>\n")
        f.write(f"    <p>Download the <a href='{obj_filename}'>3D model</a> (OBJ format)</p>\n")

        f.write("</body>\n")
        f.write("</html>\n")

    print(f"HTML report saved to: {html_file}")

    # Open the report in the default browser (optional)
    # import webbrowser
    # webbrowser.open(f'file://{html_file.absolute()}')

    return {
        'centroid': pred_centroid,
        'size': pred_size,
        'orientation': pred_orientation,
        'camera_pitch': camera_pred[0, 0].item(),
        'camera_roll': camera_pred[0, 1].item()
    }


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)

    print(f"Using device: {device}")
    print(f"Processing image: {args.image}")

    results = run_demo(config, device, args.checkpoint, args.image, args.output_dir)