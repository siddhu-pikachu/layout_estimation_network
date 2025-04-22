import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time
import datetime

# Import project modules
from models.layout_estimation import LayoutEstimationNetwork, Total3DLayoutEstimationNetwork
from data.dataset import Pix3DLayoutDataset, get_data_loaders
from utils.visualization import visualize_layout, plot_3d_box, set_axes_equal
from utils.metrics import compute_3d_iou, compute_layout_accuracy


def plot_training_progress(history, save_path):
    """
    Plot training and validation metrics.

    Args:
        history (dict): Dictionary containing training history
        save_path (Path): Path to save the plot
    """
    plt.figure(figsize=(15, 10))

    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Plot IoU
    plt.subplot(2, 2, 2)
    plt.plot(history['val_iou'], label='Val IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('Validation IoU')
    plt.legend()
    plt.grid(True)

    # Plot learning rate
    plt.subplot(2, 2, 3)
    plt.plot(history['lr'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)

    # Add more component loss plots if available
    if len(history) > 4:
        plt.subplot(2, 2, 4)
        for key in history:
            if key not in ['train_loss', 'val_loss', 'val_iou', 'lr']:
                plt.plot(history[key], label=key)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Component Losses')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train Layout Estimation Network')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda, mps, or cpu)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from the latest checkpoint')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode (fewer iterations)')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def layout_loss_function(pred_layout, gt_layout, pred_camera, gt_camera, config):
    """
    Custom loss function for layout estimation aligned with Total3D.

    Args:
        pred_layout (torch.Tensor): Predicted layout parameters [B, 7]
        gt_layout (torch.Tensor): Ground truth layout parameters [B, 7]
        pred_camera (torch.Tensor): Predicted camera parameters [B, 2]
        gt_camera (torch.Tensor): Ground truth camera parameters [B, 2]
        config (dict): Configuration parameters

    Returns:
        tuple: (total_loss, loss_dict)
    """
    # Extract layout components
    pred_centroid = pred_layout[:, :3]
    pred_size = pred_layout[:, 3:6]
    pred_orientation = pred_layout[:, 6:7]

    gt_centroid = gt_layout[:, :3]
    gt_size = gt_layout[:, 3:6]
    gt_orientation = gt_layout[:, 6:7]

    # Extract weights from config
    layout_weight = config['loss']['layout_weight']
    camera_weight = config['loss']['camera_weight']
    centroid_weight = config['loss'].get('centroid_weight', 1.0)
    size_weight = config['loss'].get('size_weight', 1.0)
    orientation_weight = config['loss'].get('orientation_weight', 0.5)

    # Centroid loss (smooth L1 loss for robustness)
    centroid_loss = F.smooth_l1_loss(pred_centroid, gt_centroid)

    # Size loss (using log space for scale invariance)
    # Handle potential negative values in pred_size
    pred_size_safe = torch.clamp(pred_size, min=1e-6)
    gt_size_safe = torch.clamp(gt_size, min=1e-6)
    size_loss = F.smooth_l1_loss(torch.log(pred_size_safe), torch.log(gt_size_safe))

    # Orientation loss (normalize angle differences)
    orientation_diff = torch.abs(pred_orientation - gt_orientation)
    orientation_diff = torch.min(orientation_diff, 2 * np.pi - orientation_diff)
    orientation_loss = orientation_diff.mean()

    # Camera parameter loss
    camera_loss = F.smooth_l1_loss(pred_camera, gt_camera)

    # Combined layout loss
    layout_loss = (
            centroid_weight * centroid_loss +
            size_weight * size_loss +
            orientation_weight * orientation_loss
    )

    # Total loss
    total_loss = layout_weight * layout_loss + camera_weight * camera_loss

    # Create dictionary for logging
    loss_dict = {
        'total_loss': total_loss.item(),
        'layout_loss': layout_loss.item(),
        'centroid_loss': centroid_loss.item(),
        'size_loss': size_loss.item(),
        'orientation_loss': orientation_loss.item(),
        'camera_loss': camera_loss.item()
    }

    return total_loss, loss_dict


def validate(model, val_loader, device, config, epoch, output_dir=None):
    """
    Validate the model on the validation set.

    Args:
        model (nn.Module): Model to validate
        val_loader (DataLoader): Validation dataloader
        device (torch.device): Device to use
        config (dict): Configuration parameters
        epoch (int): Current epoch number
        output_dir (Path, optional): Directory to save visualizations

    Returns:
        dict: Dictionary with validation metrics
    """
    model.eval()
    val_losses = []
    loss_components = {'layout_loss': [], 'centroid_loss': [], 'size_loss': [],
                       'orientation_loss': [], 'camera_loss': []}
    iou_values = []

    # Create output directory for visualizations
    if output_dir is not None:
        vis_dir = output_dir / f"val_vis_epoch_{epoch}"
        vis_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (images, layout_gt, camera_gt) in enumerate(tqdm(val_loader, desc="Validating")):
            # Move data to device
            images = images.to(device)
            layout_gt = layout_gt.to(device)
            camera_gt = camera_gt.to(device)

            # Forward pass
            layout_pred, camera_pred = model(images)

            # Calculate loss
            _, loss_dict = layout_loss_function(
                layout_pred, layout_gt, camera_pred, camera_gt, config
            )

            # Track losses
            val_losses.append(loss_dict['total_loss'])
            for key in loss_components:
                if key in loss_dict:
                    loss_components[key].append(loss_dict[key])

            # Calculate IoU for each sample in the batch
            for i in range(images.size(0)):
                # Extract components
                pred_centroid, pred_size, pred_orientation = model.get_layout_components(layout_pred[i:i + 1])
                gt_centroid, gt_size, gt_orientation = model.get_layout_components(layout_gt[i:i + 1])

                # Convert to numpy
                pred_centroid_np = pred_centroid.cpu().numpy()[0]
                pred_size_np = pred_size.cpu().numpy()[0]
                pred_orientation_np = pred_orientation.cpu().numpy()[0][0]

                gt_centroid_np = gt_centroid.cpu().numpy()[0]
                gt_size_np = gt_size.cpu().numpy()[0]
                gt_orientation_np = gt_orientation.cpu().numpy()[0][0]

                # Calculate IoU
                iou = compute_3d_iou(
                    pred_centroid_np, pred_size_np, pred_orientation_np,
                    gt_centroid_np, gt_size_np, gt_orientation_np
                )
                iou_values.append(iou)

                # Save visualizations for the first few samples
                if output_dir is not None and batch_idx < 5 and i < 2:
                    fig = visualize_layout(
                        images[i].cpu(),
                        pred_centroid_np, pred_size_np, pred_orientation_np,
                        gt_centroid_np, gt_size_np, gt_orientation_np
                    )

                    # Add IoU to the figure
                    plt.figtext(0.5, 0.01, f"IoU: {iou:.4f}", ha='center', fontsize=12,
                                bbox=dict(facecolor='white', alpha=0.8))

                    # Save figure
                    fig_path = vis_dir / f"sample_{batch_idx}_{i}.png"
                    fig.savefig(fig_path)
                    plt.close(fig)

    # Calculate average metrics
    avg_loss = np.mean(val_losses)
    avg_components = {key: np.mean(values) for key, values in loss_components.items() if values}
    avg_iou = np.mean(iou_values)

    # Compute IoU accuracy at different thresholds
    iou_accuracy = {}
    for threshold in [0.25, 0.5, 0.75]:
        iou_accuracy[f'iou_{threshold}'] = np.mean([1.0 if iou >= threshold else 0.0 for iou in iou_values])

    # Combine all metrics
    metrics = {
        'val_loss': avg_loss,
        'val_iou': avg_iou,
        **avg_components,
        **iou_accuracy
    }

    return metrics


def train_epoch(model, train_loader, optimizer, device, config, epoch):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training dataloader
        optimizer (optim.Optimizer): Optimizer
        device (torch.device): Device to use
        config (dict): Configuration parameters
        epoch (int): Current epoch number

    Returns:
        dict: Dictionary with training metrics
    """
    model.train()
    train_losses = []
    loss_components = {'layout_loss': [], 'centroid_loss': [], 'size_loss': [],
                       'orientation_loss': [], 'camera_loss': []}

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, (images, layout_gt, camera_gt) in enumerate(progress_bar):
        # Move data to device
        images = images.to(device)
        layout_gt = layout_gt.to(device)
        camera_gt = camera_gt.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        layout_pred, camera_pred = model(images)

        # Calculate loss
        loss, loss_dict = layout_loss_function(
            layout_pred, layout_gt, camera_pred, camera_gt, config
        )

        # Backward pass and optimize
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        if config['train'].get('clip_grad', False):
            clip_value = config['train'].get('clip_value', 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()

        # Track losses
        train_losses.append(loss_dict['total_loss'])
        for key in loss_components:
            if key in loss_dict:
                loss_components[key].append(loss_dict[key])

        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss_dict['total_loss'],
            'layout_loss': loss_dict['layout_loss'],
            'camera_loss': loss_dict['camera_loss']
        })

    # Calculate average metrics
    avg_loss = np.mean(train_losses)
    avg_components = {key: np.mean(values) for key, values in loss_components.items() if values}

    # Combine all metrics
    metrics = {
        'train_loss': avg_loss,
        **{f'train_{k}': v for k, v in avg_components.items()}
    }

    return metrics


def train(config, device, resume=False, debug=False):
    """
    Train the Layout Estimation Network with advanced techniques.

    Args:
        config (dict): Configuration parameters
        device (torch.device): Device to use for training
        resume (bool): Whether to resume training from checkpoint
        debug (bool): Enable debug mode (fewer iterations)
    """
    # Create save directory
    save_dir = Path(config['train']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create log directory for visualizations
    log_dir = save_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    model_name = config['model']['name']
    if model_name == "Total3DLayoutEstimationNetwork":
        from models.layout_estimation import Total3DLayoutEstimationNetwork
        model = Total3DLayoutEstimationNetwork(
            backbone=config['model']['backbone'],
            pretrained=config['model']['pretrained']
        )
    else:
        model = LayoutEstimationNetwork(
            backbone=config['model']['backbone'],
            pretrained=config['model']['pretrained']
        )
    model = model.to(device)

    # Initialize optimizer - use Adam with lower learning rate as in Total3D
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['train']['learning_rate'],
        weight_decay=config['train']['weight_decay']
    )

    # Initialize learning rate scheduler - cosine annealing works better than step
    if config['train'].get('lr_scheduler', 'cosine') == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['train']['epochs'],
            eta_min=config['train'].get('min_lr', 1e-6)
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['train']['lr_step_size'],
            gamma=config['train']['lr_gamma']
        )

    # Get dataloaders
    train_loader, val_loader, test_loader = get_data_loaders(config)

    # Setup for training
    start_epoch = 0
    best_val_iou = 0.0
    train_history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'lr': []}

    # Resume training if specified
    if resume:
        checkpoint_path = save_dir / "latest_checkpoint.pth"
        if checkpoint_path.exists():
            print(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_iou = checkpoint.get('best_val_iou', 0.0)

            # Load training history if available
            if 'history' in checkpoint:
                train_history = checkpoint['history']

            print(f"Resuming from epoch {start_epoch} with best val IoU: {best_val_iou:.4f}")

    # Training loop
    num_epochs = 5 if debug else config['train']['epochs']
    print(f"Starting training for {num_epochs} epochs on {device}")
    print(f"Training set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")

    start_time = time.time()

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()

        # Train for one epoch
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            config=config,
            epoch=epoch + 1  # 1-indexed for display
        )

        # Validate
        val_metrics = validate(
            model=model,
            val_loader=val_loader,
            device=device,
            config=config,
            epoch=epoch + 1,  # 1-indexed for display
            output_dir=log_dir if epoch % 5 == 0 else None  # Save visualizations every 5 epochs
        )

        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Update training history
        train_history['train_loss'].append(train_metrics['train_loss'])
        train_history['val_loss'].append(val_metrics['val_loss'])
        train_history['val_iou'].append(val_metrics['val_iou'])
        train_history['lr'].append(current_lr)

        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch + 1}/{num_epochs} completed in {epoch_time:.2f}s")
        print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        print(f"  Val Loss: {val_metrics['val_loss']:.4f}, Val IoU: {val_metrics['val_iou']:.4f}")
        print(f"  IoU@0.25: {val_metrics['iou_0.25']:.4f}, IoU@0.5: {val_metrics['iou_0.5']:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_metrics['val_loss'],
            'val_iou': val_metrics['val_iou'],
            'best_val_iou': best_val_iou,
            'history': train_history,
            'config': config
        }

        # Save latest checkpoint
        torch.save(checkpoint, save_dir / "latest_checkpoint.pth")

        # Save periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, save_dir / f"checkpoint_epoch_{epoch + 1}.pth")

        # Save best model based on IoU
        if val_metrics['val_iou'] > best_val_iou:
            best_val_iou = val_metrics['val_iou']
            checkpoint['best_val_iou'] = best_val_iou
            torch.save(checkpoint, save_dir / "best_model.pth")
            print(f"  New best model saved with val IoU: {best_val_iou:.4f}")

        # Plot training progress
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            plot_training_progress(train_history, log_dir / f"training_progress_epoch_{epoch + 1}.png")

        print("-" * 80)

    # Training complete
    total_time = time.time() - start_time
    print(f"\nTraining completed in {str(datetime.timedelta(seconds=int(total_time)))}")
    print(f"Best validation IoU: {best_val_iou:.4f}")

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_metrics = validate(
        model=model,
        val_loader=test_loader,  # Using the test loader
        device=device,
        config=config,
        epoch=num_epochs,
        output_dir=log_dir / "test_results"
    )

    print("\nTest Results:")
    print(f"  Test Loss: {test_metrics['val_loss']:.4f}")
    print(f"  Test IoU: {test_metrics['val_iou']:.4f}")
    print(f"  IoU@0.25: {test_metrics['iou_0.25']:.4f}, IoU@0.5: {test_metrics['iou_0.5']:.4f}")

    # Save full test results
    with open(log_dir / "test_results.txt", "w") as f:
        f.write(f"Test Results for {config['model']['name']}\n")
        f.write(f"Backbone: {config['model']['backbone']}\n")
        f.write("-" * 40 + "\n")
        for key, value in test_metrics.items():
            f.write(f"{key}: {value:.6f}\n")

    return best_val_iou, test_metrics