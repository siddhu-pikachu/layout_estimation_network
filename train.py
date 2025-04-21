import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import numpy as np
from pathlib import Path

# Import project modules
from models.layout_estimation import LayoutEstimationNetwork
from data.dataset import get_dataloader


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train Layout Estimation Network')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def layout_loss_function(pred_layout, gt_layout, pred_camera, gt_camera, config):
    """
    Custom loss function for layout estimation.

    Args:
        pred_layout (torch.Tensor): Predicted layout parameters
        gt_layout (torch.Tensor): Ground truth layout parameters
        pred_camera (torch.Tensor): Predicted camera parameters
        gt_camera (torch.Tensor): Ground truth camera parameters
        config (dict): Configuration parameters

    Returns:
        torch.Tensor: Total loss
    """
    # Extract weights from config
    layout_weight = config['loss']['layout_weight']
    camera_weight = config['loss']['camera_weight']
    regularization_weight = config['loss']['regularization_weight']

    # Calculate layout parameter loss
    layout_loss = nn.MSELoss()(pred_layout, gt_layout)

    # Calculate camera parameter loss
    camera_loss = nn.MSELoss()(pred_camera, gt_camera)

    # Calculate regularization loss (L2 norm of parameters to prevent overfitting)
    layout_reg_loss = torch.norm(pred_layout, p=2)
    camera_reg_loss = torch.norm(pred_camera, p=2)
    reg_loss = (layout_reg_loss + camera_reg_loss) / 2.0

    # Combine losses
    total_loss = (
            layout_weight * layout_loss +
            camera_weight * camera_loss +
            regularization_weight * reg_loss
    )

    return total_loss, layout_loss, camera_loss


def validate(model, val_loader, device, config):
    """
    Validate the model on the validation set.

    Args:
        model (nn.Module): Model to validate
        val_loader (DataLoader): Validation dataloader
        device (torch.device): Device to use
        config (dict): Configuration parameters

    Returns:
        tuple: (average_loss, average_layout_loss, average_camera_loss)
    """
    model.eval()
    total_loss = 0.0
    total_layout_loss = 0.0
    total_camera_loss = 0.0

    with torch.no_grad():
        for images, layout_gt, camera_gt in tqdm(val_loader, desc="Validating"):
            # Move data to device
            images = images.to(device)
            layout_gt = layout_gt.to(device)
            camera_gt = camera_gt.to(device)

            # Forward pass
            layout_pred, camera_pred = model(images)

            # Calculate loss
            loss, layout_loss, camera_loss = layout_loss_function(
                layout_pred, layout_gt, camera_pred, camera_gt, config
            )

            # Update running loss
            total_loss += loss.item()
            total_layout_loss += layout_loss.item()
            total_camera_loss += camera_loss.item()

    # Calculate averages
    avg_loss = total_loss / len(val_loader)
    avg_layout_loss = total_layout_loss / len(val_loader)
    avg_camera_loss = total_camera_loss / len(val_loader)

    return avg_loss, avg_layout_loss, avg_camera_loss


def train(config, device):
    """
    Train the Layout Estimation Network.

    Args:
        config (dict): Configuration parameters
        device (torch.device): Device to use for training
    """
    # Create save directory
    save_dir = Path(config['train']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    model = LayoutEstimationNetwork(
        backbone=config['model']['backbone'],
        pretrained=config['model']['pretrained'],
        dropout=config['model']['dropout']
    )
    model = model.to(device)

    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['train']['learning_rate'],
        weight_decay=config['train']['weight_decay']
    )

    # Initialize learning rate scheduler
    if config['train']['lr_scheduler']:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['train']['lr_step_size'],
            gamma=config['train']['lr_gamma']
        )

    # Get dataloaders
    train_loader = get_dataloader(config, split='train')
    val_loader = get_dataloader(config, split='val')

    # Resume training if specified
    start_epoch = 0
    best_val_loss = float('inf')
    if config['train']['resume']:
        checkpoint_path = config['train']['resume']
        if os.path.exists(checkpoint_path):
            print(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            print(f"Resuming from epoch {start_epoch} with best validation loss: {best_val_loss}")

    # Training loop
    num_epochs = config['train']['epochs']
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        running_layout_loss = 0.0
        running_camera_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for images, layout_gt, camera_gt in progress_bar:
            # Move data to device
            images = images.to(device)
            layout_gt = layout_gt.to(device)
            camera_gt = camera_gt.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            layout_pred, camera_pred = model(images)

            # Calculate loss
            loss, layout_loss, camera_loss = layout_loss_function(
                layout_pred, layout_gt, camera_pred, camera_gt, config
            )

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update running loss
            running_loss += loss.item()
            running_layout_loss += layout_loss.item()
            running_camera_loss += camera_loss.item()

            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'layout_loss': layout_loss.item(),
                'camera_loss': camera_loss.item()
            })

        # Calculate average losses for the epoch
        avg_train_loss = running_loss / len(train_loader)
        avg_train_layout_loss = running_layout_loss / len(train_loader)
        avg_train_camera_loss = running_camera_loss / len(train_loader)

        # Print training statistics
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, "
              f"Layout Loss: {avg_train_layout_loss:.4f}, "
              f"Camera Loss: {avg_train_camera_loss:.4f}")

        # Validate
        val_loss, val_layout_loss, val_camera_loss = validate(model, val_loader, device, config)
        print(f"Val Loss: {val_loss:.4f}, "
              f"Layout Loss: {val_layout_loss:.4f}, "
              f"Camera Loss: {val_camera_loss:.4f}")

        # Update learning rate
        if config['train']['lr_scheduler']:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"Current Learning Rate: {current_lr:.6f}")

        # Save model checkpoint
        checkpoint_path = save_dir / f"epoch_{epoch + 1}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss
        }, checkpoint_path)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = save_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'best_val_loss': best_val_loss
            }, best_model_path)
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")

        print("-" * 50)

    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)

    print(f"Using device: {device}")
    train(config, device)