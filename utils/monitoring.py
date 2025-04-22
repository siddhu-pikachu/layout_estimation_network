import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorboardX import SummaryWriter
import logging


class TrainingMonitor:
    """
    Utility class for monitoring training progress and keeping training statistics.
    Provides logging, visualization, and early stopping functionality.
    """

    def __init__(self, config, log_dir=None):
        """
        Initialize the training monitor.

        Args:
            config (dict): Configuration parameters
            log_dir (str, optional): Directory to save logs and visualizations
        """
        self.config = config
        self.log_dir = Path(log_dir) if log_dir else Path(config['logging'].get('log_dir', 'logs'))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_iou': [],
            'lr': []
        }

        # Best metrics
        self.best_val_loss = float('inf')
        self.best_val_iou = 0.0
        self.best_epoch = 0

        # Early stopping
        self.patience = config['train'].get('patience', 15)
        self.use_early_stopping = config['train'].get('early_stopping', False)
        self.counter = 0
        self.early_stop = False

        # TensorBoard
        if config['logging'].get('tensorboard', False):
            self.writer = SummaryWriter(str(self.log_dir / 'tensorboard'))
        else:
            self.writer = None

        # Weights & Biases
        if config['logging'].get('wandb', False):
            try:
                import wandb
                wandb.init(project=config['logging'].get('wandb_project', 'layout_estimation'))
                wandb.config.update(config)
                self.use_wandb = True
            except ImportError:
                self.logger.warning("Weights & Biases (wandb) not installed. Skipping wandb logging.")
                self.use_wandb = False
        else:
            self.use_wandb = False

        # Start time
        self.start_time = time.time()

    def setup_logging(self):
        """Setup logger for printing and saving logs."""
        self.logger = logging.getLogger('TrainingMonitor')
        self.logger.setLevel(logging.INFO)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)

        # Create file handler
        file_handler = logging.FileHandler(str(self.log_dir / 'training.log'))
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)

        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        self.logger.info(f"Log directory: {self.log_dir}")
        self.logger.info(f"Configuration: {self.config}")

    def update(self, epoch, train_metrics, val_metrics=None, learning_rate=None):
        """
        Update training history and check for early stopping.

        Args:
            epoch (int): Current epoch
            train_metrics (dict): Training metrics
            val_metrics (dict, optional): Validation metrics
            learning_rate (float, optional): Current learning rate

        Returns:
            bool: Whether to early stop
        """
        # Update history
        self.history['train_loss'].append(train_metrics['train_loss'])

        if val_metrics:
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['val_iou'].append(val_metrics['val_iou'])

            # Check for best model
            if val_metrics['val_iou'] > self.best_val_iou:
                self.best_val_iou = val_metrics['val_iou']
                self.best_epoch = epoch
                self.counter = 0
                return False  # Don't early stop

            # Early stopping
            if self.use_early_stopping:
                if val_metrics['val_iou'] < self.best_val_iou:
                    self.counter += 1
                    if self.counter >= self.patience:
                        self.early_stop = True
                        self.logger.info(f"Early stopping triggered after {epoch} epochs")
                        return True  # Early stop

        # Update learning rate history
        if learning_rate is not None:
            self.history['lr'].append(learning_rate)

        # Log to TensorBoard
        if self.writer:
            # Log training metrics
            for key, value in train_metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'train/{key}', value, epoch)

            # Log validation metrics
            if val_metrics:
                for key, value in val_metrics.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(f'val/{key}', value, epoch)

            # Log learning rate
            if learning_rate is not None:
                self.writer.add_scalar('train/lr', learning_rate, epoch)

        # Log to Weights & Biases
        if self.use_wandb:
            import wandb
            log_dict = {}

            # Add training metrics
            for key, value in train_metrics.items():
                if isinstance(value, (int, float)):
                    log_dict[f'train/{key}'] = value

            # Add validation metrics
            if val_metrics:
                for key, value in val_metrics.items():
                    if isinstance(value, (int, float)):
                        log_dict[f'val/{key}'] = value

            # Add learning rate
            if learning_rate is not None:
                log_dict['train/lr'] = learning_rate

            # Log to wandb
            wandb.log(log_dict, step=epoch)

        # Don't early stop
        return False

    def log_epoch(self, epoch, train_metrics, val_metrics=None, learning_rate=None, time_taken=None):
        """
        Log metrics for current epoch.

        Args:
            epoch (int): Current epoch
            train_metrics (dict): Training metrics
            val_metrics (dict, optional): Validation metrics
            learning_rate (float, optional): Current learning rate
            time_taken (float, optional): Time taken for epoch
        """
        # Create log message
        msg = f"Epoch {epoch}"

        # Add training metrics
        msg += f" | Train Loss: {train_metrics['train_loss']:.4f}"

        # Add validation metrics
        if val_metrics:
            msg += f" | Val Loss: {val_metrics['val_loss']:.4f}"
            msg += f" | Val IoU: {val_metrics['val_iou']:.4f}"

            # Add IoU accuracy metrics
            for threshold in [0.25, 0.5, 0.75]:
                if f'iou_{threshold}' in val_metrics:
                    msg += f" | IoU@{threshold}: {val_metrics[f'iou_{threshold}']:.4f}"

        # Add learning rate
        if learning_rate is not None:
            msg += f" | LR: {learning_rate:.6f}"

        # Add time taken
        if time_taken is not None:
            msg += f" | Time: {time_taken:.2f}s"

        # Log message
        self.logger.info(msg)

    def plot_history(self, save_path=None):
        """
        Plot training history.

        Args:
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(15, 10))

        # Plot training and validation loss
        plt.subplot(2, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        if self.history['val_loss']:
            plt.plot(self.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)

        # Plot validation IoU
        if self.history['val_iou']:
            plt.subplot(2, 2, 2)
            plt.plot(self.history['val_iou'], label='Val IoU')
            plt.axhline(y=0.25, color='r', linestyle='--', label='IoU=0.25')
            plt.axhline(y=0.5, color='g', linestyle='--', label='IoU=0.5')
            plt.xlabel('Epoch')
            plt.ylabel('IoU')
            plt.title('Validation IoU')
            plt.legend()
            plt.grid(True)

        # Plot learning rate
        if self.history['lr']:
            plt.subplot(2, 2, 3)
            plt.plot(self.history['lr'])
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.grid(True)

        plt.tight_layout()

        # Save plot
        if save_path:
            plt.savefig(save_path)
        else:
            plt.savefig(str(self.log_dir / 'training_history.png'))

        plt.close()

    def save_history(self, save_path=None):
        """
        Save training history to file.

        Args:
            save_path (str, optional): Path to save the history
        """
        history = {
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_val_iou': self.best_val_iou,
            'best_epoch': self.best_epoch
        }

        if save_path:
            torch.save(history, save_path)
        else:
            torch.save(history, str(self.log_dir / 'training_history.pth'))

    def close(self):
        """Clean up resources."""
        if self.writer:
            self.writer.close()

        if self.use_wandb:
            import wandb
            wandb.finish()

        # Plot and save history
        self.plot_history()
        self.save_history()

        # Log total training time
        total_time = time.time() - self.start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.logger.info(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")


def save_training_checkpoint(model, optimizer, scheduler, epoch, monitor, save_path):
    """
    Save training checkpoint.

    Args:
        model (nn.Module): Model
        optimizer (optim.Optimizer): Optimizer
        scheduler (optim.lr_scheduler._LRScheduler): Learning rate scheduler
        epoch (int): Current epoch
        monitor (TrainingMonitor): Training monitor
        save_path (str): Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_loss': monitor.best_val_loss,
        'best_val_iou': monitor.best_val_iou,
        'best_epoch': monitor.best_epoch,
        'history': monitor.history
    }

    torch.save(checkpoint, save_path)


def load_training_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='mps'):
    """
    Load training checkpoint.

    Args:
        checkpoint_path (str): Path to checkpoint
        model (nn.Module): Model
        optimizer (optim.Optimizer, optional): Optimizer
        scheduler (optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler
        device (str): Device to load checkpoint to

    Returns:
        tuple: (start_epoch, best_val_loss, best_val_iou, best_epoch, history)
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return 0, float('inf'), 0.0, 0, {}

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Extract training state
    epoch = checkpoint.get('epoch', -1) + 1  # Start from next epoch
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    best_val_iou = checkpoint.get('best_val_iou', 0.0)
    best_epoch = checkpoint.get('best_epoch', 0)
    history = checkpoint.get('history', {})

    return epoch, best_val_loss, best_val_iou, best_epoch, history