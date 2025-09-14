"""
Logging utilities for Sketch2House3D.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import datetime


def setup_logger(name: str, 
                level: int = logging.INFO,
                log_file: Optional[str] = None,
                format_string: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with console and optional file output.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        format_string: Optional custom format string
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a new one.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class TrainingLogger:
    """Logger for training progress and metrics."""
    
    def __init__(self, log_dir: str, experiment_name: str = None):
        """
        Initialize training logger.
        
        Args:
            log_dir: Directory for log files
            experiment_name: Name of the experiment
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name or f"experiment_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir = self.log_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = setup_logger(
            f"training_{self.experiment_name}",
            log_file=str(self.experiment_dir / "training.log")
        )
        
        # Metrics storage
        self.metrics_history = []
    
    def log_epoch(self, epoch: int, train_metrics: dict, val_metrics: dict):
        """Log epoch results."""
        self.logger.info(f"Epoch {epoch}:")
        self.logger.info(f"  Train - Loss: {train_metrics.get('loss', 0):.4f}, "
                        f"mIoU: {train_metrics.get('miou', 0):.4f}")
        self.logger.info(f"  Val   - Loss: {val_metrics.get('loss', 0):.4f}, "
                        f"mIoU: {val_metrics.get('miou', 0):.4f}")
        
        # Store metrics
        self.metrics_history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics
        })
    
    def log_best_model(self, epoch: int, metrics: dict):
        """Log best model achievement."""
        self.logger.info(f"New best model at epoch {epoch} with mIoU: {metrics.get('miou', 0):.4f}")
    
    def log_training_complete(self, total_time: float, best_metrics: dict):
        """Log training completion."""
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        self.logger.info(f"Best validation mIoU: {best_metrics.get('miou', 0):.4f}")
    
    def save_metrics(self):
        """Save metrics to file."""
        import json
        
        metrics_file = self.experiment_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        self.logger.info(f"Metrics saved to {metrics_file}")


def log_model_info(model: object, logger: logging.Logger):
    """Log model information."""
    if hasattr(model, 'get_model_info'):
        info = model.get_model_info()
        logger.info("Model Information:")
        for key, value in info.items():
            logger.info(f"  {key}: {value}")
    else:
        logger.info(f"Model: {type(model).__name__}")


def log_config(config: dict, logger: logging.Logger):
    """Log configuration."""
    logger.info("Configuration:")
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"    {sub_key}: {sub_value}")
        else:
            logger.info(f"  {key}: {value}")


def log_system_info(logger: logging.Logger):
    """Log system information."""
    import torch
    import platform
    
    logger.info("System Information:")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Python: {platform.python_version()}")
    logger.info(f"  PyTorch: {torch.__version__}")
    logger.info(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  CUDA version: {torch.version.cuda}")
        logger.info(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
