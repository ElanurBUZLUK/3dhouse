"""
Training module for segmentation models.
Handles training, validation, and model checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
import os
import json
from pathlib import Path
from tqdm import tqdm
import time
from sklearn.metrics import confusion_matrix, classification_report
import wandb

from .segmentation_model import SegmentationModel, SegmentationLoss, create_segmentation_model, create_segmentation_loss
from ..utils.metrics import SegmentationMetrics
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class SegmentationTrainer:
    """Trainer for segmentation models."""
    
    def __init__(self, 
                 model: SegmentationModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict[str, Any],
                 device: str = 'cuda',
                 log_dir: str = 'logs',
                 use_wandb: bool = False):
        """
        Initialize the trainer.
        
        Args:
            model: Segmentation model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to use for training
            log_dir: Directory for logging
            use_wandb: Whether to use Weights & Biases logging
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.log_dir = Path(log_dir)
        self.use_wandb = use_wandb
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.log_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.loss_fn = create_segmentation_loss(config['loss'])
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.metrics = SegmentationMetrics(num_classes=config['model']['num_classes'])
        
        # Initialize logging
        self.writer = SummaryWriter(self.log_dir / 'tensorboard')
        if self.use_wandb:
            wandb.init(project='sketch2house3d', config=config)
        
        # Training state
        self.current_epoch = 0
        self.best_val_miou = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        
        # Mixed precision training
        self.use_amp = config.get('train', {}).get('mixed_precision', False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer from config."""
        optimizer_config = self.config['optimizer']
        optimizer_type = optimizer_config['type'].lower()
        
        if optimizer_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config.get('weight_decay', 0.0)
            )
        elif optimizer_type == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config.get('weight_decay', 0.01)
            )
        elif optimizer_type == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=optimizer_config['lr'],
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=optimizer_config.get('weight_decay', 0.0)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler from config."""
        scheduler_config = self.config.get('scheduler', {})
        if not scheduler_config:
            return None
        
        scheduler_type = scheduler_config['type'].lower()
        
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['train']['epochs'],
                eta_min=scheduler_config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 10),
                min_lr=scheduler_config.get('min_lr', 1e-6)
            )
        else:
            return None
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        epoch_metrics = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1}')
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    losses = self.loss_fn(outputs['logits'], targets)
                    loss = losses['total_loss']
            else:
                outputs = self.model(images)
                losses = self.loss_fn(outputs['logits'], targets)
                loss = losses['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Calculate metrics
            predictions = outputs['predictions']
            metrics = self.metrics.calculate_metrics(predictions, targets)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'miou': f'{metrics["miou"]:.4f}'
            })
            
            # Log losses
            epoch_losses.append(loss.item())
            epoch_metrics.append(metrics)
            
            # Log to tensorboard
            if batch_idx % self.config.get('logging', {}).get('log_every_n_steps', 50) == 0:
                global_step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), global_step)
                self.writer.add_scalar('Train/mIoU', metrics['miou'], global_step)
                
                if self.use_wandb:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/miou': metrics['miou'],
                        'epoch': self.current_epoch,
                        'step': global_step
                    })
        
        # Calculate epoch averages
        avg_loss = np.mean(epoch_losses)
        avg_metrics = self.metrics.aggregate_metrics(epoch_metrics)
        
        return {
            'loss': avg_loss,
            **avg_metrics
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_losses = []
        epoch_metrics = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            
            for images, targets in pbar:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        losses = self.loss_fn(outputs['logits'], targets)
                        loss = losses['total_loss']
                else:
                    outputs = self.model(images)
                    losses = self.loss_fn(outputs['logits'], targets)
                    loss = losses['total_loss']
                
                # Calculate metrics
                predictions = outputs['predictions']
                metrics = self.metrics.calculate_metrics(predictions, targets)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'miou': f'{metrics["miou"]:.4f}'
                })
                
                epoch_losses.append(loss.item())
                epoch_metrics.append(metrics)
        
        # Calculate epoch averages
        avg_loss = np.mean(epoch_losses)
        avg_metrics = self.metrics.aggregate_metrics(epoch_metrics)
        
        return {
            'loss': avg_loss,
            **avg_metrics
        }
    
    def train(self) -> Dict[str, Any]:
        """Train the model."""
        logger.info(f"Starting training for {self.config['train']['epochs']} epochs")
        
        start_time = time.time()
        
        for epoch in range(self.config['train']['epochs']):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['loss'])
            
            # Validation
            val_metrics = self.validate_epoch()
            self.val_losses.append(val_metrics['loss'])
            self.val_metrics.append(val_metrics)
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['miou'])
                else:
                    self.scheduler.step()
            
            # Log epoch results
            logger.info(f"Epoch {epoch+1}/{self.config['train']['epochs']}: "
                       f"Train Loss: {train_metrics['loss']:.4f}, "
                       f"Val Loss: {val_metrics['loss']:.4f}, "
                       f"Val mIoU: {val_metrics['miou']:.4f}")
            
            # Log to tensorboard
            self.writer.add_scalar('Epoch/Train_Loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('Epoch/Val_Loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('Epoch/Val_mIoU', val_metrics['miou'], epoch)
            
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_metrics['loss'],
                    'val/loss': val_metrics['loss'],
                    'val/miou': val_metrics['miou'],
                    'val/pixel_accuracy': val_metrics['pixel_accuracy'],
                    'val/dice_score': val_metrics['dice_score']
                })
            
            # Save checkpoint
            if val_metrics['miou'] > self.best_val_miou:
                self.best_val_miou = val_metrics['miou']
                self.save_checkpoint(epoch, val_metrics, is_best=True)
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.get('checkpoint', {}).get('save_every_n_epochs', 10) == 0:
                self.save_checkpoint(epoch, val_metrics, is_best=False)
        
        # Training completed
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Best validation mIoU: {self.best_val_miou:.4f}")
        
        # Close logging
        self.writer.close()
        if self.use_wandb:
            wandb.finish()
        
        return {
            'best_val_miou': self.best_val_miou,
            'training_time': training_time,
            'final_epoch': self.current_epoch,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with mIoU: {metrics['miou']:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint


def train_segmentation_model(config_path: str, 
                           train_loader: DataLoader,
                           val_loader: DataLoader,
                           device: str = 'cuda',
                           log_dir: str = 'logs',
                           use_wandb: bool = False) -> Dict[str, Any]:
    """
    Train a segmentation model from configuration.
    
    Args:
        config_path: Path to configuration file
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to use for training
        log_dir: Directory for logging
        use_wandb: Whether to use Weights & Biases logging
        
    Returns:
        Training results dictionary
    """
    # Load configuration
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = create_segmentation_model(config['model'])
    
    # Create trainer
    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        log_dir=log_dir,
        use_wandb=use_wandb
    )
    
    # Train model
    results = trainer.train()
    
    return results
