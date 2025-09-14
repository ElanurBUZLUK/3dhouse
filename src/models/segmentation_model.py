"""
Segmentation model for architectural element detection in child sketches.
Supports U-Net and DeepLabV3+ architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
import logging
import timm
import segmentation_models_pytorch as smp
from torchvision import transforms

logger = logging.getLogger(__name__)


class SegmentationModel(nn.Module):
    """Segmentation model for architectural element detection."""
    
    def __init__(self, 
                 architecture: str = 'deeplabv3plus',
                 encoder: str = 'mobilenetv3_large_100',
                 num_classes: int = 4,
                 input_size: Tuple[int, int] = (256, 256),
                 pretrained: bool = True,
                 activation: str = 'softmax',
                 in_channels: int = 3):
        """
        Initialize the segmentation model.
        
        Args:
            architecture: Model architecture ('unet', 'deeplabv3plus', 'fpn', 'pspnet')
            encoder: Encoder backbone ('resnet18', 'mobilenetv3_large_100', 'efficientnet-b0', etc.)
            num_classes: Number of segmentation classes
            input_size: Input image size (height, width)
            pretrained: Whether to use pretrained weights
            activation: Output activation ('softmax', 'sigmoid', 'none')
        """
        super().__init__()
        
        self.architecture = architecture
        self.encoder = encoder
        self.num_classes = num_classes
        self.input_size = input_size
        self.activation = activation
        self.in_channels = in_channels
        
        # Create the model
        self.model = self._create_model(architecture, encoder, num_classes, pretrained)
        
        # Add uncertainty estimation head if needed
        self.uncertainty_head = None
        self.dropout_rate = 0.1
        
    def _create_model(self, architecture: str, encoder: str, 
                     num_classes: int, pretrained: bool) -> nn.Module:
        """Create the segmentation model."""
        if architecture == 'unet':
            return smp.Unet(
                encoder_name=encoder,
                encoder_weights='imagenet' if pretrained else None,
                in_channels=self.in_channels,
                classes=num_classes,
                activation=None  # We'll handle activation separately
            )
        elif architecture == 'deeplabv3plus':
            return smp.DeepLabV3Plus(
                encoder_name=encoder,
                encoder_weights='imagenet' if pretrained else None,
                in_channels=self.in_channels,
                classes=num_classes,
                activation=None
            )
        elif architecture == 'fpn':
            return smp.FPN(
                encoder_name=encoder,
                encoder_weights='imagenet' if pretrained else None,
                in_channels=self.in_channels,
                classes=num_classes,
                activation=None
            )
        elif architecture == 'pspnet':
            return smp.PSPNet(
                encoder_name=encoder,
                encoder_weights='imagenet' if pretrained else None,
                in_channels=self.in_channels,
                classes=num_classes,
                activation=None
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
    
    def forward(self, x: torch.Tensor, return_uncertainty: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Dictionary containing:
                - logits: Raw model output
                - predictions: Class predictions
                - probabilities: Class probabilities
                - uncertainty: Uncertainty estimates (if requested)
        """
        # Forward pass through the model
        logits = self.model(x)
        
        # Apply activation
        if self.activation == 'softmax':
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        elif self.activation == 'sigmoid':
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).long()
        else:
            probabilities = logits
            predictions = torch.argmax(logits, dim=1)
        
        result = {
            'logits': logits,
            'predictions': predictions,
            'probabilities': probabilities
        }
        
        # Add uncertainty estimation if requested
        if return_uncertainty:
            uncertainty = self._estimate_uncertainty(x, logits)
            result['uncertainty'] = uncertainty
        
        return result
    
    def _estimate_uncertainty(self, x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Estimate uncertainty using Monte Carlo dropout."""
        if self.uncertainty_head is None:
            # Use simple entropy-based uncertainty
            probabilities = F.softmax(logits, dim=1)
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1)
            return entropy
        
        # Monte Carlo dropout for uncertainty estimation
        self.train()  # Enable dropout
        uncertainties = []
        
        with torch.no_grad():
            for _ in range(10):  # 10 Monte Carlo samples
                mc_logits = self.model(x)
                mc_probabilities = F.softmax(mc_logits, dim=1)
                uncertainties.append(mc_probabilities)
        
        # Calculate uncertainty as variance across samples
        uncertainties = torch.stack(uncertainties, dim=0)  # (n_samples, B, C, H, W)
        mean_probabilities = torch.mean(uncertainties, dim=0)
        variance = torch.var(uncertainties, dim=0)
        uncertainty = torch.sum(variance, dim=1)  # Sum over classes
        
        self.eval()  # Disable dropout
        return uncertainty
    
    def predict(self, x: torch.Tensor, 
                return_uncertainty: bool = False) -> Dict[str, torch.Tensor]:
        """
        Make predictions on input tensor.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Prediction results dictionary
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x, return_uncertainty)
    
    def predict_from_image(self, image: torch.Tensor, 
                          preprocess: bool = True) -> Dict[str, torch.Tensor]:
        """
        Make predictions from a single image.
        
        Args:
            image: Input image tensor of shape (C, H, W) or (H, W, C)
            preprocess: Whether to preprocess the image
            
        Returns:
            Prediction results dictionary
        """
        # Ensure image is in correct format
        if len(image.shape) == 3:
            if image.shape[0] == 3:  # (C, H, W)
                image = image.unsqueeze(0)  # Add batch dimension
            else:  # (H, W, C)
                image = image.permute(2, 0, 1).unsqueeze(0)
        elif len(image.shape) == 2:  # (H, W)
            image = image.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        
        # Preprocess if requested
        if preprocess:
            image = self._preprocess_image(image)
        
        # Make prediction
        return self.predict(image)
    
    def _preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """Preprocess image for model input.

        - Scales to [0,1] if necessary
        - Resizes to `input_size`
        - Applies ImageNet normalization to first 3 channels
        - Keeps extra channels (e.g., edge) as-is in [0,1]
        """
        # Normalize to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0

        # Resize to model input size
        if image.shape[-2:] != self.input_size:
            image = F.interpolate(image, size=self.input_size, mode='bilinear', align_corners=False)

        # Split into RGB and extra channels if any
        c = image.shape[1]
        rgb = image[:, :3, :, :] if c >= 3 else image
        extra = image[:, 3:, :, :] if c > 3 else None

        # Normalize RGB with ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(image.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(image.device)
        if rgb.shape[1] == 3:
            rgb = (rgb - mean) / std

        # Concatenate back extra channels if present
        if extra is not None and extra.numel() > 0:
            image = torch.cat([rgb, extra], dim=1)
        else:
            image = rgb

        return image

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'architecture': self.architecture,
            'encoder': self.encoder,
            'num_classes': self.num_classes,
            'input_size': self.input_size,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'activation': self.activation
        }


class SegmentationLoss(nn.Module):
    """Combined loss function for segmentation."""
    
    def __init__(self, 
                 num_classes: int = 4,
                 class_weights: Optional[List[float]] = None,
                 ce_weight: float = 0.5,
                 dice_weight: float = 0.5,
                 focal_weight: float = 0.0,
                 label_smoothing: float = 0.0):
        """
        Initialize the segmentation loss.
        
        Args:
            num_classes: Number of classes
            class_weights: Class weights for cross-entropy loss
            ce_weight: Weight for cross-entropy loss
            dice_weight: Weight for dice loss
            focal_weight: Weight for focal loss
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.label_smoothing = label_smoothing
        
        # Cross-entropy loss
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        
        # Dice loss
        self.dice_loss = smp.losses.DiceLoss(mode='multiclass', classes=list(range(num_classes)))
        
        # Focal loss (optional)
        if focal_weight > 0:
            self.focal_loss = smp.losses.FocalLoss(mode='multiclass', gamma=2.0)
        else:
            self.focal_loss = None
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate the combined loss.
        
        Args:
            predictions: Model predictions of shape (B, C, H, W)
            targets: Ground truth labels of shape (B, H, W)
            
        Returns:
            Dictionary containing individual and total losses
        """
        losses = {}
        
        # Cross-entropy loss
        if self.ce_weight > 0:
            ce_loss = self.ce_loss(predictions, targets) * self.ce_weight
            losses['ce_loss'] = ce_loss
        
        # Dice loss
        if self.dice_weight > 0:
            dice_loss = self.dice_loss(predictions, targets) * self.dice_weight
            losses['dice_loss'] = dice_loss
        
        # Focal loss
        if self.focal_weight > 0 and self.focal_loss is not None:
            focal_loss = self.focal_loss(predictions, targets) * self.focal_weight
            losses['focal_loss'] = focal_loss
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        return losses


def create_segmentation_model(config: Dict[str, Any]) -> SegmentationModel:
    """
    Create a segmentation model from configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized segmentation model
    """
    return SegmentationModel(
        architecture=config.get('architecture', 'deeplabv3plus'),
        encoder=config.get('encoder', 'mobilenetv3_large_100'),
        num_classes=config.get('num_classes', 4),
        input_size=tuple(config.get('input_size', [256, 256])),
        pretrained=config.get('pretrained', True),
        activation=config.get('activation', 'softmax'),
        in_channels=int(config.get('in_channels', 3))
    )


def create_segmentation_loss(config: Dict[str, Any]) -> SegmentationLoss:
    """
    Create a segmentation loss function from configuration.
    
    Args:
        config: Loss configuration dictionary
        
    Returns:
        Initialized loss function
    """
    return SegmentationLoss(
        num_classes=config.get('num_classes', 4),
        class_weights=config.get('class_weights'),
        ce_weight=config.get('ce_weight', 0.5),
        dice_weight=config.get('dice_weight', 0.5),
        focal_weight=config.get('focal_weight', 0.0),
        label_smoothing=config.get('label_smoothing', 0.0)
    )
