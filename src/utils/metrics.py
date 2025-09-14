"""
Metrics for segmentation model evaluation.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from sklearn.metrics import confusion_matrix, classification_report
import cv2

logger = logging.getLogger(__name__)


class SegmentationMetrics:
    """Metrics calculator for segmentation tasks."""
    
    def __init__(self, num_classes: int = 4, class_names: Optional[List[str]] = None):
        """
        Initialize metrics calculator.
        
        Args:
            num_classes: Number of classes
            class_names: Names of classes
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]
        
    def calculate_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Calculate segmentation metrics.
        
        Args:
            predictions: Model predictions of shape (B, H, W) or (B, C, H, W)
            targets: Ground truth labels of shape (B, H, W)
            
        Returns:
            Dictionary of metrics
        """
        # Ensure predictions are in correct format
        if len(predictions.shape) == 4:  # (B, C, H, W)
            predictions = torch.argmax(predictions, dim=1)  # (B, H, W)
        
        # Convert to numpy
        pred_np = predictions.cpu().numpy()
        target_np = targets.cpu().numpy()
        
        # Flatten for easier calculation
        pred_flat = pred_np.flatten()
        target_flat = target_np.flatten()
        
        # Calculate metrics
        metrics = {}
        
        # Pixel accuracy
        metrics['pixel_accuracy'] = self._calculate_pixel_accuracy(pred_flat, target_flat)
        
        # Mean IoU
        metrics['miou'] = self._calculate_miou(pred_np, target_np)
        
        # Class-wise IoU
        class_iou = self._calculate_class_iou(pred_np, target_np)
        for i, iou in enumerate(class_iou):
            metrics[f'iou_class_{i}'] = iou
            if i < len(self.class_names):
                metrics[f'iou_{self.class_names[i]}'] = iou
        
        # Dice score
        metrics['dice_score'] = self._calculate_dice_score(pred_np, target_np)
        
        # Boundary F1 score
        metrics['boundary_f1'] = self._calculate_boundary_f1(pred_np, target_np)
        
        return metrics
    
    def _calculate_pixel_accuracy(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Calculate pixel accuracy."""
        correct = (pred == target).sum()
        total = len(pred)
        return correct / total if total > 0 else 0.0
    
    def _calculate_miou(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Calculate mean Intersection over Union."""
        ious = []
        
        for class_id in range(self.num_classes):
            pred_mask = (pred == class_id)
            target_mask = (target == class_id)
            
            intersection = (pred_mask & target_mask).sum()
            union = (pred_mask | target_mask).sum()
            
            if union > 0:
                iou = intersection / union
                ious.append(iou)
            else:
                ious.append(0.0)
        
        return np.mean(ious)
    
    def _calculate_class_iou(self, pred: np.ndarray, target: np.ndarray) -> List[float]:
        """Calculate IoU for each class."""
        ious = []
        
        for class_id in range(self.num_classes):
            pred_mask = (pred == class_id)
            target_mask = (target == class_id)
            
            intersection = (pred_mask & target_mask).sum()
            union = (pred_mask | target_mask).sum()
            
            if union > 0:
                iou = intersection / union
                ious.append(iou)
            else:
                ious.append(0.0)
        
        return ious
    
    def _calculate_dice_score(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Calculate Dice score."""
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        intersection = (pred_flat == target_flat).sum()
        total = len(pred_flat)
        
        return 2.0 * intersection / total if total > 0 else 0.0
    
    def _calculate_boundary_f1(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Calculate Boundary F1 score."""
        f1_scores = []
        
        for i in range(pred.shape[0]):  # For each image in batch
            pred_boundary = self._extract_boundary(pred[i])
            target_boundary = self._extract_boundary(target[i])
            
            if pred_boundary.sum() == 0 and target_boundary.sum() == 0:
                f1_scores.append(1.0)
                continue
            
            # Calculate precision and recall
            intersection = (pred_boundary & target_boundary).sum()
            precision = intersection / pred_boundary.sum() if pred_boundary.sum() > 0 else 0.0
            recall = intersection / target_boundary.sum() if target_boundary.sum() > 0 else 0.0
            
            # Calculate F1 score
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                f1_scores.append(f1)
            else:
                f1_scores.append(0.0)
        
        return np.mean(f1_scores)
    
    def _extract_boundary(self, mask: np.ndarray) -> np.ndarray:
        """Extract boundary from segmentation mask."""
        # Convert to uint8
        mask_uint8 = mask.astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create boundary mask
        boundary = np.zeros_like(mask_uint8)
        cv2.drawContours(boundary, contours, -1, 1, 1)
        
        return boundary.astype(bool)
    
    def aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics from multiple batches."""
        if not metrics_list:
            return {}
        
        aggregated = {}
        
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if key in m]
            aggregated[key] = np.mean(values)
        
        return aggregated
    
    def calculate_confusion_matrix(self, predictions: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
        """Calculate confusion matrix."""
        if len(predictions.shape) == 4:
            predictions = torch.argmax(predictions, dim=1)
        
        pred_np = predictions.cpu().numpy().flatten()
        target_np = targets.cpu().numpy().flatten()
        
        return confusion_matrix(target_np, pred_np, labels=range(self.num_classes))
    
    def generate_classification_report(self, predictions: torch.Tensor, targets: torch.Tensor) -> str:
        """Generate classification report."""
        if len(predictions.shape) == 4:
            predictions = torch.argmax(predictions, dim=1)
        
        pred_np = predictions.cpu().numpy().flatten()
        target_np = targets.cpu().numpy().flatten()
        
        return classification_report(
            target_np, pred_np, 
            target_names=self.class_names,
            labels=range(self.num_classes),
            zero_division=0
        )


class ModelEvaluator:
    """Comprehensive model evaluator."""
    
    def __init__(self, model: torch.nn.Module, device: str = 'cuda'):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model
            device: Device to use for evaluation
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def evaluate_dataset(self, data_loader, metrics_calculator: SegmentationMetrics) -> Dict[str, Any]:
        """
        Evaluate model on entire dataset.
        
        Args:
            data_loader: Data loader for evaluation
            metrics_calculator: Metrics calculator
            
        Returns:
            Evaluation results
        """
        all_metrics = []
        
        with torch.no_grad():
            for images, targets in data_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Get predictions
                outputs = self.model(images)
                predictions = outputs['predictions']
                
                # Calculate metrics
                metrics = metrics_calculator.calculate_metrics(predictions, targets)
                all_metrics.append(metrics)
        
        # Aggregate metrics
        aggregated_metrics = metrics_calculator.aggregate_metrics(all_metrics)
        
        return {
            'metrics': aggregated_metrics,
            'individual_metrics': all_metrics
        }
    
    def predict_single_image(self, image: torch.Tensor, return_uncertainty: bool = False) -> Dict[str, Any]:
        """
        Predict on a single image.
        
        Args:
            image: Input image tensor
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Prediction results
        """
        self.model.eval()
        
        with torch.no_grad():
            if len(image.shape) == 3:
                image = image.unsqueeze(0)  # Add batch dimension
            
            image = image.to(self.device)
            
            # Get prediction
            outputs = self.model(image, return_uncertainty=return_uncertainty)
            
            return {
                'predictions': outputs['predictions'].cpu(),
                'probabilities': outputs['probabilities'].cpu(),
                'logits': outputs['logits'].cpu(),
                'uncertainty': outputs.get('uncertainty', None)
            }
