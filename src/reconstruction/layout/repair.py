"""
Constraint repair module for fixing violated architectural constraints.
Implements various repair strategies to make layouts valid.
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Optional
import logging
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import copy

logger = logging.getLogger(__name__)


class ConstraintRepair:
    """Repairs violated architectural constraints."""
    
    def __init__(self, 
                 max_repair_iterations: int = 10,
                 repair_tolerance: float = 1e-6,
                 min_opening_size: int = 20):
        """
        Initialize constraint repair.
        
        Args:
            max_repair_iterations: Maximum number of repair iterations
            repair_tolerance: Tolerance for convergence
            min_opening_size: Minimum size for openings in pixels
        """
        self.max_repair_iterations = max_repair_iterations
        self.repair_tolerance = repair_tolerance
        self.min_opening_size = min_opening_size
    
    def repair_layout(self, layout_params: Dict[str, Any], 
                     constraint_violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Repair layout to fix constraint violations.
        
        Args:
            layout_params: Original layout parameters
            constraint_violations: List of constraint violations
            
        Returns:
            Repaired layout parameters
        """
        repaired_layout = copy.deepcopy(layout_params)
        
        # Group violations by type
        violation_groups = self._group_violations(constraint_violations)
        
        # Apply repair strategies
        for violation_type, violations in violation_groups.items():
            if violation_type == 'opening_within_wall':
                repaired_layout = self._repair_opening_within_wall(repaired_layout, violations)
            elif violation_type == 'no_overlapping_openings':
                repaired_layout = self._repair_overlapping_openings(repaired_layout, violations)
            elif violation_type == 'door_touches_ground':
                repaired_layout = self._repair_door_touches_ground(repaired_layout, violations)
            elif violation_type == 'window_height_constraint':
                repaired_layout = self._repair_window_height(repaired_layout, violations)
            elif violation_type == 'opening_size_constraints':
                repaired_layout = self._repair_opening_sizes(repaired_layout, violations)
            elif violation_type == 'facade_proportions':
                repaired_layout = self._repair_facade_proportions(repaired_layout, violations)
        
        # Validate repaired layout
        from .constraints import validate_layout_constraints
        validation_result = validate_layout_constraints(repaired_layout)
        
        repaired_layout['repair_metadata'] = {
            'original_violations': len(constraint_violations),
            'remaining_violations': len(validation_result['violations']),
            'repair_success': validation_result['valid'],
            'constraint_score': validation_result['score']
        }
        
        return repaired_layout
    
    def _group_violations(self, violations: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group violations by constraint type."""
        groups = {}
        for violation in violations:
            constraint_type = violation['constraint']
            if constraint_type not in groups:
                groups[constraint_type] = []
            groups[constraint_type].append(violation)
        return groups
    
    def _repair_opening_within_wall(self, layout: Dict[str, Any], 
                                   violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Repair openings that are outside wall boundaries."""
        facade_polygon = layout.get('facade_polygon', [])
        openings = layout.get('openings', [])
        
        if not facade_polygon or not openings:
            return layout
        
        facade_polygon_np = np.array(facade_polygon)
        
        repaired_openings = []
        for opening in openings:
            bbox = opening.get('bbox_px', [])
            if len(bbox) != 4:
                repaired_openings.append(opening)
                continue
            
            # Check if opening is within facade
            if self._is_opening_within_facade(bbox, facade_polygon_np):
                repaired_openings.append(opening)
            else:
                # Try to move opening inside facade
                repaired_bbox = self._move_opening_inside_facade(bbox, facade_polygon_np)
                if repaired_bbox:
                    opening['bbox_px'] = repaired_bbox
                    opening['center_px'] = [
                        (repaired_bbox[0] + repaired_bbox[2]) / 2,
                        (repaired_bbox[1] + repaired_bbox[3]) / 2
                    ]
                    repaired_openings.append(opening)
                else:
                    # If can't repair, remove opening
                    logger.warning(f"Removing opening that cannot be moved inside facade: {opening}")
        
        layout['openings'] = repaired_openings
        return layout
    
    def _is_opening_within_facade(self, bbox: List[float], 
                                 facade_polygon: np.ndarray) -> bool:
        """Check if opening is within facade polygon."""
        x_min, y_min, x_max, y_max = bbox
        
        # Check center point
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        return cv2.pointPolygonTest(facade_polygon, (center_x, center_y), False) >= 0
    
    def _move_opening_inside_facade(self, bbox: List[float], 
                                   facade_polygon: np.ndarray) -> Optional[List[float]]:
        """Move opening inside facade polygon."""
        x_min, y_min, x_max, y_max = bbox
        w = x_max - x_min
        h = y_max - y_min
        
        # Try different positions
        for offset_x in range(-int(w), int(w) + 1, 10):
            for offset_y in range(-int(h), int(h) + 1, 10):
                new_bbox = [x_min + offset_x, y_min + offset_y, 
                           x_min + offset_x + w, y_min + offset_y + h]
                
                if self._is_opening_within_facade(new_bbox, facade_polygon):
                    return new_bbox
        
        return None
    
    def _repair_overlapping_openings(self, layout: Dict[str, Any], 
                                   violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Repair overlapping openings using non-maximum suppression."""
        openings = layout.get('openings', [])
        
        if len(openings) < 2:
            return layout
        
        # Apply non-maximum suppression
        filtered_openings = self._apply_nms(openings, iou_threshold=0.3)
        
        layout['openings'] = filtered_openings
        return layout
    
    def _apply_nms(self, openings: List[Dict[str, Any]], 
                  iou_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Apply non-maximum suppression to remove overlapping openings."""
        if not openings:
            return openings
        
        # Sort by confidence
        openings.sort(key=lambda x: x.get('confidence', 0.0), reverse=True)
        
        # Calculate IoU matrix
        bboxes = [opening.get('bbox_px', []) for opening in openings]
        iou_matrix = self._calculate_iou_matrix(bboxes)
        
        # Apply NMS
        keep_indices = []
        sorted_indices = list(range(len(openings)))
        
        while sorted_indices:
            # Keep the highest scoring opening
            current_idx = sorted_indices[0]
            keep_indices.append(current_idx)
            
            # Remove openings with high IoU
            remaining_indices = []
            for idx in sorted_indices[1:]:
                if iou_matrix[current_idx, idx] < iou_threshold:
                    remaining_indices.append(idx)
            
            sorted_indices = remaining_indices
        
        return [openings[i] for i in keep_indices]
    
    def _calculate_iou_matrix(self, bboxes: List[List[float]]) -> np.ndarray:
        """Calculate IoU matrix for bounding boxes."""
        n = len(bboxes)
        iou_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                iou = self._calculate_iou(bboxes[i], bboxes[j])
                iou_matrix[i, j] = iou
                iou_matrix[j, i] = iou
        
        return iou_matrix
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU between two bounding boxes."""
        if len(bbox1) != 4 or len(bbox2) != 4:
            return 0.0
        
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)
        
        if x_max <= x_min or y_max <= y_min:
            return 0.0
        
        intersection = (x_max - x_min) * (y_max - y_min)
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _repair_door_touches_ground(self, layout: Dict[str, Any], 
                                   violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Repair doors that don't touch the ground."""
        facade_polygon = layout.get('facade_polygon', [])
        openings = layout.get('openings', [])
        
        if not facade_polygon:
            return layout
        
        facade_polygon_np = np.array(facade_polygon)
        facade_bottom = np.max(facade_polygon_np[:, 1])
        
        repaired_openings = []
        for opening in openings:
            if opening.get('kind') == 'door':
                bbox = opening.get('bbox_px', [])
                if len(bbox) == 4:
                    x_min, y_min, x_max, y_max = bbox
                    h = y_max - y_min
                    
                    # Move door to touch ground
                    new_y_min = facade_bottom - h
                    new_bbox = [x_min, new_y_min, x_max, facade_bottom]
                    
                    opening['bbox_px'] = new_bbox
                    opening['center_px'] = [
                        (x_min + x_max) / 2,
                        (new_y_min + facade_bottom) / 2
                    ]
            
            repaired_openings.append(opening)
        
        layout['openings'] = repaired_openings
        return layout
    
    def _repair_window_height(self, layout: Dict[str, Any], 
                             violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Repair windows at unreasonable height."""
        facade_polygon = layout.get('facade_polygon', [])
        openings = layout.get('openings', [])
        
        if not facade_polygon:
            return layout
        
        facade_polygon_np = np.array(facade_polygon)
        facade_top = np.min(facade_polygon_np[:, 1])
        facade_bottom = np.max(facade_polygon_np[:, 1])
        facade_height = facade_bottom - facade_top
        
        repaired_openings = []
        for opening in openings:
            if opening.get('kind') == 'window':
                bbox = opening.get('bbox_px', [])
                if len(bbox) == 4:
                    x_min, y_min, x_max, y_max = bbox
                    h = y_max - y_min
                    
                    # Calculate current relative height
                    relative_height = (y_min - facade_top) / facade_height
                    
                    # Adjust if too low or too high
                    if relative_height < 0.1:  # Too low
                        new_y_min = facade_top + facade_height * 0.1
                        new_y_max = new_y_min + h
                    elif relative_height > 0.9:  # Too high
                        new_y_max = facade_top + facade_height * 0.9
                        new_y_min = new_y_max - h
                    else:
                        new_y_min, new_y_max = y_min, y_max
                    
                    new_bbox = [x_min, new_y_min, x_max, new_y_max]
                    opening['bbox_px'] = new_bbox
                    opening['center_px'] = [
                        (x_min + x_max) / 2,
                        (new_y_min + new_y_max) / 2
                    ]
            
            repaired_openings.append(opening)
        
        layout['openings'] = repaired_openings
        return layout
    
    def _repair_opening_sizes(self, layout: Dict[str, Any], 
                             violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Repair openings with unreasonable dimensions."""
        openings = layout.get('openings', [])
        scale_factor = layout.get('scale_m_per_px', 0.01)
        
        repaired_openings = []
        for opening in openings:
            bbox = opening.get('bbox_px', [])
            if len(bbox) != 4:
                repaired_openings.append(opening)
                continue
            
            x_min, y_min, x_max, y_max = bbox
            w_px = x_max - x_min
            h_px = y_max - y_min
            w_m = w_px * scale_factor
            h_m = h_px * scale_factor
            
            opening_type = opening.get('kind', 'unknown')
            
            # Adjust dimensions based on type
            if opening_type == 'door':
                # Typical door: 0.8-1.0m wide, 2.0-2.2m high
                target_w_m = max(0.8, min(1.0, w_m))
                target_h_m = max(2.0, min(2.2, h_m))
            elif opening_type == 'window':
                # Typical window: 1.0-1.5m wide, 1.0-1.3m high
                target_w_m = max(1.0, min(1.5, w_m))
                target_h_m = max(1.0, min(1.3, h_m))
            else:
                repaired_openings.append(opening)
                continue
            
            # Convert back to pixels
            target_w_px = target_w_m / scale_factor
            target_h_px = target_h_m / scale_factor
            
            # Ensure minimum size
            target_w_px = max(target_w_px, self.min_opening_size)
            target_h_px = max(target_h_px, self.min_opening_size)
            
            # Calculate new bbox (centered)
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            
            new_x_min = center_x - target_w_px / 2
            new_y_min = center_y - target_h_px / 2
            new_x_max = center_x + target_w_px / 2
            new_y_max = center_y + target_h_px / 2
            
            opening['bbox_px'] = [new_x_min, new_y_min, new_x_max, new_y_max]
            opening['center_px'] = [center_x, center_y]
            opening['size_px'] = [target_w_px, target_h_px]
            
            repaired_openings.append(opening)
        
        layout['openings'] = repaired_openings
        return layout
    
    def _repair_facade_proportions(self, layout: Dict[str, Any], 
                                  violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Repair facade proportions."""
        facade_polygon = layout.get('facade_polygon', [])
        
        if not facade_polygon or len(facade_polygon) < 3:
            return layout
        
        facade_polygon_np = np.array(facade_polygon)
        
        # Calculate current dimensions
        min_x = np.min(facade_polygon_np[:, 0])
        max_x = np.max(facade_polygon_np[:, 0])
        min_y = np.min(facade_polygon_np[:, 1])
        max_y = np.max(facade_polygon_np[:, 1])
        
        width = max_x - min_x
        height = max_y - min_y
        aspect_ratio = width / height
        
        # Target aspect ratio (closer to 1.5)
        target_aspect_ratio = 1.5
        
        if aspect_ratio < target_aspect_ratio:
            # Too tall, increase width
            new_width = height * target_aspect_ratio
            width_diff = new_width - width
            min_x -= width_diff / 2
            max_x += width_diff / 2
        elif aspect_ratio > target_aspect_ratio:
            # Too wide, increase height
            new_height = width / target_aspect_ratio
            height_diff = new_height - height
            min_y -= height_diff / 2
            max_y += height_diff / 2
        
        # Create new rectangular facade
        new_facade = [
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y]
        ]
        
        layout['facade_polygon'] = new_facade
        return layout


def repair_layout_constraints(layout_params: Dict[str, Any], 
                             constraint_violations: List[Dict[str, Any]],
                             **kwargs) -> Dict[str, Any]:
    """
    Convenience function to repair layout constraints.
    
    Args:
        layout_params: Layout parameters to repair
        constraint_violations: Constraint violations to fix
        **kwargs: Additional arguments for ConstraintRepair
        
    Returns:
        Repaired layout parameters
    """
    repair = ConstraintRepair(**kwargs)
    return repair.repair_layout(layout_params, constraint_violations)
