"""
Opening solver for detecting and analyzing doors and windows.
Handles opening detection, classification, and constraint validation.
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Optional
import logging
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import json

logger = logging.getLogger(__name__)


class OpeningSolver:
    """Solves for door and window openings from segmentation results."""
    
    def __init__(self, 
                 min_opening_area: int = 100,
                 door_aspect_ratio_range: Tuple[float, float] = (0.3, 0.6),
                 window_aspect_ratio_range: Tuple[float, float] = (0.5, 2.0),
                 door_height_ratio_range: Tuple[float, float] = (0.6, 1.0),
                 window_height_ratio_range: Tuple[float, float] = (0.2, 0.8)):
        """
        Initialize the opening solver.
        
        Args:
            min_opening_area: Minimum area for opening detection
            door_aspect_ratio_range: Valid aspect ratio range for doors
            window_aspect_ratio_range: Valid aspect ratio range for windows
            door_height_ratio_range: Door height as ratio of wall height
            window_height_ratio_range: Window height as ratio of wall height
        """
        self.min_opening_area = min_opening_area
        self.door_aspect_ratio_range = door_aspect_ratio_range
        self.window_aspect_ratio_range = window_aspect_ratio_range
        self.door_height_ratio_range = door_height_ratio_range
        self.window_height_ratio_range = window_height_ratio_range
        
    def solve_openings(self, segmentation_result: Dict[str, Any], 
                      facade_polygon: List[List[float]],
                      scale_factor: float) -> Dict[str, Any]:
        """
        Solve for door and window openings.
        
        Args:
            segmentation_result: Segmentation results with opening masks
            facade_polygon: Facade outline polygon
            scale_factor: Scale factor from pixels to meters
            
        Returns:
            Dictionary containing opening analysis results
        """
        # Extract opening mask
        opening_mask = segmentation_result.get('opening_mask', np.zeros((100, 100), dtype=np.uint8))
        
        # Find opening contours
        contours = self._find_opening_contours(opening_mask)
        
        if not contours:
            return {
                'doors': [],
                'windows': [],
                'metadata': {
                    'total_openings': 0,
                    'doors_count': 0,
                    'windows_count': 0
                }
            }
        
        # Analyze each contour
        openings = []
        for contour in contours:
            opening = self._analyze_opening(contour, facade_polygon, scale_factor)
            if opening:
                openings.append(opening)
        
        # Classify openings
        doors, windows = self._classify_openings(openings)
        
        # Apply constraints and repair
        doors = self._apply_door_constraints(doors, facade_polygon, scale_factor)
        windows = self._apply_window_constraints(windows, facade_polygon, scale_factor)
        
        return {
            'doors': doors,
            'windows': windows,
            'metadata': {
                'total_openings': len(openings),
                'doors_count': len(doors),
                'windows_count': len(windows),
                'scale_factor': scale_factor
            }
        }
    
    def _find_opening_contours(self, opening_mask: np.ndarray) -> List[np.ndarray]:
        """Find contours in opening mask."""
        contours, _ = cv2.findContours(opening_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by area
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.min_opening_area:
                filtered_contours.append(contour)
        
        return filtered_contours
    
    def _analyze_opening(self, contour: np.ndarray, 
                        facade_polygon: List[List[float]], 
                        scale_factor: float) -> Optional[Dict[str, Any]]:
        """Analyze a single opening contour."""
        # Calculate bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate properties
        area = cv2.contourArea(contour)
        aspect_ratio = w / h if h > 0 else 0
        center_x = x + w / 2
        center_y = y + h / 2
        
        # Convert to meters
        w_m = w * scale_factor
        h_m = h * scale_factor
        area_m2 = area * scale_factor * scale_factor
        
        # Check if opening is within facade
        if not self._is_within_facade((x, y, w, h), facade_polygon):
            return None
        
        # Calculate position relative to facade
        facade_polygon_np = np.array(facade_polygon)
        facade_min_y = np.min(facade_polygon_np[:, 1])
        facade_max_y = np.max(facade_polygon_np[:, 1])
        facade_height = facade_max_y - facade_min_y
        
        relative_y = (center_y - facade_min_y) / facade_height if facade_height > 0 else 0
        
        return {
            'bbox_px': [x, y, x + w, y + h],
            'center_px': [center_x, center_y],
            'size_px': [w, h],
            'size_m': [w_m, h_m],
            'area_px': area,
            'area_m2': area_m2,
            'aspect_ratio': aspect_ratio,
            'relative_y': relative_y,
            'contour': contour.tolist()
        }
    
    def _is_within_facade(self, bbox: Tuple[int, int, int, int], 
                         facade_polygon: List[List[float]]) -> bool:
        """Check if bounding box is within facade polygon."""
        x, y, w, h = bbox
        facade_polygon_np = np.array(facade_polygon)
        
        # Check center point
        center_x = x + w / 2
        center_y = y + h / 2
        
        return cv2.pointPolygonTest(facade_polygon_np, (center_x, center_y), False) >= 0
    
    def _classify_openings(self, openings: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Classify openings as doors or windows."""
        doors = []
        windows = []
        
        for opening in openings:
            aspect_ratio = opening['aspect_ratio']
            relative_y = opening['relative_y']
            size_m = opening['size_m']
            
            # Classification rules
            is_door = self._is_likely_door(aspect_ratio, relative_y, size_m)
            is_window = self._is_likely_window(aspect_ratio, relative_y, size_m)
            
            if is_door and not is_window:
                opening['kind'] = 'door'
                opening['confidence'] = self._calculate_door_confidence(opening)
                doors.append(opening)
            elif is_window and not is_door:
                opening['kind'] = 'window'
                opening['confidence'] = self._calculate_window_confidence(opening)
                windows.append(opening)
            else:
                # Ambiguous case - use position and size
                if relative_y > 0.6:  # Lower part of facade
                    opening['kind'] = 'door'
                    opening['confidence'] = 0.6
                    doors.append(opening)
                else:  # Upper part of facade
                    opening['kind'] = 'window'
                    opening['confidence'] = 0.6
                    windows.append(opening)
        
        return doors, windows
    
    def _is_likely_door(self, aspect_ratio: float, relative_y: float, size_m: List[float]) -> bool:
        """Check if opening is likely a door."""
        w_m, h_m = size_m
        
        # Aspect ratio check
        aspect_ok = self.door_aspect_ratio_range[0] <= aspect_ratio <= self.door_aspect_ratio_range[1]
        
        # Position check (doors are typically in lower part)
        position_ok = relative_y >= 0.5
        
        # Size check (typical door dimensions)
        size_ok = 0.6 <= w_m <= 1.2 and 1.8 <= h_m <= 2.4
        
        return aspect_ok and position_ok and size_ok
    
    def _is_likely_window(self, aspect_ratio: float, relative_y: float, size_m: List[float]) -> bool:
        """Check if opening is likely a window."""
        w_m, h_m = size_m
        
        # Aspect ratio check
        aspect_ok = self.window_aspect_ratio_range[0] <= aspect_ratio <= self.window_aspect_ratio_range[1]
        
        # Position check (windows are typically in upper part)
        position_ok = relative_y <= 0.8
        
        # Size check (typical window dimensions)
        size_ok = 0.6 <= w_m <= 2.5 and 0.6 <= h_m <= 1.8
        
        return aspect_ok and position_ok and size_ok
    
    def _calculate_door_confidence(self, opening: Dict[str, Any]) -> float:
        """Calculate confidence score for door classification."""
        aspect_ratio = opening['aspect_ratio']
        relative_y = opening['relative_y']
        size_m = opening['size_m']
        w_m, h_m = size_m
        
        confidence = 1.0
        
        # Aspect ratio score
        if self.door_aspect_ratio_range[0] <= aspect_ratio <= self.door_aspect_ratio_range[1]:
            confidence *= 1.0
        else:
            confidence *= 0.7
        
        # Position score
        if relative_y >= 0.6:
            confidence *= 1.0
        else:
            confidence *= 0.5
        
        # Size score
        if 0.8 <= w_m <= 1.0 and 2.0 <= h_m <= 2.2:
            confidence *= 1.0
        elif 0.6 <= w_m <= 1.2 and 1.8 <= h_m <= 2.4:
            confidence *= 0.8
        else:
            confidence *= 0.6
        
        return min(confidence, 1.0)
    
    def _calculate_window_confidence(self, opening: Dict[str, Any]) -> float:
        """Calculate confidence score for window classification."""
        aspect_ratio = opening['aspect_ratio']
        relative_y = opening['relative_y']
        size_m = opening['size_m']
        w_m, h_m = size_m
        
        confidence = 1.0
        
        # Aspect ratio score
        if self.window_aspect_ratio_range[0] <= aspect_ratio <= self.window_aspect_ratio_range[1]:
            confidence *= 1.0
        else:
            confidence *= 0.7
        
        # Position score
        if relative_y <= 0.7:
            confidence *= 1.0
        else:
            confidence *= 0.5
        
        # Size score
        if 1.0 <= w_m <= 1.5 and 1.0 <= h_m <= 1.3:
            confidence *= 1.0
        elif 0.6 <= w_m <= 2.5 and 0.6 <= h_m <= 1.8:
            confidence *= 0.8
        else:
            confidence *= 0.6
        
        return min(confidence, 1.0)
    
    def _apply_door_constraints(self, doors: List[Dict[str, Any]], 
                               facade_polygon: List[List[float]], 
                               scale_factor: float) -> List[Dict[str, Any]]:
        """Apply constraints to door openings."""
        if not doors:
            return doors
        
        # Sort by confidence
        doors.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Apply non-maximum suppression to remove overlapping doors
        filtered_doors = self._apply_nms(doors, iou_threshold=0.3)
        
        # Apply additional constraints
        for door in filtered_doors:
            # Ensure door touches ground
            door = self._ensure_door_touches_ground(door, facade_polygon, scale_factor)
            
            # Ensure reasonable dimensions
            door = self._ensure_reasonable_dimensions(door, 'door')
        
        return filtered_doors
    
    def _apply_window_constraints(self, windows: List[Dict[str, Any]], 
                                 facade_polygon: List[List[float]], 
                                 scale_factor: float) -> List[Dict[str, Any]]:
        """Apply constraints to window openings."""
        if not windows:
            return windows
        
        # Sort by confidence
        windows.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Apply non-maximum suppression to remove overlapping windows
        filtered_windows = self._apply_nms(windows, iou_threshold=0.2)
        
        # Apply additional constraints
        for window in filtered_windows:
            # Ensure reasonable dimensions
            window = self._ensure_reasonable_dimensions(window, 'window')
            
            # Ensure window is not too close to ground
            window = self._ensure_window_height(window, facade_polygon, scale_factor)
        
        return filtered_windows
    
    def _apply_nms(self, openings: List[Dict[str, Any]], iou_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Apply non-maximum suppression to remove overlapping openings."""
        if not openings:
            return openings
        
        # Convert to numpy arrays for easier calculation
        bboxes = np.array([opening['bbox_px'] for opening in openings])
        scores = np.array([opening['confidence'] for opening in openings])
        
        # Calculate IoU matrix
        iou_matrix = self._calculate_iou_matrix(bboxes)
        
        # Apply NMS
        keep_indices = []
        sorted_indices = np.argsort(scores)[::-1]
        
        while len(sorted_indices) > 0:
            # Keep the highest scoring opening
            current_idx = sorted_indices[0]
            keep_indices.append(current_idx)
            
            # Remove openings with high IoU
            remaining_indices = []
            for idx in sorted_indices[1:]:
                if iou_matrix[current_idx, idx] < iou_threshold:
                    remaining_indices.append(idx)
            
            sorted_indices = np.array(remaining_indices)
        
        return [openings[i] for i in keep_indices]
    
    def _calculate_iou_matrix(self, bboxes: np.ndarray) -> np.ndarray:
        """Calculate IoU matrix for bounding boxes."""
        n = len(bboxes)
        iou_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                iou = self._calculate_iou(bboxes[i], bboxes[j])
                iou_matrix[i, j] = iou
                iou_matrix[j, i] = iou
        
        return iou_matrix
    
    def _calculate_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate IoU between two bounding boxes."""
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
    
    def _ensure_door_touches_ground(self, door: Dict[str, Any], 
                                   facade_polygon: List[List[float]], 
                                   scale_factor: float) -> Dict[str, Any]:
        """Ensure door touches the ground (bottom of facade)."""
        facade_polygon_np = np.array(facade_polygon)
        facade_bottom = np.max(facade_polygon_np[:, 1])
        
        bbox = door['bbox_px']
        x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
        
        # Adjust door position to touch ground
        new_y = facade_bottom - h
        door['bbox_px'] = [x, new_y, x + w, facade_bottom]
        door['center_px'] = [x + w/2, new_y + h/2]
        
        return door
    
    def _ensure_reasonable_dimensions(self, opening: Dict[str, Any], 
                                     opening_type: str) -> Dict[str, Any]:
        """Ensure opening has reasonable dimensions."""
        bbox = opening['bbox_px']
        x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
        
        if opening_type == 'door':
            # Typical door dimensions
            min_w, max_w = 60, 120  # pixels
            min_h, max_h = 180, 240  # pixels
        else:  # window
            # Typical window dimensions
            min_w, max_w = 60, 200  # pixels
            min_h, max_h = 60, 150  # pixels
        
        # Clamp dimensions
        w = max(min_w, min(w, max_w))
        h = max(min_h, min(h, max_h))
        
        opening['bbox_px'] = [x, y, x + w, y + h]
        opening['size_px'] = [w, h]
        
        return opening
    
    def _ensure_window_height(self, window: Dict[str, Any], 
                             facade_polygon: List[List[float]], 
                             scale_factor: float) -> Dict[str, Any]:
        """Ensure window is at reasonable height."""
        facade_polygon_np = np.array(facade_polygon)
        facade_bottom = np.max(facade_polygon_np[:, 1])
        facade_top = np.min(facade_polygon_np[:, 1])
        facade_height = facade_bottom - facade_top
        
        bbox = window['bbox_px']
        x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
        
        # Ensure window is not too close to ground
        min_height = facade_bottom - facade_height * 0.8  # At least 20% from bottom
        if y + h > min_height:
            new_y = min_height - h
            window['bbox_px'] = [x, new_y, x + w, min_height]
            window['center_px'] = [x + w/2, new_y + h/2]
        
        return window


def solve_openings(segmentation_result: Dict[str, Any], 
                  facade_polygon: List[List[float]],
                  scale_factor: float,
                  **kwargs) -> Dict[str, Any]:
    """
    Convenience function to solve openings.
    
    Args:
        segmentation_result: Segmentation results
        facade_polygon: Facade outline
        scale_factor: Scale factor
        **kwargs: Additional arguments for OpeningSolver
        
    Returns:
        Opening analysis results
    """
    solver = OpeningSolver(**kwargs)
    return solver.solve_openings(segmentation_result, facade_polygon, scale_factor)
