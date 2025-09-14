"""
Facade solver for extracting 2D facade layout from segmented images.
Converts pixel coordinates to parametric layout parameters.
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Optional
import logging
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import json

logger = logging.getLogger(__name__)


class FacadeSolver:
    """Solves for facade layout parameters from segmentation results."""
    
    def __init__(self, 
                 min_wall_thickness: float = 0.12,
                 door_height_m: float = 2.0,
                 window_height_ratio: Tuple[float, float] = (0.3, 0.8),
                 manhattan_tolerance: float = 15.0):
        """
        Initialize the facade solver.
        
        Args:
            min_wall_thickness: Minimum wall thickness in meters
            door_height_m: Standard door height in meters
            window_height_ratio: Window height as ratio of wall height
            manhattan_tolerance: Tolerance for Manhattan alignment in degrees
        """
        self.min_wall_thickness = min_wall_thickness
        self.door_height_m = door_height_m
        self.window_height_ratio = window_height_ratio
        self.manhattan_tolerance = manhattan_tolerance
        
    def solve_facade(self, segmentation_result: Dict[str, Any], 
                    image_size: Tuple[int, int]) -> Dict[str, Any]:
        """
        Solve for facade layout parameters.
        
        Args:
            segmentation_result: Segmentation results with wall/opening masks
            image_size: Original image size (width, height)
            
        Returns:
            Dictionary containing facade layout parameters
        """
        width, height = image_size
        
        # Extract wall and opening masks
        wall_mask = segmentation_result.get('wall_mask', np.zeros((height, width), dtype=np.uint8))
        opening_mask = segmentation_result.get('opening_mask', np.zeros((height, width), dtype=np.uint8))
        
        # Find facade outline
        facade_polygon = self._extract_facade_outline(wall_mask)
        
        if facade_polygon is None or len(facade_polygon) < 3:
            logger.warning("Could not extract valid facade outline")
            return self._create_empty_layout()
        
        # Apply Manhattan alignment
        facade_polygon = self._apply_manhattan_alignment(facade_polygon)
        
        # Calculate scale factor
        scale_factor = self._calculate_scale_factor(facade_polygon, wall_mask)
        
        # Extract openings
        openings = self._extract_openings(opening_mask, facade_polygon, scale_factor)
        
        # Create layout parameters
        layout_params = {
            'scale_m_per_px': scale_factor,
            'facade_polygon': facade_polygon.tolist(),
            'openings': openings,
            'metadata': {
                'image_size': image_size,
                'manhattan_aligned': True,
                'scale_calibrated': True
            }
        }
        
        return layout_params
    
    def _extract_facade_outline(self, wall_mask: np.ndarray) -> Optional[np.ndarray]:
        """Extract facade outline from wall mask."""
        # Find contours
        contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find the largest contour (main facade)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Simplify contour
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        simplified = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Convert to 2D points
        points = simplified.reshape(-1, 2)
        
        # Ensure we have at least 3 points
        if len(points) < 3:
            return None
        
        return points
    
    def _apply_manhattan_alignment(self, polygon: np.ndarray) -> np.ndarray:
        """Apply Manhattan alignment to polygon edges."""
        aligned_polygon = polygon.copy()
        
        # Calculate edge directions
        edges = []
        for i in range(len(polygon)):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % len(polygon)]
            edge = p2 - p1
            length = np.linalg.norm(edge)
            if length > 0:
                direction = edge / length
                edges.append((direction, i, p1, p2))
        
        # Group edges by direction
        edge_groups = self._group_edges_by_direction(edges)
        
        # Align each group to Manhattan directions
        for group in edge_groups:
            if len(group) >= 2:  # Only align if we have multiple edges
                aligned_direction = self._align_to_manhattan(group[0][0])
                
                # Update all edges in the group
                for direction, edge_idx, p1, p2 in group:
                    # Calculate new endpoint
                    length = np.linalg.norm(p2 - p1)
                    new_p2 = p1 + aligned_direction * length
                    
                    # Update polygon
                    aligned_polygon[(edge_idx + 1) % len(aligned_polygon)] = new_p2
        
        return aligned_polygon
    
    def _group_edges_by_direction(self, edges: List[Tuple]) -> List[List[Tuple]]:
        """Group edges by similar directions."""
        groups = []
        used_edges = set()
        
        for i, (dir1, idx1, p1_1, p2_1) in enumerate(edges):
            if i in used_edges:
                continue
            
            group = [(dir1, idx1, p1_1, p2_1)]
            used_edges.add(i)
            
            for j, (dir2, idx2, p1_2, p2_2) in enumerate(edges[i+1:], i+1):
                if j in used_edges:
                    continue
                
                # Calculate angle between directions
                dot_product = np.dot(dir1, dir2)
                angle = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
                
                if angle < self.manhattan_tolerance or angle > (180 - self.manhattan_tolerance):
                    group.append((dir2, idx2, p1_2, p2_2))
                    used_edges.add(j)
            
            groups.append(group)
        
        return groups
    
    def _align_to_manhattan(self, direction: np.ndarray) -> np.ndarray:
        """Align direction vector to Manhattan (horizontal/vertical)."""
        # Check if closer to horizontal or vertical
        if abs(direction[0]) > abs(direction[1]):
            # Closer to horizontal
            return np.array([np.sign(direction[0]), 0.0])
        else:
            # Closer to vertical
            return np.array([0.0, np.sign(direction[1])])
    
    def _calculate_scale_factor(self, facade_polygon: np.ndarray, 
                               wall_mask: np.ndarray) -> float:
        """Calculate scale factor from pixels to meters."""
        # Estimate facade height from polygon
        min_y = np.min(facade_polygon[:, 1])
        max_y = np.max(facade_polygon[:, 1])
        facade_height_px = max_y - min_y
        
        # Estimate facade width from polygon
        min_x = np.min(facade_polygon[:, 0])
        max_x = np.max(facade_polygon[:, 0])
        facade_width_px = max_x - min_x
        
        # Use the larger dimension for scale calculation
        facade_size_px = max(facade_height_px, facade_width_px)
        
        # Estimate real-world size (assume typical house facade is 8-12 meters)
        estimated_facade_size_m = 10.0  # meters
        
        # Calculate scale factor
        scale_factor = estimated_facade_size_m / facade_size_px
        
        # Ensure reasonable scale factor (between 0.001 and 0.1 m/px)
        scale_factor = np.clip(scale_factor, 0.001, 0.1)
        
        return scale_factor
    
    def _extract_openings(self, opening_mask: np.ndarray, 
                         facade_polygon: np.ndarray, 
                         scale_factor: float) -> List[Dict[str, Any]]:
        """Extract door and window openings."""
        # Find contours in opening mask
        contours, _ = cv2.findContours(opening_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        openings = []
        
        for contour in contours:
            # Calculate bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size
            if w < 10 or h < 10:  # Too small
                continue
            
            # Check if opening is within facade
            if not self._is_within_facade((x, y, w, h), facade_polygon):
                continue
            
            # Classify as door or window based on aspect ratio and position
            aspect_ratio = w / h
            center_y = y + h / 2
            
            if aspect_ratio > 0.8 and aspect_ratio < 1.5 and center_y > opening_mask.shape[0] * 0.6:
                # Square-ish and in lower part - likely a door
                opening_type = 'door'
            else:
                # Rectangular and in upper part - likely a window
                opening_type = 'window'
            
            # Calculate confidence based on size and position
            confidence = self._calculate_opening_confidence((x, y, w, h), opening_type, scale_factor)
            
            opening = {
                'kind': opening_type,
                'bbox_px': [x, y, x + w, y + h],
                'confidence': confidence
            }
            
            openings.append(opening)
        
        return openings
    
    def _is_within_facade(self, bbox: Tuple[int, int, int, int], 
                         facade_polygon: np.ndarray) -> bool:
        """Check if bounding box is within facade polygon."""
        x, y, w, h = bbox
        
        # Check center point
        center_x = x + w / 2
        center_y = y + h / 2
        
        return cv2.pointPolygonTest(facade_polygon, (center_x, center_y), False) >= 0
    
    def _calculate_opening_confidence(self, bbox: Tuple[int, int, int, int], 
                                    opening_type: str, scale_factor: float) -> float:
        """Calculate confidence score for opening detection."""
        x, y, w, h = bbox
        
        # Convert to meters
        w_m = w * scale_factor
        h_m = h * scale_factor
        
        confidence = 1.0
        
        # Adjust confidence based on size
        if opening_type == 'door':
            # Typical door: 0.8-1.0m wide, 2.0-2.2m high
            if 0.6 <= w_m <= 1.2 and 1.8 <= h_m <= 2.4:
                confidence *= 1.0
            else:
                confidence *= 0.7
        else:  # window
            # Typical window: 0.8-2.0m wide, 0.8-1.5m high
            if 0.6 <= w_m <= 2.5 and 0.6 <= h_m <= 1.8:
                confidence *= 1.0
            else:
                confidence *= 0.7
        
        # Adjust confidence based on aspect ratio
        aspect_ratio = w / h
        if opening_type == 'door':
            if 0.3 <= aspect_ratio <= 0.6:  # Tall and narrow
                confidence *= 1.0
            else:
                confidence *= 0.8
        else:  # window
            if 0.5 <= aspect_ratio <= 2.0:  # Reasonable window proportions
                confidence *= 1.0
            else:
                confidence *= 0.8
        
        return min(confidence, 1.0)
    
    def _create_empty_layout(self) -> Dict[str, Any]:
        """Create empty layout when facade extraction fails."""
        return {
            'scale_m_per_px': 0.01,
            'facade_polygon': [],
            'openings': [],
            'metadata': {
                'image_size': (0, 0),
                'manhattan_aligned': False,
                'scale_calibrated': False,
                'error': 'Failed to extract facade outline'
            }
        }


def solve_facade_layout(segmentation_result: Dict[str, Any], 
                       image_size: Tuple[int, int],
                       **kwargs) -> Dict[str, Any]:
    """
    Convenience function to solve facade layout.
    
    Args:
        segmentation_result: Segmentation results
        image_size: Image dimensions
        **kwargs: Additional arguments for FacadeSolver
        
    Returns:
        Facade layout parameters
    """
    solver = FacadeSolver(**kwargs)
    return solver.solve_facade(segmentation_result, image_size)
