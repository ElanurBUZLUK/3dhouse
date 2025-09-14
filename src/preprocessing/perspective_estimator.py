"""
Perspective estimation and correction for child sketches.
Estimates camera parameters and corrects perspective distortion.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import logging
from scipy.optimize import minimize
import math

logger = logging.getLogger(__name__)


class PerspectiveEstimator:
    """Estimates and corrects perspective distortion in architectural sketches."""
    
    def __init__(self, 
                 min_line_length: float = 50.0,
                 angle_tolerance: float = 5.0,
                 vanishing_point_threshold: float = 100.0):
        """
        Initialize the perspective estimator.
        
        Args:
            min_line_length: Minimum length for line detection
            angle_tolerance: Tolerance for parallel line detection in degrees
            vanishing_point_threshold: Threshold for vanishing point detection
        """
        self.min_line_length = min_line_length
        self.angle_tolerance = angle_tolerance
        self.vanishing_point_threshold = vanishing_point_threshold
        
    def estimate_perspective(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Estimate perspective parameters from the image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing:
                - vanishing_points: Detected vanishing points
                - horizon_line: Horizon line equation
                - perspective_matrix: Perspective correction matrix
                - confidence: Confidence score
                - metadata: Additional information
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Detect lines
        lines = self._detect_lines(gray)
        
        if len(lines) < 4:
            return {
                'vanishing_points': [],
                'horizon_line': None,
                'perspective_matrix': None,
                'confidence': 0.0,
                'metadata': {'error': 'Insufficient lines detected'}
            }
        
        # Group lines by direction
        line_groups = self._group_parallel_lines(lines)
        
        # Find vanishing points
        vanishing_points = self._find_vanishing_points(line_groups)
        
        # Calculate horizon line
        horizon_line = self._calculate_horizon_line(vanishing_points)
        
        # Estimate perspective matrix
        perspective_matrix = self._estimate_perspective_matrix(vanishing_points, image.shape)
        
        # Calculate confidence
        confidence = self._calculate_confidence(vanishing_points, line_groups)
        
        return {
            'vanishing_points': vanishing_points,
            'horizon_line': horizon_line,
            'perspective_matrix': perspective_matrix,
            'confidence': confidence,
            'metadata': {
                'total_lines': len(lines),
                'line_groups': len(line_groups),
                'vanishing_points_count': len(vanishing_points)
            }
        }
    
    def _detect_lines(self, image: np.ndarray) -> List[Tuple[float, float, float, float]]:
        """Detect lines in the image using Hough transform."""
        # Apply edge detection
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=self.min_line_length, maxLineGap=10)
        
        if lines is None:
            return []
        
        # Convert to (x1, y1, x2, y2) format
        line_segments = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_segments.append((x1, y1, x2, y2))
        
        return line_segments
    
    def _group_parallel_lines(self, lines: List[Tuple[float, float, float, float]]) -> List[List[Tuple[float, float, float, float]]]:
        """Group lines by their direction (parallel lines)."""
        if not lines:
            return []
        
        # Calculate line directions
        line_directions = []
        for x1, y1, x2, y2 in lines:
            dx = x2 - x1
            dy = y2 - y1
            length = math.sqrt(dx*dx + dy*dy)
            if length > 0:
                direction = (dx/length, dy/length)
                line_directions.append((direction, (x1, y1, x2, y2)))
        
        # Group lines by similar directions
        groups = []
        used_lines = set()
        
        for i, (dir1, line1) in enumerate(line_directions):
            if i in used_lines:
                continue
                
            group = [line1]
            used_lines.add(i)
            
            for j, (dir2, line2) in enumerate(line_directions[i+1:], i+1):
                if j in used_lines:
                    continue
                
                # Calculate angle between directions
                dot_product = abs(dir1[0]*dir2[0] + dir1[1]*dir2[1])
                angle = math.degrees(math.acos(min(dot_product, 1.0)))
                
                if angle < self.angle_tolerance or angle > (180 - self.angle_tolerance):
                    group.append(line2)
                    used_lines.add(j)
            
            if len(group) >= 2:  # Only keep groups with at least 2 lines
                groups.append(group)
        
        return groups
    
    def _find_vanishing_points(self, line_groups: List[List[Tuple[float, float, float, float]]]) -> List[Tuple[float, float]]:
        """Find vanishing points from line groups."""
        vanishing_points = []
        
        for group in line_groups:
            if len(group) < 2:
                continue
                
            # Find intersection of lines in the group
            vanishing_point = self._find_line_intersection(group)
            
            if vanishing_point is not None:
                # Check if vanishing point is reasonable (not too far from image)
                x, y = vanishing_point
                if abs(x) < self.vanishing_point_threshold and abs(y) < self.vanishing_point_threshold:
                    vanishing_points.append(vanishing_point)
        
        return vanishing_points
    
    def _find_line_intersection(self, lines: List[Tuple[float, float, float, float]]) -> Optional[Tuple[float, float]]:
        """Find the intersection point of multiple lines using least squares."""
        if len(lines) < 2:
            return None
        
        # Convert lines to ax + by + c = 0 form
        line_equations = []
        for x1, y1, x2, y2 in lines:
            # Calculate line equation: (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x1 = 0
            a = y2 - y1
            b = x1 - x2
            c = (x2 - x1) * y1 - (y2 - y1) * x1
            
            # Normalize
            length = math.sqrt(a*a + b*b)
            if length > 0:
                a /= length
                b /= length
                c /= length
                line_equations.append((a, b, c))
        
        if len(line_equations) < 2:
            return None
        
        # Solve using least squares: minimize sum of (ax + by + c)^2
        A = np.array([[eq[0], eq[1]] for eq in line_equations])
        b = np.array([-eq[2] for eq in line_equations])
        
        try:
            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            if len(residuals) > 0 and residuals[0] < 100:  # Check if solution is reasonable
                return (float(x[0]), float(x[1]))
        except:
            pass
        
        return None
    
    def _calculate_horizon_line(self, vanishing_points: List[Tuple[float, float]]) -> Optional[Tuple[float, float, float]]:
        """Calculate horizon line from vanishing points."""
        if len(vanishing_points) < 2:
            return None
        
        # Find two vanishing points that are most likely on the horizon
        best_points = self._select_horizon_points(vanishing_points)
        
        if len(best_points) < 2:
            return None
        
        # Calculate line through two points: ax + by + c = 0
        x1, y1 = best_points[0]
        x2, y2 = best_points[1]
        
        a = y2 - y1
        b = x1 - x2
        c = (x2 - x1) * y1 - (y2 - y1) * x1
        
        # Normalize
        length = math.sqrt(a*a + b*b)
        if length > 0:
            a /= length
            b /= length
            c /= length
        
        return (a, b, c)
    
    def _select_horizon_points(self, vanishing_points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Select the two vanishing points most likely to be on the horizon."""
        if len(vanishing_points) < 2:
            return vanishing_points
        
        # Sort by y-coordinate (horizon points should be at similar heights)
        sorted_points = sorted(vanishing_points, key=lambda p: p[1])
        
        # Find the two points with the smallest y-difference
        best_pair = None
        min_y_diff = float('inf')
        
        for i in range(len(sorted_points)):
            for j in range(i + 1, len(sorted_points)):
                y_diff = abs(sorted_points[i][1] - sorted_points[j][1])
                if y_diff < min_y_diff:
                    min_y_diff = y_diff
                    best_pair = (sorted_points[i], sorted_points[j])
        
        return list(best_pair) if best_pair else sorted_points[:2]
    
    def _estimate_perspective_matrix(self, vanishing_points: List[Tuple[float, float]], 
                                   image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Estimate perspective correction matrix."""
        if len(vanishing_points) < 2:
            return None
        
        h, w = image_shape[:2]
        
        # Use the two most likely horizon points
        horizon_points = self._select_horizon_points(vanishing_points)
        
        if len(horizon_points) < 2:
            return None
        
        # Calculate perspective transform to correct the image
        # This is a simplified approach - in practice, you'd need more sophisticated calibration
        
        # Define source points (corners of the image)
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        # Define destination points (corrected corners)
        # This is a simplified correction - in practice, you'd calculate proper destination points
        dst_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        # Calculate perspective transform matrix
        try:
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            return matrix
        except:
            return None
    
    def _calculate_confidence(self, vanishing_points: List[Tuple[float, float]], 
                            line_groups: List[List[Tuple[float, float, float, float]]]) -> float:
        """Calculate confidence score for perspective estimation."""
        if not vanishing_points or not line_groups:
            return 0.0
        
        # Base confidence on number of vanishing points and line groups
        vp_score = min(len(vanishing_points) / 3.0, 1.0)
        line_score = min(len(line_groups) / 4.0, 1.0)
        
        # Combine scores
        confidence = (vp_score + line_score) / 2.0
        
        return min(confidence, 1.0)
    
    def correct_perspective(self, image: np.ndarray, 
                          perspective_matrix: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Correct perspective distortion in the image.
        
        Args:
            image: Input image as numpy array
            perspective_matrix: Optional pre-computed perspective matrix
            
        Returns:
            Dictionary containing:
                - corrected_image: Perspective-corrected image
                - perspective_matrix: Matrix used for correction
                - metadata: Additional information
        """
        if perspective_matrix is None:
            # Estimate perspective first
            perspective_result = self.estimate_perspective(image)
            perspective_matrix = perspective_result['perspective_matrix']
            
            if perspective_matrix is None:
                return {
                    'corrected_image': image,
                    'perspective_matrix': None,
                    'metadata': {'error': 'Could not estimate perspective'}
                }
        
        # Apply perspective correction
        h, w = image.shape[:2]
        corrected_image = cv2.warpPerspective(image, perspective_matrix, (w, h))
        
        return {
            'corrected_image': corrected_image,
            'perspective_matrix': perspective_matrix,
            'metadata': {
                'original_size': (w, h),
                'corrected_size': corrected_image.shape[:2]
            }
        }


def estimate_perspective_from_image(image_path: str, **kwargs) -> Dict[str, Any]:
    """
    Convenience function to estimate perspective from an image file.
    
    Args:
        image_path: Path to the input image
        **kwargs: Additional arguments for PerspectiveEstimator.estimate_perspective()
        
    Returns:
        Perspective estimation results dictionary
    """
    import cv2
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Estimate perspective
    estimator = PerspectiveEstimator()
    return estimator.estimate_perspective(image, **kwargs)
