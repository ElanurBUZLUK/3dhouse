"""
Shape detection and analysis for architectural elements in child sketches.
Detects and analyzes geometric shapes like rectangles, triangles, and polygons.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
import logging
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from sklearn.cluster import DBSCAN
import math

logger = logging.getLogger(__name__)


class ShapeDetector:
    """Detects and analyzes geometric shapes in architectural sketches."""
    
    def __init__(self, 
                 min_contour_area: int = 50,
                 min_vertices: int = 3,
                 max_vertices: int = 20,
                 angle_tolerance: float = 15.0,
                 line_tolerance: float = 0.02):
        """
        Initialize the shape detector.
        
        Args:
            min_contour_area: Minimum area for contour filtering
            min_vertices: Minimum number of vertices for polygon detection
            max_vertices: Maximum number of vertices for polygon detection
            angle_tolerance: Tolerance for angle detection in degrees
            line_tolerance: Tolerance for line detection (fraction of perimeter)
        """
        self.min_contour_area = min_contour_area
        self.min_vertices = min_vertices
        self.max_vertices = max_vertices
        self.angle_tolerance = angle_tolerance
        self.line_tolerance = line_tolerance
        
    def detect_shapes(self, contours: List[np.ndarray]) -> Dict[str, Any]:
        """
        Detect geometric shapes in the given contours.
        
        Args:
            contours: List of contours to analyze
            
        Returns:
            Dictionary containing:
                - rectangles: List of detected rectangles
                - triangles: List of detected triangles
                - polygons: List of detected polygons
                - circles: List of detected circles
                - lines: List of detected lines
                - metadata: Analysis metadata
        """
        rectangles = []
        triangles = []
        polygons = []
        circles = []
        lines = []
        
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue
                
            # Detect shape type
            shape_type, shape_data = self._classify_shape(contour)
            
            if shape_type == 'rectangle':
                rectangles.append(shape_data)
            elif shape_type == 'triangle':
                triangles.append(shape_data)
            elif shape_type == 'polygon':
                polygons.append(shape_data)
            elif shape_type == 'circle':
                circles.append(shape_data)
            elif shape_type == 'line':
                lines.append(shape_data)
        
        return {
            'rectangles': rectangles,
            'triangles': triangles,
            'polygons': polygons,
            'circles': circles,
            'lines': lines,
            'metadata': {
                'total_contours': len(contours),
                'processed_contours': len(rectangles) + len(triangles) + 
                                    len(polygons) + len(circles) + len(lines),
                'rectangle_count': len(rectangles),
                'triangle_count': len(triangles),
                'polygon_count': len(polygons),
                'circle_count': len(circles),
                'line_count': len(lines)
            }
        }
    
    def _classify_shape(self, contour: np.ndarray) -> Tuple[str, Dict[str, Any]]:
        """Classify a single contour into a shape type."""
        # Calculate basic properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Approximate contour to polygon
        epsilon = self.line_tolerance * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Calculate shape properties
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Check for circle
        if self._is_circle(contour, area, perimeter):
            return 'circle', self._extract_circle_data(contour)
        
        # Check for line
        if self._is_line(contour, area, perimeter, aspect_ratio):
            return 'line', self._extract_line_data(contour)
        
        # Check for triangle
        if len(approx) == 3 and solidity > 0.8:
            return 'triangle', self._extract_triangle_data(contour, approx)
        
        # Check for rectangle
        if len(approx) == 4 and solidity > 0.8:
            if self._is_rectangle(approx):
                return 'rectangle', self._extract_rectangle_data(contour, approx)
        
        # Check for polygon
        if self.min_vertices <= len(approx) <= self.max_vertices and solidity > 0.6:
            return 'polygon', self._extract_polygon_data(contour, approx)
        
        # Default to line if nothing else matches
        return 'line', self._extract_line_data(contour)
    
    def _is_circle(self, contour: np.ndarray, area: float, perimeter: float) -> bool:
        """Check if contour is approximately circular."""
        if area < 100:  # Too small to be a meaningful circle
            return False
            
        # Calculate circularity
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        
        # Check if it's roughly circular
        return circularity > 0.7
    
    def _is_line(self, contour: np.ndarray, area: float, perimeter: float, aspect_ratio: float) -> bool:
        """Check if contour is approximately a line."""
        # Very thin and long shapes are likely lines
        if aspect_ratio > 5.0 or aspect_ratio < 0.2:
            return True
            
        # Check if area is very small compared to perimeter
        if area < perimeter * 2:
            return True
            
        return False
    
    def _is_rectangle(self, approx: np.ndarray) -> bool:
        """Check if approximated polygon is a rectangle."""
        if len(approx) != 4:
            return False
            
        # Calculate angles between consecutive edges
        angles = []
        for i in range(4):
            p1 = approx[i][0]
            p2 = approx[(i + 1) % 4][0]
            p3 = approx[(i + 2) % 4][0]
            
            # Calculate angle
            v1 = p1 - p2
            v2 = p3 - p2
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = math.degrees(math.acos(cos_angle))
            angles.append(angle)
        
        # Check if all angles are approximately 90 degrees
        for angle in angles:
            if abs(angle - 90.0) > self.angle_tolerance:
                return False
                
        return True
    
    def _extract_circle_data(self, contour: np.ndarray) -> Dict[str, Any]:
        """Extract circle data from contour."""
        # Fit circle using least squares
        (x, y), radius = cv2.minEnclosingCircle(contour)
        
        # Calculate circle properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        
        return {
            'center': (float(x), float(y)),
            'radius': float(radius),
            'area': float(area),
            'perimeter': float(perimeter),
            'circularity': float(circularity),
            'contour': contour
        }
    
    def _extract_line_data(self, contour: np.ndarray) -> Dict[str, Any]:
        """Extract line data from contour."""
        # Fit line using least squares
        [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        
        # Calculate line properties
        length = np.linalg.norm([vx, vy])
        angle = math.degrees(math.atan2(vy, vx))
        
        # Get endpoints
        points = contour.reshape(-1, 2)
        start_point = points[np.argmin(np.linalg.norm(points - [x, y], axis=1))]
        end_point = points[np.argmax(np.linalg.norm(points - [x, y], axis=1))]
        
        return {
            'start_point': start_point.tolist(),
            'end_point': end_point.tolist(),
            'direction': (float(vx[0]), float(vy[0])),
            'length': float(length),
            'angle': float(angle),
            'contour': contour
        }
    
    def _extract_triangle_data(self, contour: np.ndarray, approx: np.ndarray) -> Dict[str, Any]:
        """Extract triangle data from contour."""
        # Get vertices
        vertices = approx.reshape(-1, 2)
        
        # Calculate triangle properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Calculate angles
        angles = []
        for i in range(3):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % 3]
            p3 = vertices[(i + 2) % 3]
            
            v1 = p1 - p2
            v2 = p3 - p2
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = math.degrees(math.acos(cos_angle))
            angles.append(angle)
        
        # Classify triangle type
        triangle_type = self._classify_triangle_type(angles)
        
        return {
            'vertices': vertices.tolist(),
            'area': float(area),
            'perimeter': float(perimeter),
            'angles': angles,
            'triangle_type': triangle_type,
            'contour': contour
        }
    
    def _extract_rectangle_data(self, contour: np.ndarray, approx: np.ndarray) -> Dict[str, Any]:
        """Extract rectangle data from contour."""
        # Get vertices
        vertices = approx.reshape(-1, 2)
        
        # Calculate rectangle properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Calculate angles
        angles = []
        for i in range(4):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % 4]
            p3 = vertices[(i + 2) % 4]
            
            v1 = p1 - p2
            v2 = p3 - p2
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = math.degrees(math.acos(cos_angle))
            angles.append(angle)
        
        return {
            'vertices': vertices.tolist(),
            'area': float(area),
            'perimeter': float(perimeter),
            'width': float(w),
            'height': float(h),
            'aspect_ratio': float(aspect_ratio),
            'angles': angles,
            'contour': contour
        }
    
    def _extract_polygon_data(self, contour: np.ndarray, approx: np.ndarray) -> Dict[str, Any]:
        """Extract polygon data from contour."""
        # Get vertices
        vertices = approx.reshape(-1, 2)
        
        # Calculate polygon properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Calculate angles
        angles = []
        for i in range(len(vertices)):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % len(vertices)]
            p3 = vertices[(i + 2) % len(vertices)]
            
            v1 = p1 - p2
            v2 = p3 - p2
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = math.degrees(math.acos(cos_angle))
            angles.append(angle)
        
        return {
            'vertices': vertices.tolist(),
            'area': float(area),
            'perimeter': float(perimeter),
            'vertex_count': len(vertices),
            'angles': angles,
            'contour': contour
        }
    
    def _classify_triangle_type(self, angles: List[float]) -> str:
        """Classify triangle type based on angles."""
        # Check for right triangle
        for angle in angles:
            if abs(angle - 90.0) < self.angle_tolerance:
                return 'right'
        
        # Check for equilateral triangle
        if all(abs(angle - 60.0) < self.angle_tolerance for angle in angles):
            return 'equilateral'
        
        # Check for isosceles triangle
        if len(set(round(angle, 1) for angle in angles)) == 2:
            return 'isosceles'
        
        # Check for acute triangle
        if all(angle < 90.0 for angle in angles):
            return 'acute'
        
        # Check for obtuse triangle
        if any(angle > 90.0 for angle in angles):
            return 'obtuse'
        
        return 'scalene'
    
    def detect_architectural_shapes(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect architectural shapes in the image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing detected architectural elements
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Detect shapes
        shapes = self.detect_shapes(contours)
        
        # Classify shapes by architectural function
        architectural_elements = self._classify_architectural_elements(shapes)
        
        return {
            'shapes': shapes,
            'architectural_elements': architectural_elements,
            'metadata': {
                'total_contours': len(contours),
                'detected_shapes': sum(len(shapes[key]) for key in ['rectangles', 'triangles', 'polygons', 'circles', 'lines'])
            }
        }
    
    def _classify_architectural_elements(self, shapes: Dict[str, Any]) -> Dict[str, Any]:
        """Classify detected shapes into architectural elements."""
        walls = []
        roofs = []
        openings = []
        
        # Classify rectangles as walls or openings
        for rect in shapes['rectangles']:
            aspect_ratio = rect['aspect_ratio']
            area = rect['area']
            
            # Tall and narrow rectangles are likely walls
            if aspect_ratio > 2.0 or aspect_ratio < 0.5:
                walls.append(rect)
            # Medium-sized rectangles are likely openings
            elif 0.5 <= aspect_ratio <= 2.0 and area > 100:
                openings.append(rect)
        
        # Classify triangles as roofs
        for triangle in shapes['triangles']:
            if triangle['triangle_type'] in ['right', 'isosceles']:
                roofs.append(triangle)
        
        # Classify polygons as roofs or walls
        for polygon in shapes['polygons']:
            if polygon['vertex_count'] >= 4:
                # Check if it's roughly triangular (roof)
                if polygon['vertex_count'] <= 6:
                    roofs.append(polygon)
                else:
                    walls.append(polygon)
        
        return {
            'walls': walls,
            'roofs': roofs,
            'openings': openings
        }


def detect_shapes_in_image(image_path: str, **kwargs) -> Dict[str, Any]:
    """
    Convenience function to detect shapes in an image.
    
    Args:
        image_path: Path to the input image
        **kwargs: Additional arguments for ShapeDetector.detect_architectural_shapes()
        
    Returns:
        Shape detection results dictionary
    """
    import cv2
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Detect shapes
    detector = ShapeDetector()
    return detector.detect_architectural_shapes(image, **kwargs)
