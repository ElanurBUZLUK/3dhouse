"""
Edge detection and contour extraction for child sketches.
Specialized for detecting architectural elements like walls, roofs, and openings.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
from scipy import ndimage
from skimage import measure, morphology, segmentation
from skimage.filters import gaussian, sobel, scharr
from skimage.morphology import disk, closing, opening

logger = logging.getLogger(__name__)


class EdgeDetector:
    """Advanced edge detection for architectural sketches."""
    
    def __init__(self, 
                 canny_low: int = 50,
                 canny_high: int = 150,
                 gaussian_sigma: float = 1.0,
                 min_contour_area: int = 100):
        """
        Initialize the edge detector.
        
        Args:
            canny_low: Lower threshold for Canny edge detection
            canny_high: Upper threshold for Canny edge detection
            gaussian_sigma: Sigma for Gaussian blur
            min_contour_area: Minimum area for contour filtering
        """
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.gaussian_sigma = gaussian_sigma
        self.min_contour_area = min_contour_area
        
    def detect_edges(self, image: np.ndarray, 
                    method: str = 'canny',
                    preprocess: bool = True) -> Dict[str, Any]:
        """
        Detect edges in the input image.
        
        Args:
            image: Input image as numpy array
            method: Edge detection method ('canny', 'sobel', 'scharr', 'combined')
            preprocess: Whether to preprocess the image
            
        Returns:
            Dictionary containing:
                - edges: Binary edge image
                - contours: List of detected contours
                - method_used: Method used for detection
                - metadata: Additional processing information
        """
        if preprocess:
            image = self._preprocess_image(image)
            
        if method == 'canny':
            edges, contours = self._detect_canny_edges(image)
        elif method == 'sobel':
            edges, contours = self._detect_sobel_edges(image)
        elif method == 'scharr':
            edges, contours = self._detect_scharr_edges(image)
        elif method == 'combined':
            edges, contours = self._detect_combined_edges(image)
        else:
            raise ValueError(f"Unknown edge detection method: {method}")
            
        # Filter contours by area
        filtered_contours = self._filter_contours(contours)
        
        return {
            'edges': edges,
            'contours': filtered_contours,
            'method_used': method,
            'metadata': {
                'total_contours': len(contours),
                'filtered_contours': len(filtered_contours),
                'preprocessed': preprocess
            }
        }
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better edge detection."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), self.gaussian_sigma)
        
        # Apply histogram equalization for better contrast
        equalized = cv2.equalizeHist(blurred)
        
        return equalized
    
    def _detect_canny_edges(self, image: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Detect edges using Canny algorithm."""
        # Apply Canny edge detection
        edges = cv2.Canny(image, self.canny_low, self.canny_high)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return edges, contours
    
    def _detect_sobel_edges(self, image: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Detect edges using Sobel operator."""
        # Calculate Sobel gradients
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate magnitude
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Normalize and convert to uint8
        magnitude = np.uint8(magnitude / magnitude.max() * 255)
        
        # Threshold to create binary edge image
        _, edges = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return edges, contours
    
    def _detect_scharr_edges(self, image: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Detect edges using Scharr operator."""
        # Calculate Scharr gradients
        scharr_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
        scharr_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)
        
        # Calculate magnitude
        magnitude = np.sqrt(scharr_x**2 + scharr_y**2)
        
        # Normalize and convert to uint8
        magnitude = np.uint8(magnitude / magnitude.max() * 255)
        
        # Threshold to create binary edge image
        _, edges = cv2.threshold(magnitude, 30, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.CHAIN_APPROX_SIMPLE)
        
        return edges, contours
    
    def _detect_combined_edges(self, image: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Detect edges using a combination of methods."""
        # Canny edges
        canny_edges, canny_contours = self._detect_canny_edges(image)
        
        # Sobel edges
        sobel_edges, sobel_contours = self._detect_sobel_edges(image)
        
        # Combine edges using bitwise OR
        combined_edges = cv2.bitwise_or(canny_edges, sobel_edges)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return combined_edges, contours
    
    def _filter_contours(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        """Filter contours by area and other criteria."""
        filtered = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.min_contour_area:
                # Additional filtering based on aspect ratio
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Filter out very thin or very wide contours
                if 0.1 <= aspect_ratio <= 10.0:
                    filtered.append(contour)
                    
        return filtered
    
    def detect_architectural_elements(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect specific architectural elements in the sketch.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing detected elements:
                - walls: Wall contours
                - roofs: Roof contours  
                - openings: Door/window contours
                - edges: All detected edges
        """
        # Detect edges
        edge_result = self.detect_edges(image, method='combined')
        edges = edge_result['edges']
        contours = edge_result['contours']
        
        # Classify contours by shape and position
        walls = []
        roofs = []
        openings = []
        
        for contour in contours:
            # Calculate contour properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Approximate contour to polygon
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Classify based on shape and position
            element_type = self._classify_contour(contour, approx, area, aspect_ratio, x, y, w, h)
            
            if element_type == 'wall':
                walls.append(contour)
            elif element_type == 'roof':
                roofs.append(contour)
            elif element_type == 'opening':
                openings.append(contour)
        
        return {
            'walls': walls,
            'roofs': roofs,
            'openings': openings,
            'edges': edges,
            'all_contours': contours,
            'metadata': {
                'wall_count': len(walls),
                'roof_count': len(roofs),
                'opening_count': len(openings)
            }
        }
    
    def _classify_contour(self, contour: np.ndarray, approx: np.ndarray, 
                         area: float, aspect_ratio: float, 
                         x: int, y: int, w: int, h: int) -> str:
        """Classify a contour as wall, roof, or opening."""
        # Calculate additional properties
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Calculate position in image
        image_height = contour.max(axis=0)[1] - contour.min(axis=0)[1]
        relative_y = y / image_height if image_height > 0 else 0
        
        # Classification rules
        if len(approx) >= 4 and solidity > 0.8:
            # Likely a rectangular opening
            if aspect_ratio > 0.5 and aspect_ratio < 2.0 and relative_y > 0.3:
                return 'opening'
        
        if len(approx) >= 3 and solidity > 0.7:
            # Likely a roof (triangular or polygonal)
            if relative_y < 0.4 and aspect_ratio > 1.0:
                return 'roof'
        
        if len(approx) >= 4 and solidity > 0.6:
            # Likely a wall (rectangular)
            if aspect_ratio > 0.3 and aspect_ratio < 3.0 and relative_y > 0.2:
                return 'wall'
        
        # Default classification based on position
        if relative_y < 0.3:
            return 'roof'
        elif relative_y > 0.7:
            return 'wall'
        else:
            return 'opening'


def detect_sketch_edges(image_path: str, **kwargs) -> Dict[str, Any]:
    """
    Convenience function to detect edges in a sketch image.
    
    Args:
        image_path: Path to the input image
        **kwargs: Additional arguments for EdgeDetector.detect_edges()
        
    Returns:
        Edge detection results dictionary
    """
    import cv2
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Detect edges
    detector = EdgeDetector()
    return detector.detect_edges(image, **kwargs)
