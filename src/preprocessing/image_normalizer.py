"""
Image normalization and preprocessing for child sketches.
Handles perspective correction, noise reduction, and standardization.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
from PIL import Image, ImageEnhance, ImageFilter
import logging

logger = logging.getLogger(__name__)


class ImageNormalizer:
    """Normalizes child sketch images for consistent processing."""
    
    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        """
        Initialize the image normalizer.
        
        Args:
            target_size: Target size for normalized images (width, height)
        """
        self.target_size = target_size
        self.aspect_ratio_threshold = 0.3  # Minimum aspect ratio to avoid extreme distortions
        
    def normalize(self, image: np.ndarray, 
                 enhance_contrast: bool = True,
                 reduce_noise: bool = True,
                 correct_perspective: bool = True) -> Dict[str, Any]:
        """
        Normalize a child sketch image.
        
        Args:
            image: Input image as numpy array (H, W, C) or (H, W)
            enhance_contrast: Whether to enhance contrast
            reduce_noise: Whether to reduce noise
            correct_perspective: Whether to attempt perspective correction
            
        Returns:
            Dictionary containing:
                - normalized_image: Preprocessed image
                - original_size: Original image dimensions
                - scale_factor: Scale factor applied
                - perspective_correction: Whether perspective was corrected
                - metadata: Additional processing metadata
        """
        original_size = image.shape[:2]
        metadata = {}
        
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image.copy()
            
        # Convert to PIL for easier processing
        pil_image = Image.fromarray(image_rgb)
        
        # Enhance contrast if requested
        if enhance_contrast:
            pil_image = self._enhance_contrast(pil_image)
            metadata['contrast_enhanced'] = True
            
        # Reduce noise if requested
        if reduce_noise:
            pil_image = self._reduce_noise(pil_image)
            metadata['noise_reduced'] = True
            
        # Convert back to numpy
        image_processed = np.array(pil_image)
        
        # Correct perspective if requested
        perspective_corrected = False
        if correct_perspective:
            image_processed, perspective_corrected = self._correct_perspective(image_processed)
            metadata['perspective_corrected'] = perspective_corrected
            
        # Resize to target size while maintaining aspect ratio
        normalized_image, scale_factor = self._resize_with_aspect_ratio(
            image_processed, self.target_size
        )
        
        # Final normalization
        normalized_image = self._final_normalization(normalized_image)
        
        return {
            'normalized_image': normalized_image,
            'original_size': original_size,
            'scale_factor': scale_factor,
            'perspective_corrected': perspective_corrected,
            'metadata': metadata
        }
    
    def _enhance_contrast(self, image: Image.Image) -> Image.Image:
        """Enhance contrast of the image."""
        # Convert to grayscale for contrast enhancement
        gray = image.convert('L')
        
        # Apply histogram equalization
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(1.5)
        
        # Convert back to RGB
        return Image.merge('RGB', (enhanced, enhanced, enhanced))
    
    def _reduce_noise(self, image: Image.Image) -> Image.Image:
        """Reduce noise in the image."""
        # Apply bilateral filter to reduce noise while preserving edges
        image_array = np.array(image)
        
        # Convert to grayscale for noise reduction
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
            
        # Apply bilateral filter
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Convert back to RGB
        if len(image_array.shape) == 3:
            return Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB))
        else:
            return Image.fromarray(denoised)
    
    def _correct_perspective(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Attempt to correct perspective distortion."""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
                
            # Detect edges
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return image, False
                
            # Find the largest contour (likely the main shape)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Approximate the contour
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # Check if we have a quadrilateral
            if len(approx) == 4:
                # Order the points (top-left, top-right, bottom-right, bottom-left)
                ordered_points = self._order_points(approx.reshape(4, 2))
                
                # Calculate the perspective transform
                width = max(np.linalg.norm(ordered_points[1] - ordered_points[0]),
                           np.linalg.norm(ordered_points[3] - ordered_points[2]))
                height = max(np.linalg.norm(ordered_points[2] - ordered_points[1]),
                            np.linalg.norm(ordered_points[0] - ordered_points[3]))
                
                # Check aspect ratio to avoid extreme distortions
                aspect_ratio = min(width, height) / max(width, height)
                if aspect_ratio < self.aspect_ratio_threshold:
                    return image, False
                
                # Define destination points
                dst = np.array([
                    [0, 0],
                    [width - 1, 0],
                    [width - 1, height - 1],
                    [0, height - 1]
                ], dtype=np.float32)
                
                # Calculate perspective transform matrix
                matrix = cv2.getPerspectiveTransform(ordered_points.astype(np.float32), dst)
                
                # Apply perspective correction
                corrected = cv2.warpPerspective(image, matrix, (int(width), int(height)))
                
                return corrected, True
            else:
                return image, False
                
        except Exception as e:
            logger.warning(f"Perspective correction failed: {e}")
            return image, False
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points in the format: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Sum and difference of coordinates
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        # Top-left point has smallest sum
        rect[0] = pts[np.argmin(s)]
        # Bottom-right point has largest sum
        rect[2] = pts[np.argmax(s)]
        # Top-right point has smallest difference
        rect[1] = pts[np.argmin(diff)]
        # Bottom-left point has largest difference
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def _resize_with_aspect_ratio(self, image: np.ndarray, 
                                 target_size: Tuple[int, int]) -> Tuple[np.ndarray, float]:
        """Resize image while maintaining aspect ratio."""
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scale factor
        scale_w = target_w / w
        scale_h = target_h / h
        scale = min(scale_w, scale_h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create padded image
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Calculate padding offsets
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        # Place resized image in center
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return padded, scale
    
    def _final_normalization(self, image: np.ndarray) -> np.ndarray:
        """Apply final normalization to the image."""
        # Convert to float and normalize to [0, 1]
        normalized = image.astype(np.float32) / 255.0
        
        # Apply slight sharpening to enhance edges
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(normalized, -1, kernel)
        
        # Clamp values to [0, 1]
        sharpened = np.clip(sharpened, 0, 1)
        
        # Convert back to uint8
        return (sharpened * 255).astype(np.uint8)


def normalize_sketch_image(image_path: str, 
                          target_size: Tuple[int, int] = (256, 256),
                          **kwargs) -> Dict[str, Any]:
    """
    Convenience function to normalize a sketch image from file path.
    
    Args:
        image_path: Path to the input image
        target_size: Target size for normalized image
        **kwargs: Additional arguments for ImageNormalizer.normalize()
        
    Returns:
        Normalization results dictionary
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Normalize
    normalizer = ImageNormalizer(target_size)
    return normalizer.normalize(image, **kwargs)
