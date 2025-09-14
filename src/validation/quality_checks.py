"""
Quality validation and testing for Sketch2House3D.
Provides comprehensive validation for 3D models, segmentation results, and geometry.
"""

import numpy as np
import trimesh
import cv2
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from enum import Enum
import json
import time

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"


class ValidationResult(Enum):
    """Validation results."""
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


@dataclass
class ValidationCheck:
    """Individual validation check result."""
    name: str
    result: ValidationResult
    message: str
    value: Any
    threshold: Optional[Any] = None
    severity: str = "medium"
    metadata: Dict[str, Any] = None


@dataclass
class ValidationReport:
    """Complete validation report."""
    timestamp: float
    validation_level: ValidationLevel
    overall_result: ValidationResult
    checks: List[ValidationCheck]
    summary: Dict[str, Any]
    recommendations: List[str]


class GeometryValidator:
    """Validates 3D geometry quality."""
    
    def __init__(self, 
                 min_face_count: int = 100,
                 max_face_count: int = 100000,
                 min_volume: float = 0.001,
                 max_aspect_ratio: float = 100.0):
        """
        Initialize geometry validator.
        
        Args:
            min_face_count: Minimum number of faces
            max_face_count: Maximum number of faces
            min_volume: Minimum volume in cubic meters
            max_aspect_ratio: Maximum aspect ratio for faces
        """
        self.min_face_count = min_face_count
        self.max_face_count = max_face_count
        self.min_volume = min_volume
        self.max_aspect_ratio = max_aspect_ratio
    
    def validate_mesh(self, mesh: trimesh.Trimesh) -> List[ValidationCheck]:
        """Validate mesh geometry."""
        checks = []
        
        # Face count check
        face_count = len(mesh.faces)
        if face_count < self.min_face_count:
            checks.append(ValidationCheck(
                name="face_count_min",
                result=ValidationResult.FAIL,
                message=f"Too few faces: {face_count} < {self.min_face_count}",
                value=face_count,
                threshold=self.min_face_count,
                severity="high"
            ))
        elif face_count > self.max_face_count:
            checks.append(ValidationCheck(
                name="face_count_max",
                result=ValidationResult.WARNING,
                message=f"Too many faces: {face_count} > {self.max_face_count}",
                value=face_count,
                threshold=self.max_face_count,
                severity="medium"
            ))
        else:
            checks.append(ValidationCheck(
                name="face_count",
                result=ValidationResult.PASS,
                message=f"Face count is acceptable: {face_count}",
                value=face_count,
                severity="low"
            ))
        
        # Volume check
        volume = mesh.volume
        if volume < self.min_volume:
            checks.append(ValidationCheck(
                name="volume_min",
                result=ValidationResult.FAIL,
                message=f"Volume too small: {volume:.6f} < {self.min_volume}",
                value=volume,
                threshold=self.min_volume,
                severity="high"
            ))
        else:
            checks.append(ValidationCheck(
                name="volume",
                result=ValidationResult.PASS,
                message=f"Volume is acceptable: {volume:.6f}",
                value=volume,
                severity="low"
            ))
        
        # Watertight check
        is_watertight = mesh.is_watertight
        checks.append(ValidationCheck(
            name="watertight",
            result=ValidationResult.PASS if is_watertight else ValidationResult.WARNING,
            message="Mesh is watertight" if is_watertight else "Mesh is not watertight",
            value=is_watertight,
            severity="medium"
        ))
        
        # Face aspect ratio check
        aspect_ratios = self._calculate_face_aspect_ratios(mesh)
        max_ratio = np.max(aspect_ratios) if len(aspect_ratios) > 0 else 0
        
        if max_ratio > self.max_aspect_ratio:
            checks.append(ValidationCheck(
                name="aspect_ratio",
                result=ValidationResult.WARNING,
                message=f"High aspect ratio faces detected: {max_ratio:.2f} > {self.max_aspect_ratio}",
                value=max_ratio,
                threshold=self.max_aspect_ratio,
                severity="medium"
            ))
        else:
            checks.append(ValidationCheck(
                name="aspect_ratio",
                result=ValidationResult.PASS,
                message=f"Face aspect ratios are acceptable: {max_ratio:.2f}",
                value=max_ratio,
                severity="low"
            ))
        
        # Normal consistency check
        normal_consistency = self._check_normal_consistency(mesh)
        checks.append(ValidationCheck(
            name="normal_consistency",
            result=ValidationResult.PASS if normal_consistency > 0.8 else ValidationResult.WARNING,
            message=f"Normal consistency: {normal_consistency:.2f}",
            value=normal_consistency,
            threshold=0.8,
            severity="medium"
        ))
        
        return checks
    
    def _calculate_face_aspect_ratios(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Calculate aspect ratios for all faces."""
        if len(mesh.faces) == 0:
            return np.array([])
        
        # Get face vertices
        face_vertices = mesh.vertices[mesh.faces]
        
        # Calculate edge lengths for each face
        edge_lengths = []
        for face in face_vertices:
            # Calculate three edge lengths
            edge1 = np.linalg.norm(face[1] - face[0])
            edge2 = np.linalg.norm(face[2] - face[1])
            edge3 = np.linalg.norm(face[0] - face[2])
            
            # Calculate aspect ratio (max/min edge length)
            edges = [edge1, edge2, edge3]
            aspect_ratio = max(edges) / min(edges) if min(edges) > 0 else 0
            edge_lengths.append(aspect_ratio)
        
        return np.array(edge_lengths)
    
    def _check_normal_consistency(self, mesh: trimesh.Trimesh) -> float:
        """Check normal consistency (0-1, higher is better)."""
        if len(mesh.faces) == 0:
            return 1.0
        
        # Get face normals
        face_normals = mesh.face_normals
        
        # Calculate normal consistency
        # This is a simplified check - in practice, you'd want more sophisticated normal validation
        normal_lengths = np.linalg.norm(face_normals, axis=1)
        valid_normals = np.sum(normal_lengths > 0.9)  # Normals should be close to unit length
        
        return valid_normals / len(face_normals) if len(face_normals) > 0 else 1.0


class SegmentationValidator:
    """Validates segmentation results."""
    
    def __init__(self, 
                 min_class_coverage: float = 0.01,
                 max_class_coverage: float = 0.95,
                 min_confidence: float = 0.5):
        """
        Initialize segmentation validator.
        
        Args:
            min_class_coverage: Minimum coverage for each class
            max_class_coverage: Maximum coverage for each class
            min_confidence: Minimum confidence threshold
        """
        self.min_class_coverage = min_class_coverage
        self.max_class_coverage = max_class_coverage
        self.min_confidence = min_confidence
    
    def validate_segmentation(self, 
                            segmentation_result: Dict[str, Any]) -> List[ValidationCheck]:
        """Validate segmentation results."""
        checks = []
        
        # Get segmentation mask
        segmentation_mask = segmentation_result.get('segmentation_mask')
        if segmentation_mask is None:
            checks.append(ValidationCheck(
                name="segmentation_mask_exists",
                result=ValidationResult.FAIL,
                message="No segmentation mask found",
                value=False,
                severity="high"
            ))
            return checks
        
        # Check mask dimensions
        height, width = segmentation_mask.shape[:2]
        if height < 64 or width < 64:
            checks.append(ValidationCheck(
                name="mask_resolution",
                result=ValidationResult.WARNING,
                message=f"Low resolution mask: {height}x{width}",
                value=(height, width),
                threshold=(64, 64),
                severity="medium"
            ))
        else:
            checks.append(ValidationCheck(
                name="mask_resolution",
                result=ValidationResult.PASS,
                message=f"Mask resolution is acceptable: {height}x{width}",
                value=(height, width),
                severity="low"
            ))
        
        # Check class coverage
        class_coverage = self._calculate_class_coverage(segmentation_mask)
        for class_name, coverage in class_coverage.items():
            if coverage < self.min_class_coverage:
                checks.append(ValidationCheck(
                    name=f"class_coverage_{class_name}",
                    result=ValidationResult.WARNING,
                    message=f"Low coverage for {class_name}: {coverage:.3f} < {self.min_class_coverage}",
                    value=coverage,
                    threshold=self.min_class_coverage,
                    severity="medium"
                ))
            elif coverage > self.max_class_coverage:
                checks.append(ValidationCheck(
                    name=f"class_coverage_{class_name}",
                    result=ValidationResult.WARNING,
                    message=f"High coverage for {class_name}: {coverage:.3f} > {self.max_class_coverage}",
                    value=coverage,
                    threshold=self.max_class_coverage,
                    severity="medium"
                ))
            else:
                checks.append(ValidationCheck(
                    name=f"class_coverage_{class_name}",
                    result=ValidationResult.PASS,
                    message=f"Coverage for {class_name} is acceptable: {coverage:.3f}",
                    value=coverage,
                    severity="low"
                ))
        
        # Check confidence scores
        confidence_scores = segmentation_result.get('confidence_scores')
        if confidence_scores is not None:
            avg_confidence = np.mean(confidence_scores)
            if avg_confidence < self.min_confidence:
                checks.append(ValidationCheck(
                    name="confidence_score",
                    result=ValidationResult.WARNING,
                    message=f"Low confidence score: {avg_confidence:.3f} < {self.min_confidence}",
                    value=avg_confidence,
                    threshold=self.min_confidence,
                    severity="medium"
                ))
            else:
                checks.append(ValidationCheck(
                    name="confidence_score",
                    result=ValidationResult.PASS,
                    message=f"Confidence score is acceptable: {avg_confidence:.3f}",
                    value=avg_confidence,
                    severity="low"
                ))
        
        return checks
    
    def _calculate_class_coverage(self, segmentation_mask: np.ndarray) -> Dict[str, float]:
        """Calculate coverage for each class."""
        total_pixels = segmentation_mask.size
        
        class_coverage = {}
        unique_classes = np.unique(segmentation_mask)
        
        for class_id in unique_classes:
            class_pixels = np.sum(segmentation_mask == class_id)
            coverage = class_pixels / total_pixels
            class_coverage[f"class_{class_id}"] = coverage
        
        return class_coverage


class LayoutValidator:
    """Validates layout parameters and constraints."""
    
    def __init__(self, 
                 min_wall_length: float = 0.5,
                 max_wall_length: float = 50.0,
                 min_opening_area: float = 0.1,
                 max_opening_area: float = 10.0):
        """
        Initialize layout validator.
        
        Args:
            min_wall_length: Minimum wall length in meters
            max_wall_length: Maximum wall length in meters
            min_opening_area: Minimum opening area in square meters
            max_opening_area: Maximum opening area in square meters
        """
        self.min_wall_length = min_wall_length
        self.max_wall_length = max_wall_length
        self.min_opening_area = min_opening_area
        self.max_opening_area = max_opening_area
    
    def validate_layout(self, layout_params: Dict[str, Any]) -> List[ValidationCheck]:
        """Validate layout parameters."""
        checks = []
        
        # Check facade polygon
        facade_polygon = layout_params.get('facade_polygon', [])
        if len(facade_polygon) < 3:
            checks.append(ValidationCheck(
                name="facade_polygon",
                result=ValidationResult.FAIL,
                message=f"Invalid facade polygon: {len(facade_polygon)} points",
                value=len(facade_polygon),
                threshold=3,
                severity="high"
            ))
        else:
            checks.append(ValidationCheck(
                name="facade_polygon",
                result=ValidationResult.PASS,
                message=f"Facade polygon is valid: {len(facade_polygon)} points",
                value=len(facade_polygon),
                severity="low"
            ))
        
        # Check wall dimensions
        wall_length = self._calculate_wall_length(facade_polygon)
        if wall_length < self.min_wall_length:
            checks.append(ValidationCheck(
                name="wall_length_min",
                result=ValidationResult.FAIL,
                message=f"Wall too short: {wall_length:.2f}m < {self.min_wall_length}m",
                value=wall_length,
                threshold=self.min_wall_length,
                severity="high"
            ))
        elif wall_length > self.max_wall_length:
            checks.append(ValidationCheck(
                name="wall_length_max",
                result=ValidationResult.WARNING,
                message=f"Wall too long: {wall_length:.2f}m > {self.max_wall_length}m",
                value=wall_length,
                threshold=self.max_wall_length,
                severity="medium"
            ))
        else:
            checks.append(ValidationCheck(
                name="wall_length",
                result=ValidationResult.PASS,
                message=f"Wall length is acceptable: {wall_length:.2f}m",
                value=wall_length,
                severity="low"
            ))
        
        # Check openings
        doors = layout_params.get('doors', [])
        windows = layout_params.get('windows', [])
        
        for i, door in enumerate(doors):
            door_area = self._calculate_opening_area(door)
            if door_area < self.min_opening_area:
                checks.append(ValidationCheck(
                    name=f"door_area_min_{i}",
                    result=ValidationResult.WARNING,
                    message=f"Door {i} too small: {door_area:.2f}m² < {self.min_opening_area}m²",
                    value=door_area,
                    threshold=self.min_opening_area,
                    severity="medium"
                ))
            elif door_area > self.max_opening_area:
                checks.append(ValidationCheck(
                    name=f"door_area_max_{i}",
                    result=ValidationResult.WARNING,
                    message=f"Door {i} too large: {door_area:.2f}m² > {self.max_opening_area}m²",
                    value=door_area,
                    threshold=self.max_opening_area,
                    severity="medium"
                ))
            else:
                checks.append(ValidationCheck(
                    name=f"door_area_{i}",
                    result=ValidationResult.PASS,
                    message=f"Door {i} area is acceptable: {door_area:.2f}m²",
                    value=door_area,
                    severity="low"
                ))
        
        for i, window in enumerate(windows):
            window_area = self._calculate_opening_area(window)
            if window_area < self.min_opening_area:
                checks.append(ValidationCheck(
                    name=f"window_area_min_{i}",
                    result=ValidationResult.WARNING,
                    message=f"Window {i} too small: {window_area:.2f}m² < {self.min_opening_area}m²",
                    value=window_area,
                    threshold=self.min_opening_area,
                    severity="medium"
                ))
            elif window_area > self.max_opening_area:
                checks.append(ValidationCheck(
                    name=f"window_area_max_{i}",
                    result=ValidationResult.WARNING,
                    message=f"Window {i} too large: {window_area:.2f}m² > {self.max_opening_area}m²",
                    value=window_area,
                    threshold=self.max_opening_area,
                    severity="medium"
                ))
            else:
                checks.append(ValidationCheck(
                    name=f"window_area_{i}",
                    result=ValidationResult.PASS,
                    message=f"Window {i} area is acceptable: {window_area:.2f}m²",
                    value=window_area,
                    severity="low"
                ))
        
        return checks
    
    def _calculate_wall_length(self, facade_polygon: List[List[float]]) -> float:
        """Calculate total wall length."""
        if len(facade_polygon) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(facade_polygon)):
            p1 = np.array(facade_polygon[i])
            p2 = np.array(facade_polygon[(i + 1) % len(facade_polygon)])
            total_length += np.linalg.norm(p2 - p1)
        
        return total_length
    
    def _calculate_opening_area(self, opening: Dict[str, Any]) -> float:
        """Calculate opening area."""
        size_m = opening.get('size_m', [0, 0])
        if len(size_m) >= 2:
            return size_m[0] * size_m[1]
        return 0.0


class QualityValidator:
    """Main quality validator that combines all validation checks."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        """
        Initialize quality validator.
        
        Args:
            validation_level: Validation level to use
        """
        self.validation_level = validation_level
        self.geometry_validator = GeometryValidator()
        self.segmentation_validator = SegmentationValidator()
        self.layout_validator = LayoutValidator()
    
    def validate_model(self, 
                      model_data: Dict[str, Any],
                      segmentation_result: Optional[Dict[str, Any]] = None,
                      layout_params: Optional[Dict[str, Any]] = None) -> ValidationReport:
        """
        Validate complete model.
        
        Args:
            model_data: 3D model data
            segmentation_result: Segmentation results
            layout_params: Layout parameters
            
        Returns:
            Validation report
        """
        checks = []
        
        # Validate 3D geometry
        for element_name, element_data in model_data.items():
            if isinstance(element_data, dict) and 'mesh' in element_data:
                mesh = element_data['mesh']
                if isinstance(mesh, trimesh.Trimesh):
                    element_checks = self.geometry_validator.validate_mesh(mesh)
                    # Add element name to check names
                    for check in element_checks:
                        check.name = f"{element_name}_{check.name}"
                    checks.extend(element_checks)
        
        # Validate segmentation if provided
        if segmentation_result is not None:
            seg_checks = self.segmentation_validator.validate_segmentation(segmentation_result)
            checks.extend(seg_checks)
        
        # Validate layout if provided
        if layout_params is not None:
            layout_checks = self.layout_validator.validate_layout(layout_params)
            checks.extend(layout_checks)
        
        # Determine overall result
        overall_result = self._determine_overall_result(checks)
        
        # Generate summary
        summary = self._generate_summary(checks)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(checks)
        
        return ValidationReport(
            timestamp=time.time(),
            validation_level=self.validation_level,
            overall_result=overall_result,
            checks=checks,
            summary=summary,
            recommendations=recommendations
        )
    
    def _determine_overall_result(self, checks: List[ValidationCheck]) -> ValidationResult:
        """Determine overall validation result."""
        if not checks:
            return ValidationResult.PASS
        
        # Count results
        fail_count = sum(1 for check in checks if check.result == ValidationResult.FAIL)
        warning_count = sum(1 for check in checks if check.result == ValidationResult.WARNING)
        
        if fail_count > 0:
            return ValidationResult.FAIL
        elif warning_count > 0:
            return ValidationResult.WARNING
        else:
            return ValidationResult.PASS
    
    def _generate_summary(self, checks: List[ValidationCheck]) -> Dict[str, Any]:
        """Generate validation summary."""
        total_checks = len(checks)
        pass_checks = sum(1 for check in checks if check.result == ValidationResult.PASS)
        warning_checks = sum(1 for check in checks if check.result == ValidationResult.WARNING)
        fail_checks = sum(1 for check in checks if check.result == ValidationResult.FAIL)
        
        return {
            'total_checks': total_checks,
            'pass_checks': pass_checks,
            'warning_checks': warning_checks,
            'fail_checks': fail_checks,
            'pass_rate': pass_checks / total_checks if total_checks > 0 else 0.0
        }
    
    def _generate_recommendations(self, checks: List[ValidationCheck]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        for check in checks:
            if check.result == ValidationResult.FAIL:
                if "face_count" in check.name:
                    recommendations.append("Increase mesh resolution or add more geometry")
                elif "volume" in check.name:
                    recommendations.append("Check geometry scaling and ensure proper dimensions")
                elif "watertight" in check.name:
                    recommendations.append("Fix mesh holes and ensure watertight geometry")
            elif check.result == ValidationResult.WARNING:
                if "aspect_ratio" in check.name:
                    recommendations.append("Consider subdividing high aspect ratio faces")
                elif "confidence" in check.name:
                    recommendations.append("Improve input image quality or adjust model parameters")
                elif "coverage" in check.name:
                    recommendations.append("Check segmentation results and adjust thresholds")
        
        return recommendations


def validate_model_quality(model_data: Dict[str, Any], **kwargs) -> ValidationReport:
    """
    Convenience function to validate model quality.
    
    Args:
        model_data: 3D model data
        **kwargs: Additional arguments for validation
        
    Returns:
        Validation report
    """
    validator = QualityValidator(**kwargs)
    return validator.validate_model(model_data)
