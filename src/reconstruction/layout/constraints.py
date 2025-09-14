"""
Constraint definitions and validation for architectural layouts.
Ensures geometric and architectural constraints are satisfied.
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Optional, Callable
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ConstraintType(Enum):
    """Types of constraints."""
    GEOMETRIC = "geometric"
    ARCHITECTURAL = "architectural"
    PHYSICAL = "physical"
    AESTHETIC = "aesthetic"


@dataclass
class Constraint:
    """Represents a single constraint."""
    name: str
    constraint_type: ConstraintType
    description: str
    validator: Callable
    weight: float = 1.0
    required: bool = True


class ConstraintValidator:
    """Validates architectural layout constraints."""
    
    def __init__(self):
        """Initialize constraint validator."""
        self.constraints = self._define_constraints()
    
    def _define_constraints(self) -> List[Constraint]:
        """Define all architectural constraints."""
        constraints = []
        
        # Geometric constraints
        constraints.append(Constraint(
            name="opening_within_wall",
            constraint_type=ConstraintType.GEOMETRIC,
            description="Openings must be within wall boundaries",
            validator=self._validate_opening_within_wall,
            weight=2.0,
            required=True
        ))
        
        constraints.append(Constraint(
            name="no_overlapping_openings",
            constraint_type=ConstraintType.GEOMETRIC,
            description="Openings must not overlap",
            validator=self._validate_no_overlapping_openings,
            weight=1.5,
            required=True
        ))
        
        constraints.append(Constraint(
            name="door_touches_ground",
            constraint_type=ConstraintType.ARCHITECTURAL,
            description="Doors must touch the ground",
            validator=self._validate_door_touches_ground,
            weight=2.0,
            required=True
        ))
        
        constraints.append(Constraint(
            name="window_height_constraint",
            constraint_type=ConstraintType.ARCHITECTURAL,
            description="Windows must be at reasonable height",
            validator=self._validate_window_height,
            weight=1.0,
            required=False
        ))
        
        constraints.append(Constraint(
            name="roof_above_walls",
            constraint_type=ConstraintType.ARCHITECTURAL,
            description="Roof must be above walls",
            validator=self._validate_roof_above_walls,
            weight=1.5,
            required=True
        ))
        
        constraints.append(Constraint(
            name="minimum_wall_thickness",
            constraint_type=ConstraintType.PHYSICAL,
            description="Walls must have minimum thickness",
            validator=self._validate_minimum_wall_thickness,
            weight=1.0,
            required=True
        ))
        
        constraints.append(Constraint(
            name="opening_size_constraints",
            constraint_type=ConstraintType.ARCHITECTURAL,
            description="Openings must have reasonable dimensions",
            validator=self._validate_opening_sizes,
            weight=1.0,
            required=False
        ))
        
        constraints.append(Constraint(
            name="facade_proportions",
            constraint_type=ConstraintType.AESTHETIC,
            description="Facade must have reasonable proportions",
            validator=self._validate_facade_proportions,
            weight=0.5,
            required=False
        ))
        
        return constraints
    
    def validate_layout(self, layout_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate layout parameters against all constraints.
        
        Args:
            layout_params: Layout parameters to validate
            
        Returns:
            Validation results
        """
        results = {
            'valid': True,
            'constraint_results': {},
            'violations': [],
            'warnings': [],
            'score': 0.0
        }
        
        total_weight = 0.0
        weighted_score = 0.0
        
        for constraint in self.constraints:
            try:
                is_valid, message, score = constraint.validator(layout_params)
                
                results['constraint_results'][constraint.name] = {
                    'valid': is_valid,
                    'message': message,
                    'score': score,
                    'weight': constraint.weight,
                    'required': constraint.required
                }
                
                if not is_valid:
                    if constraint.required:
                        results['valid'] = False
                        results['violations'].append({
                            'constraint': constraint.name,
                            'message': message,
                            'type': constraint.constraint_type.value
                        })
                    else:
                        results['warnings'].append({
                            'constraint': constraint.name,
                            'message': message,
                            'type': constraint.constraint_type.value
                        })
                
                # Calculate weighted score
                weighted_score += score * constraint.weight
                total_weight += constraint.weight
                
            except Exception as e:
                logger.error(f"Error validating constraint {constraint.name}: {e}")
                results['constraint_results'][constraint.name] = {
                    'valid': False,
                    'message': f"Validation error: {str(e)}",
                    'score': 0.0,
                    'weight': constraint.weight,
                    'required': constraint.required
                }
        
        # Calculate overall score
        if total_weight > 0:
            results['score'] = weighted_score / total_weight
        else:
            results['score'] = 0.0
        
        return results
    
    def _validate_opening_within_wall(self, layout_params: Dict[str, Any]) -> Tuple[bool, str, float]:
        """Validate that openings are within wall boundaries."""
        facade_polygon = layout_params.get('facade_polygon', [])
        openings = layout_params.get('openings', [])
        
        if not facade_polygon or not openings:
            return True, "No openings to validate", 1.0
        
        facade_polygon_np = np.array(facade_polygon)
        violations = 0
        
        for opening in openings:
            bbox = opening.get('bbox_px', [])
            if len(bbox) != 4:
                continue
            
            x_min, y_min, x_max, y_max = bbox
            
            # Check if all corners are within facade
            corners = [
                (x_min, y_min),
                (x_max, y_min),
                (x_max, y_max),
                (x_min, y_max)
            ]
            
            for corner in corners:
                if cv2.pointPolygonTest(facade_polygon_np, corner, False) < 0:
                    violations += 1
                    break
        
        if violations == 0:
            return True, "✅ Tüm açıklıklar duvar sınırları içinde", 1.0
        else:
            score = max(0.0, 1.0 - violations / len(openings))
            return False, f"❌ {violations} açıklık duvar sınırları dışında. Lütfen kapı ve pencereleri duvar içine taşıyın.", score
    
    def _validate_no_overlapping_openings(self, layout_params: Dict[str, Any]) -> Tuple[bool, str, float]:
        """Validate that openings do not overlap."""
        openings = layout_params.get('openings', [])
        
        if len(openings) < 2:
            return True, "Not enough openings to check overlap", 1.0
        
        overlaps = 0
        total_pairs = 0
        
        for i in range(len(openings)):
            for j in range(i + 1, len(openings)):
                total_pairs += 1
                
                bbox1 = openings[i].get('bbox_px', [])
                bbox2 = openings[j].get('bbox_px', [])
                
                if len(bbox1) != 4 or len(bbox2) != 4:
                    continue
                
                if self._bboxes_overlap(bbox1, bbox2):
                    overlaps += 1
        
        if overlaps == 0:
            return True, "✅ Açıklıklar çakışmıyor", 1.0
        else:
            score = max(0.0, 1.0 - overlaps / total_pairs)
            return False, f"❌ {overlaps} çift açıklık çakışıyor. Lütfen kapı ve pencereleri birbirinden uzaklaştırın.", score
    
    def _bboxes_overlap(self, bbox1: List[float], bbox2: List[float]) -> bool:
        """Check if two bounding boxes overlap."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        return not (x1_max <= x2_min or x2_max <= x1_min or 
                   y1_max <= y2_min or y2_max <= y1_min)
    
    def _validate_door_touches_ground(self, layout_params: Dict[str, Any]) -> Tuple[bool, str, float]:
        """Validate that doors touch the ground."""
        facade_polygon = layout_params.get('facade_polygon', [])
        openings = layout_params.get('openings', [])
        
        if not facade_polygon:
            return True, "No facade to validate against", 1.0
        
        facade_polygon_np = np.array(facade_polygon)
        facade_bottom = np.max(facade_polygon_np[:, 1])
        
        doors = [opening for opening in openings if opening.get('kind') == 'door']
        
        if not doors:
            return True, "No doors to validate", 1.0
        
        violations = 0
        tolerance = 10  # pixels
        
        for door in doors:
            bbox = door.get('bbox_px', [])
            if len(bbox) != 4:
                continue
            
            y_max = bbox[3]
            if abs(y_max - facade_bottom) > tolerance:
                violations += 1
        
        if violations == 0:
            return True, "✅ Tüm kapılar zemine değiyor", 1.0
        else:
            score = max(0.0, 1.0 - violations / len(doors))
            return False, f"❌ {violations} kapı zemine değmiyor. Kapıları zemine hizalayın.", score
    
    def _validate_window_height(self, layout_params: Dict[str, Any]) -> Tuple[bool, str, float]:
        """Validate that windows are at reasonable height."""
        facade_polygon = layout_params.get('facade_polygon', [])
        openings = layout_params.get('openings', [])
        
        if not facade_polygon:
            return True, "No facade to validate against", 1.0
        
        facade_polygon_np = np.array(facade_polygon)
        facade_top = np.min(facade_polygon_np[:, 1])
        facade_bottom = np.max(facade_polygon_np[:, 1])
        facade_height = facade_bottom - facade_top
        
        windows = [opening for opening in openings if opening.get('kind') == 'window']
        
        if not windows:
            return True, "No windows to validate", 1.0
        
        violations = 0
        min_height_ratio = 0.1  # Windows should be at least 10% from bottom
        max_height_ratio = 0.9  # Windows should be at most 90% from bottom
        
        for window in windows:
            bbox = window.get('bbox_px', [])
            if len(bbox) != 4:
                continue
            
            y_min = bbox[1]
            relative_height = (y_min - facade_top) / facade_height
            
            if relative_height < min_height_ratio or relative_height > max_height_ratio:
                violations += 1
        
        if violations == 0:
            return True, "All windows are at reasonable height", 1.0
        else:
            score = max(0.0, 1.0 - violations / len(windows))
            return False, f"{violations} windows are at unreasonable height", score
    
    def _validate_roof_above_walls(self, layout_params: Dict[str, Any]) -> Tuple[bool, str, float]:
        """Validate that roof is above walls."""
        facade_polygon = layout_params.get('facade_polygon', [])
        roof = layout_params.get('roof', {})
        
        if not facade_polygon or not roof:
            return True, "No roof or facade to validate", 1.0
        
        facade_polygon_np = np.array(facade_polygon)
        facade_top = np.min(facade_polygon_np[:, 1])
        
        # Check if roof is above facade
        ridge_line = roof.get('ridge_line_px', [])
        if ridge_line:
            ridge_y = min(point[1] for point in ridge_line)
            if ridge_y < facade_top:
                return False, "Roof is below facade", 0.0
        
        return True, "Roof is above facade", 1.0
    
    def _validate_minimum_wall_thickness(self, layout_params: Dict[str, Any]) -> Tuple[bool, str, float]:
        """Validate minimum wall thickness."""
        facade_polygon = layout_params.get('facade_polygon', [])
        scale_factor = layout_params.get('scale_m_per_px', 0.01)
        
        if not facade_polygon:
            return True, "No facade to validate", 1.0
        
        # This is a simplified check - in practice, you'd need to analyze wall thickness
        # For now, we'll assume the facade polygon represents the outer wall
        facade_polygon_np = np.array(facade_polygon)
        
        # Calculate approximate wall thickness (simplified)
        # In practice, you'd need to analyze the actual wall structure
        min_thickness_px = 10  # pixels
        min_thickness_m = min_thickness_px * scale_factor
        
        # For now, assume walls are thick enough if facade is reasonably sized
        facade_width = np.max(facade_polygon_np[:, 0]) - np.min(facade_polygon_np[:, 0])
        facade_height = np.max(facade_polygon_np[:, 1]) - np.min(facade_polygon_np[:, 1])
        
        if facade_width > min_thickness_px and facade_height > min_thickness_px:
            return True, "Wall thickness appears adequate", 1.0
        else:
            return False, "Wall thickness may be insufficient", 0.5
    
    def _validate_opening_sizes(self, layout_params: Dict[str, Any]) -> Tuple[bool, str, float]:
        """Validate that openings have reasonable dimensions."""
        openings = layout_params.get('openings', [])
        scale_factor = layout_params.get('scale_m_per_px', 0.01)
        
        if not openings:
            return True, "No openings to validate", 1.0
        
        violations = 0
        
        for opening in openings:
            bbox = opening.get('bbox_px', [])
            if len(bbox) != 4:
                continue
            
            x_min, y_min, x_max, y_max = bbox
            w_px = x_max - x_min
            h_px = y_max - y_min
            w_m = w_px * scale_factor
            h_m = h_px * scale_factor
            
            opening_type = opening.get('kind', 'unknown')
            
            if opening_type == 'door':
                # Typical door dimensions: 0.8-1.0m wide, 2.0-2.2m high
                if w_m < 0.6 or w_m > 1.2 or h_m < 1.8 or h_m > 2.4:
                    violations += 1
            elif opening_type == 'window':
                # Typical window dimensions: 0.8-2.0m wide, 0.8-1.5m high
                if w_m < 0.6 or w_m > 2.5 or h_m < 0.6 or h_m > 1.8:
                    violations += 1
        
        if violations == 0:
            return True, "All openings have reasonable dimensions", 1.0
        else:
            score = max(0.0, 1.0 - violations / len(openings))
            return False, f"{violations} openings have unreasonable dimensions", score
    
    def _validate_facade_proportions(self, layout_params: Dict[str, Any]) -> Tuple[bool, str, float]:
        """Validate that facade has reasonable proportions."""
        facade_polygon = layout_params.get('facade_polygon', [])
        
        if not facade_polygon or len(facade_polygon) < 3:
            return False, "Invalid facade polygon", 0.0
        
        facade_polygon_np = np.array(facade_polygon)
        
        # Calculate facade dimensions
        min_x = np.min(facade_polygon_np[:, 0])
        max_x = np.max(facade_polygon_np[:, 0])
        min_y = np.min(facade_polygon_np[:, 1])
        max_y = np.max(facade_polygon_np[:, 1])
        
        width = max_x - min_x
        height = max_y - min_y
        
        if width <= 0 or height <= 0:
            return False, "Invalid facade dimensions", 0.0
        
        aspect_ratio = width / height
        
        # Reasonable aspect ratios for facades (0.5 to 3.0)
        if 0.5 <= aspect_ratio <= 3.0:
            return True, "Facade has reasonable proportions", 1.0
        else:
            score = max(0.0, 1.0 - abs(aspect_ratio - 1.5) / 1.5)
            return False, f"Facade aspect ratio {aspect_ratio:.2f} is unusual", score


def validate_layout_constraints(layout_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to validate layout constraints.
    
    Args:
        layout_params: Layout parameters to validate
        
    Returns:
        Validation results
    """
    validator = ConstraintValidator()
    return validator.validate_layout(layout_params)
