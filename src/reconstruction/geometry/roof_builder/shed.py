"""
Shed roof builder for creating 3D shed roof geometry.
Handles single-slope roof with one high side and one low side.
"""

import numpy as np
import open3d as o3d
import trimesh
from typing import Dict, Any, List, Tuple, Optional
import logging
import math

logger = logging.getLogger(__name__)


class ShedRoofBuilder:
    """Builds 3D shed roof geometry."""
    
    def __init__(self, 
                 default_pitch: float = 20.0,
                 min_pitch: float = 10.0,
                 max_pitch: float = 45.0,
                 overhang: float = 0.5):
        """
        Initialize the shed roof builder.
        
        Args:
            default_pitch: Default roof pitch in degrees
            min_pitch: Minimum roof pitch in degrees
            max_pitch: Maximum roof pitch in degrees
            overhang: Roof overhang in meters
        """
        self.default_pitch = default_pitch
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.overhang = overhang
    
    def build_shed_roof(self, layout_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build shed roof from layout parameters.
        
        Args:
            layout_params: Layout parameters containing facade and roof info
            
        Returns:
            Dictionary containing roof geometry and metadata
        """
        facade_polygon = layout_params.get('facade_polygon', [])
        roof_params = layout_params.get('roof', {})
        scale_factor = layout_params.get('scale_m_per_px', 0.01)
        
        if not facade_polygon or len(facade_polygon) < 3:
            logger.warning("Invalid facade polygon for roof building")
            return self._create_empty_result()
        
        # Convert facade to meters
        facade_polygon_m = self._convert_to_meters(facade_polygon, scale_factor)
        
        # Extract roof parameters
        pitch_deg = roof_params.get('pitch_deg', self.default_pitch)
        overhang = roof_params.get('overhang_m', self.overhang)
        
        # Clamp pitch to valid range
        pitch_deg = max(self.min_pitch, min(pitch_deg, self.max_pitch))
        
        # Calculate roof geometry
        roof_geometry = self._calculate_roof_geometry(facade_polygon_m, pitch_deg, overhang)
        
        if not roof_geometry:
            logger.warning("Failed to calculate roof geometry")
            return self._create_empty_result()
        
        # Create roof mesh
        roof_mesh = self._create_roof_mesh(roof_geometry)
        
        if roof_mesh is None:
            logger.warning("Failed to create roof mesh")
            return self._create_empty_result()
        
        return {
            'roof_mesh': roof_mesh,
            'roof_geometry': roof_geometry,
            'metadata': {
                'roof_type': 'shed',
                'pitch_deg': pitch_deg,
                'overhang_m': overhang,
                'total_vertices': len(roof_mesh.vertices),
                'total_faces': len(roof_mesh.triangles)
            }
        }
    
    def _convert_to_meters(self, polygon_px: List[List[float]], 
                          scale_factor: float) -> List[List[float]]:
        """Convert polygon from pixels to meters."""
        return [[x * scale_factor, y * scale_factor] for x, y in polygon_px]
    
    def _calculate_roof_geometry(self, facade_polygon: List[List[float]], 
                                pitch_deg: float, overhang: float) -> Optional[Dict[str, Any]]:
        """Calculate shed roof geometry parameters."""
        if len(facade_polygon) < 3:
            return None
        
        polygon_np = np.array(facade_polygon)
        
        # Find facade dimensions
        min_x = np.min(polygon_np[:, 0])
        max_x = np.max(polygon_np[:, 0])
        min_y = np.min(polygon_np[:, 1])
        max_y = np.max(polygon_np[:, 1])
        
        width = max_x - min_x
        height = max_y - min_y
        
        # Calculate roof height
        roof_width = width + 2 * overhang
        roof_height = roof_width * math.tan(math.radians(pitch_deg))
        
        # Calculate shed roof vertices
        roof_vertices = self._calculate_shed_vertices(
            polygon_np, roof_height, overhang
        )
        
        return {
            'roof_height': roof_height,
            'pitch_deg': pitch_deg,
            'overhang': overhang,
            'roof_vertices': roof_vertices,
            'facade_polygon': facade_polygon
        }
    
    def _calculate_shed_vertices(self, facade_polygon: np.ndarray, 
                                roof_height: float, overhang: float) -> List[List[float]]:
        """Calculate shed roof vertices."""
        vertices = []
        
        # Extend polygon with overhang
        extended_polygon = self._extend_polygon_with_overhang(facade_polygon, overhang)
        
        # Find the highest and lowest points
        min_y = np.min(extended_polygon[:, 1])
        max_y = np.max(extended_polygon[:, 1])
        
        # Create high and low eave lines
        high_eave = []
        low_eave = []
        
        for point in extended_polygon:
            # High eave (at max_y)
            high_vertex = [point[0], max_y, roof_height]
            high_eave.append(high_vertex)
            
            # Low eave (at min_y)
            low_vertex = [point[0], min_y, 0]
            low_eave.append(low_vertex)
        
        # Add vertices
        vertices.extend(high_eave)
        vertices.extend(low_eave)
        
        return vertices
    
    def _extend_polygon_with_overhang(self, polygon: np.ndarray, 
                                     overhang: float) -> np.ndarray:
        """Extend polygon with overhang."""
        if len(polygon) < 3:
            return polygon
        
        extended_points = []
        
        for i in range(len(polygon)):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % len(polygon)]
            
            # Calculate outward normal
            direction = p2 - p1
            length = np.linalg.norm(direction)
            if length > 0:
                direction = direction / length
                normal = np.array([-direction[1], direction[0]])  # Perpendicular
                
                # Extend point outward
                extended_point = p1 + normal * overhang
                extended_points.append(extended_point)
        
        return np.array(extended_points)
    
    def _create_roof_mesh(self, roof_geometry: Dict[str, Any]) -> Optional[o3d.geometry.TriangleMesh]:
        """Create 3D mesh for shed roof."""
        vertices = roof_geometry['roof_vertices']
        
        if len(vertices) < 4:
            return None
        
        # Create roof faces
        faces = []
        
        # Split vertices into high and low eave
        num_points = len(vertices) // 2
        high_eave_start = 0
        low_eave_start = num_points
        
        # Create triangular faces between high and low eave
        for i in range(num_points - 1):
            # First triangle
            faces.append([high_eave_start + i, high_eave_start + i + 1, low_eave_start + i])
            # Second triangle
            faces.append([high_eave_start + i + 1, low_eave_start + i + 1, low_eave_start + i])
        
        # Close the roof at the end
        if num_points > 2:
            faces.append([high_eave_start + num_points - 1, high_eave_start, low_eave_start + num_points - 1])
            faces.append([high_eave_start, low_eave_start, low_eave_start + num_points - 1])
        
        # Create mesh
        try:
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.compute_vertex_normals()
            return mesh
        except Exception as e:
            logger.error(f"Error creating roof mesh: {e}")
            return None
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """Create empty result when roof building fails."""
        return {
            'roof_mesh': o3d.geometry.TriangleMesh(),
            'roof_geometry': {},
            'metadata': {
                'roof_type': 'shed',
                'pitch_deg': 0,
                'overhang_m': 0,
                'total_vertices': 0,
                'total_faces': 0
            }
        }


def build_shed_roof(layout_params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Convenience function to build shed roof.
    
    Args:
        layout_params: Layout parameters
        **kwargs: Additional arguments for ShedRoofBuilder
        
    Returns:
        Shed roof building results
    """
    builder = ShedRoofBuilder(**kwargs)
    return builder.build_shed_roof(layout_params)
