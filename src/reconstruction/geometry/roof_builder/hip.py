"""
Hip roof builder for creating 3D hip roof geometry.
Handles four-sided roof with sloping ends.
"""

import numpy as np
import open3d as o3d
import trimesh
from typing import Dict, Any, List, Tuple, Optional
import logging
import math

logger = logging.getLogger(__name__)


class HipRoofBuilder:
    """Builds 3D hip roof geometry."""
    
    def __init__(self, 
                 default_pitch: float = 30.0,
                 min_pitch: float = 15.0,
                 max_pitch: float = 60.0,
                 overhang: float = 0.5):
        """
        Initialize the hip roof builder.
        
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
    
    def build_hip_roof(self, layout_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build hip roof from layout parameters.
        
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
                'roof_type': 'hip',
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
        """Calculate hip roof geometry parameters."""
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
        roof_height = (roof_width / 2) * math.tan(math.radians(pitch_deg))
        
        # Calculate hip roof vertices
        roof_vertices = self._calculate_hip_vertices(
            polygon_np, roof_height, overhang
        )
        
        return {
            'roof_height': roof_height,
            'pitch_deg': pitch_deg,
            'overhang': overhang,
            'roof_vertices': roof_vertices,
            'facade_polygon': facade_polygon
        }
    
    def _calculate_hip_vertices(self, facade_polygon: np.ndarray, 
                               roof_height: float, overhang: float) -> List[List[float]]:
        """Calculate hip roof vertices."""
        vertices = []
        
        # Extend polygon with overhang
        extended_polygon = self._extend_polygon_with_overhang(facade_polygon, overhang)
        
        # Find center point
        center_x = np.mean(extended_polygon[:, 0])
        center_y = np.mean(extended_polygon[:, 1])
        center_3d = [center_x, center_y, roof_height]
        
        # Add center vertex
        vertices.append(center_3d)
        
        # Add eave vertices
        for point in extended_polygon:
            eave_vertex = [point[0], point[1], 0]  # Eaves at ground level
            vertices.append(eave_vertex)
        
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
        """Create 3D mesh for hip roof."""
        vertices = roof_geometry['roof_vertices']
        
        if len(vertices) < 4:
            return None
        
        # Create roof faces
        faces = []
        
        # Center vertex is first
        center_idx = 0
        
        # Create triangular faces from center to eaves
        for i in range(1, len(vertices) - 1):
            faces.append([center_idx, i, i + 1])
        
        # Close the roof at the end
        if len(vertices) > 3:
            faces.append([center_idx, len(vertices) - 1, 1])
        
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
                'roof_type': 'hip',
                'pitch_deg': 0,
                'overhang_m': 0,
                'total_vertices': 0,
                'total_faces': 0
            }
        }


def build_hip_roof(layout_params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Convenience function to build hip roof.
    
    Args:
        layout_params: Layout parameters
        **kwargs: Additional arguments for HipRoofBuilder
        
    Returns:
        Hip roof building results
    """
    builder = HipRoofBuilder(**kwargs)
    return builder.build_hip_roof(layout_params)
