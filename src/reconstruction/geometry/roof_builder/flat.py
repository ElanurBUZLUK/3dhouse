"""
Flat roof builder for creating 3D flat roof geometry.
Handles flat roof with optional parapet walls.
"""

import numpy as np
import open3d as o3d
import trimesh
from typing import Dict, Any, List, Tuple, Optional
import logging
import math

logger = logging.getLogger(__name__)


class FlatRoofBuilder:
    """Builds 3D flat roof geometry."""
    
    def __init__(self, 
                 roof_height: float = 0.1,
                 parapet_height: float = 0.5,
                 overhang: float = 0.5):
        """
        Initialize the flat roof builder.
        
        Args:
            roof_height: Roof thickness in meters
            parapet_height: Parapet wall height in meters
            overhang: Roof overhang in meters
        """
        self.roof_height = roof_height
        self.parapet_height = parapet_height
        self.overhang = overhang
    
    def build_flat_roof(self, layout_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build flat roof from layout parameters.
        
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
        overhang = roof_params.get('overhang_m', self.overhang)
        parapet_height = roof_params.get('parapet_height', self.parapet_height)
        
        # Calculate roof geometry
        roof_geometry = self._calculate_roof_geometry(facade_polygon_m, overhang, parapet_height)
        
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
                'roof_type': 'flat',
                'roof_height': self.roof_height,
                'parapet_height': parapet_height,
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
                                overhang: float, parapet_height: float) -> Optional[Dict[str, Any]]:
        """Calculate flat roof geometry parameters."""
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
        
        # Calculate roof vertices
        roof_vertices = self._calculate_flat_vertices(
            polygon_np, overhang, parapet_height
        )
        
        return {
            'roof_height': self.roof_height,
            'parapet_height': parapet_height,
            'overhang': overhang,
            'roof_vertices': roof_vertices,
            'facade_polygon': facade_polygon
        }
    
    def _calculate_flat_vertices(self, facade_polygon: np.ndarray, 
                                overhang: float, parapet_height: float) -> List[List[float]]:
        """Calculate flat roof vertices."""
        vertices = []
        
        # Extend polygon with overhang
        extended_polygon = self._extend_polygon_with_overhang(facade_polygon, overhang)
        
        # Create roof surface vertices
        for point in extended_polygon:
            roof_vertex = [point[0], point[1], self.roof_height]
            vertices.append(roof_vertex)
        
        # Create parapet vertices
        for point in extended_polygon:
            parapet_vertex = [point[0], point[1], self.roof_height + parapet_height]
            vertices.append(parapet_vertex)
        
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
        """Create 3D mesh for flat roof."""
        vertices = roof_geometry['roof_vertices']
        
        if len(vertices) < 4:
            return None
        
        # Create roof faces
        faces = []
        
        # Split vertices into roof surface and parapet
        num_points = len(vertices) // 2
        roof_surface_start = 0
        parapet_start = num_points
        
        # Create roof surface faces
        for i in range(num_points - 1):
            faces.append([roof_surface_start + i, roof_surface_start + i + 1, roof_surface_start])
        
        # Close the roof surface
        if num_points > 2:
            faces.append([roof_surface_start + num_points - 1, roof_surface_start, roof_surface_start + 1])
        
        # Create parapet faces
        for i in range(num_points - 1):
            # Parapet wall faces
            faces.append([roof_surface_start + i, parapet_start + i, parapet_start + i + 1])
            faces.append([roof_surface_start + i, parapet_start + i + 1, roof_surface_start + i + 1])
        
        # Close the parapet
        if num_points > 2:
            faces.append([roof_surface_start + num_points - 1, parapet_start + num_points - 1, parapet_start])
            faces.append([roof_surface_start + num_points - 1, parapet_start, roof_surface_start])
        
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
                'roof_type': 'flat',
                'roof_height': 0,
                'parapet_height': 0,
                'overhang_m': 0,
                'total_vertices': 0,
                'total_faces': 0
            }
        }


def build_flat_roof(layout_params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Convenience function to build flat roof.
    
    Args:
        layout_params: Layout parameters
        **kwargs: Additional arguments for FlatRoofBuilder
        
    Returns:
        Flat roof building results
    """
    builder = FlatRoofBuilder(**kwargs)
    return builder.build_flat_roof(layout_params)
