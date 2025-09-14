"""
Walls builder for creating 3D wall geometry from 2D layout parameters.
Handles wall extrusion, thickness, and corner connections.
"""

import numpy as np
import open3d as o3d
from typing import Dict, Any, List, Tuple, Optional
import logging
from scipy.spatial.distance import cdist
import trimesh

logger = logging.getLogger(__name__)


class WallsBuilder:
    """Builds 3D wall geometry from 2D layout parameters."""
    
    def __init__(self, 
                 wall_thickness: float = 0.2,
                 wall_height: float = 3.0,
                 corner_radius: float = 0.05,
                 min_wall_length: float = 0.5):
        """
        Initialize the walls builder.
        
        Args:
            wall_thickness: Wall thickness in meters
            wall_height: Wall height in meters
            corner_radius: Radius for corner smoothing
            min_wall_length: Minimum wall segment length
        """
        self.wall_thickness = wall_thickness
        self.wall_height = wall_height
        self.corner_radius = corner_radius
        self.min_wall_length = min_wall_length
    
    def build_walls(self, layout_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build 3D walls from layout parameters.
        
        Args:
            layout_params: Layout parameters containing facade polygon
            
        Returns:
            Dictionary containing wall geometry and metadata
        """
        facade_polygon = layout_params.get('facade_polygon', [])
        scale_factor = layout_params.get('scale_m_per_px', 0.01)
        
        if not facade_polygon or len(facade_polygon) < 3:
            logger.warning("Invalid facade polygon for wall building")
            return self._create_empty_result()
        
        # Convert facade polygon to meters
        facade_polygon_m = self._convert_to_meters(facade_polygon, scale_factor)
        
        # Create wall segments
        wall_segments = self._create_wall_segments(facade_polygon_m)
        
        if not wall_segments:
            logger.warning("No valid wall segments created")
            return self._create_empty_result()
        
        # Build 3D wall geometry
        wall_meshes = []
        for segment in wall_segments:
            wall_mesh = self._create_wall_mesh(segment)
            if wall_mesh is not None:
                wall_meshes.append(wall_mesh)
        
        if not wall_meshes:
            logger.warning("No wall meshes created")
            return self._create_empty_result()
        
        # Combine all wall meshes
        combined_mesh = self._combine_meshes(wall_meshes)
        
        # Add floor and foundation
        floor_mesh = self._create_floor_mesh(facade_polygon_m)
        foundation_mesh = self._create_foundation_mesh(facade_polygon_m)
        
        return {
            'wall_mesh': combined_mesh,
            'floor_mesh': floor_mesh,
            'foundation_mesh': foundation_mesh,
            'wall_segments': wall_segments,
            'metadata': {
                'wall_count': len(wall_segments),
                'wall_thickness': self.wall_thickness,
                'wall_height': self.wall_height,
                'total_vertices': len(combined_mesh.vertices) if combined_mesh else 0,
                'total_faces': len(combined_mesh.triangles) if combined_mesh else 0
            }
        }
    
    def _convert_to_meters(self, polygon_px: List[List[float]], 
                          scale_factor: float) -> List[List[float]]:
        """Convert polygon from pixels to meters."""
        return [[x * scale_factor, y * scale_factor] for x, y in polygon_px]
    
    def _create_wall_segments(self, facade_polygon: List[List[float]]) -> List[Dict[str, Any]]:
        """Create wall segments from facade polygon."""
        if len(facade_polygon) < 3:
            return []
        
        segments = []
        polygon_np = np.array(facade_polygon)
        
        for i in range(len(facade_polygon)):
            p1 = polygon_np[i]
            p2 = polygon_np[(i + 1) % len(facade_polygon)]
            
            # Calculate segment length
            length = np.linalg.norm(p2 - p1)
            
            if length >= self.min_wall_length:
                # Calculate wall direction and normal
                direction = (p2 - p1) / length
                normal = np.array([-direction[1], direction[0]])  # Perpendicular
                
                segment = {
                    'start': p1.tolist(),
                    'end': p2.tolist(),
                    'length': length,
                    'direction': direction.tolist(),
                    'normal': normal.tolist(),
                    'thickness': self.wall_thickness,
                    'height': self.wall_height
                }
                segments.append(segment)
        
        return segments
    
    def _create_wall_mesh(self, segment: Dict[str, Any]) -> Optional[o3d.geometry.TriangleMesh]:
        """Create 3D mesh for a single wall segment."""
        start = np.array(segment['start'])
        end = np.array(segment['end'])
        direction = np.array(segment['direction'])
        normal = np.array(segment['normal'])
        thickness = segment['thickness']
        height = segment['height']
        
        # Create wall vertices
        vertices = []
        
        # Bottom face vertices
        bottom_center = start
        bottom_left = start + normal * (thickness / 2)
        bottom_right = start - normal * (thickness / 2)
        
        # Top face vertices
        top_center = start + np.array([0, 0, height])
        top_left = bottom_left + np.array([0, 0, height])
        top_right = bottom_right + np.array([0, 0, height])
        
        # End vertices
        end_bottom_center = end
        end_bottom_left = end + normal * (thickness / 2)
        end_bottom_right = end - normal * (thickness / 2)
        end_top_center = end + np.array([0, 0, height])
        end_top_left = end_bottom_left + np.array([0, 0, height])
        end_top_right = end_bottom_right + np.array([0, 0, height])
        
        # Add vertices
        vertices.extend([
            bottom_center, bottom_left, bottom_right,
            top_center, top_left, top_right,
            end_bottom_center, end_bottom_left, end_bottom_right,
            end_top_center, end_top_left, end_top_right
        ])
        
        # Create faces
        faces = []
        
        # Bottom face
        faces.extend([
            [0, 1, 2],  # start bottom
            [6, 8, 7],  # end bottom
        ])
        
        # Top face
        faces.extend([
            [3, 5, 4],  # start top
            [9, 10, 11],  # end top
        ])
        
        # Left side
        faces.extend([
            [1, 4, 7], [1, 7, 10],  # left side
        ])
        
        # Right side
        faces.extend([
            [2, 8, 5], [2, 11, 8],  # right side
        ])
        
        # Front and back faces
        faces.extend([
            [0, 2, 5], [0, 5, 3],  # front
            [6, 9, 11], [6, 11, 8],  # back
        ])
        
        # Create mesh
        try:
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.compute_vertex_normals()
            return mesh
        except Exception as e:
            logger.error(f"Error creating wall mesh: {e}")
            return None
    
    def _combine_meshes(self, meshes: List[o3d.geometry.TriangleMesh]) -> o3d.geometry.TriangleMesh:
        """Combine multiple meshes into one."""
        if not meshes:
            return o3d.geometry.TriangleMesh()
        
        if len(meshes) == 1:
            return meshes[0]
        
        # Start with first mesh
        combined = meshes[0]
        
        # Add remaining meshes
        for mesh in meshes[1:]:
            combined += mesh
        
        return combined
    
    def _create_floor_mesh(self, facade_polygon: List[List[float]]) -> Optional[o3d.geometry.TriangleMesh]:
        """Create floor mesh from facade polygon."""
        if len(facade_polygon) < 3:
            return None
        
        # Triangulate the polygon
        polygon_np = np.array(facade_polygon)
        
        # Create floor vertices (add z=0)
        floor_vertices = np.column_stack([polygon_np, np.zeros(len(polygon_np))])
        
        # Simple triangulation (fan triangulation)
        triangles = []
        for i in range(1, len(floor_vertices) - 1):
            triangles.append([0, i, i + 1])
        
        # Create mesh
        try:
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(floor_vertices)
            mesh.triangles = o3d.utility.Vector3iVector(triangles)
            mesh.compute_vertex_normals()
            return mesh
        except Exception as e:
            logger.error(f"Error creating floor mesh: {e}")
            return None
    
    def _create_foundation_mesh(self, facade_polygon: List[List[float]]) -> Optional[o3d.geometry.TriangleMesh]:
        """Create foundation mesh from facade polygon."""
        if len(facade_polygon) < 3:
            return None
        
        foundation_height = 0.3  # 30cm foundation
        foundation_depth = 0.5  # 50cm below ground
        
        # Create foundation vertices
        polygon_np = np.array(facade_polygon)
        
        # Bottom vertices (below ground)
        bottom_vertices = np.column_stack([
            polygon_np, 
            np.full(len(polygon_np), -foundation_depth)
        ])
        
        # Top vertices (at ground level)
        top_vertices = np.column_stack([
            polygon_np, 
            np.zeros(len(polygon_np))
        ])
        
        # Combine vertices
        all_vertices = np.vstack([bottom_vertices, top_vertices])
        
        # Create faces
        faces = []
        
        # Bottom face
        for i in range(1, len(polygon_np) - 1):
            faces.append([0, i, i + 1])
        
        # Top face
        offset = len(polygon_np)
        for i in range(1, len(polygon_np) - 1):
            faces.append([offset, offset + i + 1, offset + i])
        
        # Side faces
        for i in range(len(polygon_np)):
            next_i = (i + 1) % len(polygon_np)
            faces.extend([
                [i, next_i, offset + i],  # side face 1
                [next_i, offset + next_i, offset + i],  # side face 2
            ])
        
        # Create mesh
        try:
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(all_vertices)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.compute_vertex_normals()
            return mesh
        except Exception as e:
            logger.error(f"Error creating foundation mesh: {e}")
            return None
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """Create empty result when wall building fails."""
        return {
            'wall_mesh': o3d.geometry.TriangleMesh(),
            'floor_mesh': o3d.geometry.TriangleMesh(),
            'foundation_mesh': o3d.geometry.TriangleMesh(),
            'wall_segments': [],
            'metadata': {
                'wall_count': 0,
                'wall_thickness': self.wall_thickness,
                'wall_height': self.wall_height,
                'total_vertices': 0,
                'total_faces': 0
            }
        }


def build_walls(layout_params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Convenience function to build walls.
    
    Args:
        layout_params: Layout parameters
        **kwargs: Additional arguments for WallsBuilder
        
    Returns:
        Wall building results
    """
    builder = WallsBuilder(**kwargs)
    return builder.build_walls(layout_params)
