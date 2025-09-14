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
        # Convert 2D segment data to 3D basis vectors
        start2 = np.array(segment['start'], dtype=float)
        end2 = np.array(segment['end'], dtype=float)
        direction2 = np.array(segment['direction'], dtype=float)
        normal2 = np.array(segment['normal'], dtype=float)
        thickness = float(segment['thickness'])
        height = float(segment['height'])

        # Build a rectangular prism along the segment, thickness along normal, height along +Z
        up = np.array([0.0, 0.0, 1.0])
        start = np.array([start2[0], start2[1], 0.0])
        end = np.array([end2[0], end2[1], 0.0])
        normal = np.array([normal2[0], normal2[1], 0.0])
        normal_len = np.linalg.norm(normal[:2])
        if normal_len == 0:
            return None
        normal = normal / normal_len

        # Eight corners of the prism
        s_lb = start + normal * (thickness / 2.0)                   # start left bottom
        s_rb = start - normal * (thickness / 2.0)                   # start right bottom
        s_lt = s_lb + up * height                                   # start left top
        s_rt = s_rb + up * height                                   # start right top

        e_lb = end + normal * (thickness / 2.0)                     # end left bottom
        e_rb = end - normal * (thickness / 2.0)                     # end right bottom
        e_lt = e_lb + up * height                                    # end left top
        e_rt = e_rb + up * height                                    # end right top

        vertices = [s_lb, s_rb, s_lt, s_rt, e_lb, e_rb, e_lt, e_rt]

        # Triangles for 6 faces (two each)
        faces = [
            # bottom (s_lb, s_rb, e_rb, e_lb)
            [0, 1, 5], [0, 5, 4],
            # top (s_lt, e_lt, e_rt, s_rt)
            [2, 6, 7], [2, 7, 3],
            # start face (s_lb, s_lt, s_rt, s_rb)
            [0, 2, 3], [0, 3, 1],
            # end face (e_lb, e_rb, e_rt, e_lt)
            [4, 5, 7], [4, 7, 6],
            # left face (s_lb, e_lb, e_lt, s_lt)
            [0, 4, 6], [0, 6, 2],
            # right face (s_rb, s_rt, e_rt, e_rb)
            [1, 3, 7], [1, 7, 5],
        ]
        
        # Create mesh
        try:
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices))
            mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32))
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
