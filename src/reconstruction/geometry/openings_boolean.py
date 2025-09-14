"""
Openings boolean operations for creating door and window openings in walls.
Handles CSG operations to cut openings from wall geometry.
"""

import numpy as np
import open3d as o3d
import trimesh
from typing import Dict, Any, List, Tuple, Optional
import logging
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class OpeningsBoolean:
    """Handles boolean operations for creating openings in walls."""
    
    def __init__(self, 
                 boolean_tolerance: float = 1e-6,
                 boolean_precision: float = 1e-8,
                 max_boolean_iterations: int = 10):
        """
        Initialize the openings boolean operations.
        
        Args:
            boolean_tolerance: Tolerance for boolean operations
            boolean_precision: Precision for boolean operations
            max_boolean_iterations: Maximum iterations for boolean operations
        """
        self.boolean_tolerance = boolean_tolerance
        self.boolean_precision = boolean_precision
        self.max_boolean_iterations = max_boolean_iterations
    
    def create_openings(self, wall_mesh: o3d.geometry.TriangleMesh, 
                       layout_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create openings in wall mesh.
        
        Args:
            wall_mesh: Wall mesh to cut openings from
            layout_params: Layout parameters containing openings
            
        Returns:
            Dictionary containing modified wall mesh and metadata
        """
        if not wall_mesh or len(wall_mesh.vertices) == 0:
            logger.warning("Invalid wall mesh for opening creation")
            return self._create_empty_result()
        
        openings = layout_params.get('openings', [])
        scale_factor = layout_params.get('scale_m_per_px', 0.01)
        
        if not openings:
            return {
                'wall_mesh': wall_mesh,
                'opening_meshes': [],
                'metadata': {
                    'opening_count': 0,
                    'boolean_operations': 0,
                    'success': True
                }
            }
        
        # Convert to trimesh for better boolean operations
        wall_trimesh = self._o3d_to_trimesh(wall_mesh)
        
        # Create opening meshes
        opening_meshes = []
        boolean_operations = 0
        
        for opening in openings:
            opening_mesh = self._create_opening_mesh(opening, scale_factor)
            if opening_mesh is not None:
                opening_meshes.append(opening_mesh)
        
        if not opening_meshes:
            logger.warning("No valid opening meshes created")
            return {
                'wall_mesh': wall_mesh,
                'opening_meshes': [],
                'metadata': {
                    'opening_count': 0,
                    'boolean_operations': 0,
                    'success': True
                }
            }
        
        # Apply boolean operations
        try:
            modified_wall = wall_trimesh.copy()
            
            for opening_mesh in opening_meshes:
                # Perform boolean difference
                try:
                    modified_wall = modified_wall.difference(opening_mesh)
                    boolean_operations += 1
                except Exception as e:
                    logger.warning(f"Boolean operation failed: {e}")
                    continue
            
            # Convert back to Open3D
            result_mesh = self._trimesh_to_o3d(modified_wall)
            
            return {
                'wall_mesh': result_mesh,
                'opening_meshes': opening_meshes,
                'metadata': {
                    'opening_count': len(opening_meshes),
                    'boolean_operations': boolean_operations,
                    'success': True
                }
            }
            
        except Exception as e:
            logger.error(f"Error in boolean operations: {e}")
            return {
                'wall_mesh': wall_mesh,
                'opening_meshes': opening_meshes,
                'metadata': {
                    'opening_count': len(opening_meshes),
                    'boolean_operations': 0,
                    'success': False,
                    'error': str(e)
                }
            }
    
    def _create_opening_mesh(self, opening: Dict[str, Any], 
                           scale_factor: float) -> Optional[trimesh.Trimesh]:
        """Create 3D mesh for a single opening."""
        bbox_px = opening.get('bbox_px', [])
        opening_type = opening.get('kind', 'unknown')
        
        if len(bbox_px) != 4:
            return None
        
        # Convert to meters
        x_min, y_min, x_max, y_max = bbox_px
        x_min_m = x_min * scale_factor
        y_min_m = y_min * scale_factor
        x_max_m = x_max * scale_factor
        y_max_m = y_max * scale_factor
        
        # Calculate opening dimensions
        width = x_max_m - x_min_m
        height = y_max_m - y_min_m
        
        # Create opening box
        opening_box = self._create_opening_box(
            width, height, opening_type, 
            (x_min_m, y_min_m), scale_factor
        )
        
        return opening_box
    
    def _create_opening_box(self, width: float, height: float, 
                           opening_type: str, position: Tuple[float, float],
                           scale_factor: float) -> Optional[trimesh.Trimesh]:
        """Create a box mesh for opening."""
        x_min, y_min = position
        
        # Calculate depth based on opening type
        if opening_type == 'door':
            depth = 0.1  # 10cm door thickness
            z_min = 0.0  # Door starts at ground level
            z_max = height
        else:  # window
            depth = 0.2  # 20cm window thickness
            z_min = 0.0  # Window starts at ground level
            z_max = height
        
        # Create box vertices
        vertices = np.array([
            [x_min, y_min, z_min],           # 0: bottom-left-back
            [x_min + width, y_min, z_min],   # 1: bottom-right-back
            [x_min + width, y_min + depth, z_min],  # 2: bottom-right-front
            [x_min, y_min + depth, z_min],   # 3: bottom-left-front
            [x_min, y_min, z_max],           # 4: top-left-back
            [x_min + width, y_min, z_max],   # 5: top-right-back
            [x_min + width, y_min + depth, z_max],  # 6: top-right-front
            [x_min, y_min + depth, z_max],   # 7: top-left-front
        ])
        
        # Create box faces
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 7, 6], [4, 6, 5],  # top
            [0, 4, 5], [0, 5, 1],  # back
            [2, 6, 7], [2, 7, 3],  # front
            [0, 3, 7], [0, 7, 4],  # left
            [1, 5, 6], [1, 6, 2],  # right
        ])
        
        try:
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            return mesh
        except Exception as e:
            logger.error(f"Error creating opening box: {e}")
            return None
    
    def _o3d_to_trimesh(self, mesh: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
        """Convert Open3D mesh to trimesh."""
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        
        return trimesh.Trimesh(vertices=vertices, faces=faces)
    
    def _trimesh_to_o3d(self, mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
        """Convert trimesh to Open3D mesh."""
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        o3d_mesh.compute_vertex_normals()
        
        return o3d_mesh
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """Create empty result when opening creation fails."""
        return {
            'wall_mesh': o3d.geometry.TriangleMesh(),
            'opening_meshes': [],
            'metadata': {
                'opening_count': 0,
                'boolean_operations': 0,
                'success': False
            }
        }


class OpeningMeshGenerator:
    """Generates detailed opening meshes for doors and windows."""
    
    def __init__(self, 
                 door_thickness: float = 0.05,
                 window_thickness: float = 0.02,
                 frame_thickness: float = 0.1):
        """
        Initialize opening mesh generator.
        
        Args:
            door_thickness: Door panel thickness
            window_thickness: Window glass thickness
            frame_thickness: Frame thickness
        """
        self.door_thickness = door_thickness
        self.window_thickness = window_thickness
        self.frame_thickness = frame_thickness
    
    def generate_door_mesh(self, opening: Dict[str, Any], 
                          scale_factor: float) -> Optional[trimesh.Trimesh]:
        """Generate detailed door mesh."""
        bbox_px = opening.get('bbox_px', [])
        if len(bbox_px) != 4:
            return None
        
        # Convert to meters
        x_min, y_min, x_max, y_max = bbox_px
        x_min_m = x_min * scale_factor
        y_min_m = y_min * scale_factor
        x_max_m = x_max * scale_factor
        y_max_m = y_max * scale_factor
        
        width = x_max_m - x_min_m
        height = y_max_m - y_min_m
        
        # Create door frame
        frame_mesh = self._create_door_frame(width, height, (x_min_m, y_min_m))
        
        # Create door panel
        panel_mesh = self._create_door_panel(width, height, (x_min_m, y_min_m))
        
        # Combine meshes
        if frame_mesh is not None and panel_mesh is not None:
            return frame_mesh + panel_mesh
        elif frame_mesh is not None:
            return frame_mesh
        elif panel_mesh is not None:
            return panel_mesh
        else:
            return None
    
    def generate_window_mesh(self, opening: Dict[str, Any], 
                            scale_factor: float) -> Optional[trimesh.Trimesh]:
        """Generate detailed window mesh."""
        bbox_px = opening.get('bbox_px', [])
        if len(bbox_px) != 4:
            return None
        
        # Convert to meters
        x_min, y_min, x_max, y_max = bbox_px
        x_min_m = x_min * scale_factor
        y_min_m = y_min * scale_factor
        x_max_m = x_max * scale_factor
        y_max_m = y_max * scale_factor
        
        width = x_max_m - x_min_m
        height = y_max_m - y_min_m
        
        # Create window frame
        frame_mesh = self._create_window_frame(width, height, (x_min_m, y_min_m))
        
        # Create window glass
        glass_mesh = self._create_window_glass(width, height, (x_min_m, y_min_m))
        
        # Combine meshes
        if frame_mesh is not None and glass_mesh is not None:
            return frame_mesh + glass_mesh
        elif frame_mesh is not None:
            return frame_mesh
        elif glass_mesh is not None:
            return glass_mesh
        else:
            return None
    
    def _create_door_frame(self, width: float, height: float, 
                          position: Tuple[float, float]) -> Optional[trimesh.Trimesh]:
        """Create door frame mesh."""
        x_min, y_min = position
        
        # Frame dimensions
        frame_width = self.frame_thickness
        frame_depth = 0.1
        
        # Create frame vertices
        vertices = []
        faces = []
        
        # Top frame
        vertices.extend([
            [x_min, y_min, height - frame_width],
            [x_min + width, y_min, height - frame_width],
            [x_min + width, y_min + frame_depth, height - frame_width],
            [x_min, y_min + frame_depth, height - frame_width],
        ])
        
        # Left frame
        vertices.extend([
            [x_min, y_min, 0],
            [x_min + frame_width, y_min, 0],
            [x_min + frame_width, y_min + frame_depth, 0],
            [x_min, y_min + frame_depth, 0],
        ])
        
        # Right frame
        vertices.extend([
            [x_min + width - frame_width, y_min, 0],
            [x_min + width, y_min, 0],
            [x_min + width, y_min + frame_depth, 0],
            [x_min + width - frame_width, y_min + frame_depth, 0],
        ])
        
        # Create faces (simplified)
        for i in range(0, len(vertices), 4):
            base_idx = i
            faces.extend([
                [base_idx, base_idx + 1, base_idx + 2],
                [base_idx, base_idx + 2, base_idx + 3],
            ])
        
        try:
            return trimesh.Trimesh(vertices=vertices, faces=faces)
        except Exception as e:
            logger.error(f"Error creating door frame: {e}")
            return None
    
    def _create_door_panel(self, width: float, height: float, 
                          position: Tuple[float, float]) -> Optional[trimesh.Trimesh]:
        """Create door panel mesh."""
        x_min, y_min = position
        
        # Panel dimensions
        panel_width = width - 2 * self.frame_thickness
        panel_height = height - self.frame_thickness
        panel_thickness = self.door_thickness
        
        # Create panel vertices
        vertices = np.array([
            [x_min + self.frame_thickness, y_min, 0],
            [x_min + self.frame_thickness + panel_width, y_min, 0],
            [x_min + self.frame_thickness + panel_width, y_min + panel_thickness, 0],
            [x_min + self.frame_thickness, y_min + panel_thickness, 0],
            [x_min + self.frame_thickness, y_min, panel_height],
            [x_min + self.frame_thickness + panel_width, y_min, panel_height],
            [x_min + self.frame_thickness + panel_width, y_min + panel_thickness, panel_height],
            [x_min + self.frame_thickness, y_min + panel_thickness, panel_height],
        ])
        
        # Create panel faces
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 7, 6], [4, 6, 5],  # top
            [0, 4, 5], [0, 5, 1],  # back
            [2, 6, 7], [2, 7, 3],  # front
            [0, 3, 7], [0, 7, 4],  # left
            [1, 5, 6], [1, 6, 2],  # right
        ])
        
        try:
            return trimesh.Trimesh(vertices=vertices, faces=faces)
        except Exception as e:
            logger.error(f"Error creating door panel: {e}")
            return None
    
    def _create_window_frame(self, width: float, height: float, 
                           position: Tuple[float, float]) -> Optional[trimesh.Trimesh]:
        """Create window frame mesh."""
        x_min, y_min = position
        
        # Frame dimensions
        frame_width = self.frame_thickness
        frame_depth = 0.1
        
        # Create frame vertices (simplified)
        vertices = []
        faces = []
        
        # Create a simple rectangular frame
        vertices.extend([
            [x_min, y_min, 0],
            [x_min + width, y_min, 0],
            [x_min + width, y_min + frame_depth, 0],
            [x_min, y_min + frame_depth, 0],
            [x_min, y_min, height],
            [x_min + width, y_min, height],
            [x_min + width, y_min + frame_depth, height],
            [x_min, y_min + frame_depth, height],
        ])
        
        # Create faces
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 7, 6], [4, 6, 5],  # top
            [0, 4, 5], [0, 5, 1],  # back
            [2, 6, 7], [2, 7, 3],  # front
            [0, 3, 7], [0, 7, 4],  # left
            [1, 5, 6], [1, 6, 2],  # right
        ])
        
        try:
            return trimesh.Trimesh(vertices=vertices, faces=faces)
        except Exception as e:
            logger.error(f"Error creating window frame: {e}")
            return None
    
    def _create_window_glass(self, width: float, height: float, 
                           position: Tuple[float, float]) -> Optional[trimesh.Trimesh]:
        """Create window glass mesh."""
        x_min, y_min = position
        
        # Glass dimensions
        glass_width = width - 2 * self.frame_thickness
        glass_height = height - 2 * self.frame_thickness
        glass_thickness = self.window_thickness
        
        # Create glass vertices
        vertices = np.array([
            [x_min + self.frame_thickness, y_min + self.frame_thickness, 0],
            [x_min + self.frame_thickness + glass_width, y_min + self.frame_thickness, 0],
            [x_min + self.frame_thickness + glass_width, y_min + self.frame_thickness + glass_thickness, 0],
            [x_min + self.frame_thickness, y_min + self.frame_thickness + glass_thickness, 0],
            [x_min + self.frame_thickness, y_min + self.frame_thickness, glass_height],
            [x_min + self.frame_thickness + glass_width, y_min + self.frame_thickness, glass_height],
            [x_min + self.frame_thickness + glass_width, y_min + self.frame_thickness + glass_thickness, glass_height],
            [x_min + self.frame_thickness, y_min + self.frame_thickness + glass_thickness, glass_height],
        ])
        
        # Create glass faces
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 7, 6], [4, 6, 5],  # top
            [0, 4, 5], [0, 5, 1],  # back
            [2, 6, 7], [2, 7, 3],  # front
            [0, 3, 7], [0, 7, 4],  # left
            [1, 5, 6], [1, 6, 2],  # right
        ])
        
        try:
            return trimesh.Trimesh(vertices=vertices, faces=faces)
        except Exception as e:
            logger.error(f"Error creating window glass: {e}")
            return None


def create_openings(wall_mesh: o3d.geometry.TriangleMesh, 
                   layout_params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Convenience function to create openings in walls.
    
    Args:
        wall_mesh: Wall mesh to cut openings from
        layout_params: Layout parameters
        **kwargs: Additional arguments for OpeningsBoolean
        
    Returns:
        Opening creation results
    """
    boolean_ops = OpeningsBoolean(**kwargs)
    return boolean_ops.create_openings(wall_mesh, layout_params)
