"""
UV unwrapping and texture mapping for 3D models.
Handles planar unwrapping, island packing, and UV optimization.
"""

import numpy as np
import trimesh
from typing import Dict, Any, List, Tuple, Optional
import logging
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class UVUnwrapper:
    """Handles UV unwrapping and optimization for 3D meshes."""
    
    def __init__(self, 
                 texel_density: float = 256.0,
                 padding: float = 0.01,
                 max_island_size: float = 0.8):
        """
        Initialize the UV unwrapper.
        
        Args:
            texel_density: Target texel density in pixels per meter
            padding: Padding between UV islands
            max_island_size: Maximum size for UV islands
        """
        self.texel_density = texel_density
        self.padding = padding
        self.max_island_size = max_island_size
    
    def unwrap_mesh(self, mesh: trimesh.Trimesh, 
                   unwrap_method: str = 'planar') -> Dict[str, Any]:
        """
        Unwrap mesh UV coordinates.
        
        Args:
            mesh: Input mesh
            unwrap_method: Unwrapping method ('planar', 'smart', 'seam_based')
            
        Returns:
            Dictionary containing UV data and metadata
        """
        try:
            if unwrap_method == 'planar':
                uv_data = self._planar_unwrap(mesh)
            elif unwrap_method == 'smart':
                uv_data = self._smart_unwrap(mesh)
            elif unwrap_method == 'seam_based':
                uv_data = self._seam_based_unwrap(mesh)
            else:
                raise ValueError(f"Unknown unwrap method: {unwrap_method}")
            
            # Optimize UV layout
            optimized_uv = self._optimize_uv_layout(uv_data)
            
            return {
                'uv_coordinates': optimized_uv['uv_coordinates'],
                'islands': optimized_uv['islands'],
                'seams': optimized_uv.get('seams', []),
                'metadata': {
                    'method': unwrap_method,
                    'texel_density': self.texel_density,
                    'island_count': len(optimized_uv['islands']),
                    'uv_area': self._calculate_uv_area(optimized_uv['uv_coordinates']),
                    'efficiency': self._calculate_uv_efficiency(optimized_uv)
                }
            }
            
        except Exception as e:
            logger.error(f"Error unwrapping UV: {e}")
            return {
                'uv_coordinates': np.zeros((len(mesh.vertices), 2)),
                'islands': [],
                'seams': [],
                'metadata': {'error': str(e)}
            }
    
    def _planar_unwrap(self, mesh: trimesh.Trimesh) -> Dict[str, Any]:
        """Perform planar UV unwrapping."""
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Calculate face normals
        face_normals = mesh.face_normals
        
        # Group faces by normal direction
        face_groups = self._group_faces_by_normal(face_normals)
        
        uv_coordinates = np.zeros((len(vertices), 2))
        islands = []
        
        current_uv_x = 0.0
        
        for group_id, face_indices in face_groups.items():
            # Calculate UV coordinates for this group
            group_uv = self._calculate_planar_uv(vertices, faces[face_indices])
            
            # Scale and position UV coordinates
            group_uv = self._scale_and_position_uv(group_uv, current_uv_x)
            
            # Update UV coordinates
            for i, face_idx in enumerate(face_indices):
                face = faces[face_idx]
                for j, vertex_idx in enumerate(face):
                    uv_coordinates[vertex_idx] = group_uv[i * 3 + j]
            
            # Create island
            island = {
                'id': group_id,
                'faces': face_indices.tolist(),
                'uv_bounds': self._calculate_uv_bounds(group_uv),
                'area': self._calculate_face_area(vertices, faces[face_indices])
            }
            islands.append(island)
            
            current_uv_x += island['uv_bounds'][2] + self.padding
        
        return {
            'uv_coordinates': uv_coordinates,
            'islands': islands
        }
    
    def _smart_unwrap(self, mesh: trimesh.Trimesh) -> Dict[str, Any]:
        """Perform smart UV unwrapping with seam detection."""
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Detect seams based on angle threshold
        seams = self._detect_seams(mesh, angle_threshold=60.0)
        
        # Create UV coordinates based on seams
        uv_coordinates = self._unwrap_with_seams(vertices, faces, seams)
        
        # Group into islands
        islands = self._create_islands_from_uv(uv_coordinates, faces)
        
        return {
            'uv_coordinates': uv_coordinates,
            'islands': islands,
            'seams': seams
        }
    
    def _seam_based_unwrap(self, mesh: trimesh.Trimesh) -> Dict[str, Any]:
        """Perform seam-based UV unwrapping."""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated seam detection
        return self._planar_unwrap(mesh)
    
    def _group_faces_by_normal(self, face_normals: np.ndarray) -> Dict[int, np.ndarray]:
        """Group faces by similar normal directions."""
        # Use K-means clustering to group faces
        n_clusters = min(6, len(face_normals) // 10)  # Adaptive clustering
        if n_clusters < 2:
            n_clusters = 2
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(face_normals)
        
        groups = {}
        for i, label in enumerate(cluster_labels):
            if label not in groups:
                groups[label] = []
            groups[label].append(i)
        
        return {k: np.array(v) for k, v in groups.items()}
    
    def _calculate_planar_uv(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Calculate planar UV coordinates for faces."""
        uv_coords = []
        
        for face in faces:
            face_vertices = vertices[face]
            
            # Calculate face normal
            v0, v1, v2 = face_vertices
            normal = np.cross(v1 - v0, v2 - v0)
            normal = normal / np.linalg.norm(normal)
            
            # Project vertices to 2D plane
            if abs(normal[2]) > 0.9:  # Face is roughly horizontal
                # Project to XY plane
                uv = face_vertices[:, :2]
            elif abs(normal[1]) > 0.9:  # Face is roughly vertical (side)
                # Project to XZ plane
                uv = face_vertices[:, [0, 2]]
            else:  # Face is roughly vertical (front/back)
                # Project to YZ plane
                uv = face_vertices[:, [1, 2]]
            
            uv_coords.extend(uv)
        
        return np.array(uv_coords)
    
    def _scale_and_position_uv(self, uv_coords: np.ndarray, offset_x: float) -> np.ndarray:
        """Scale and position UV coordinates."""
        if len(uv_coords) == 0:
            return uv_coords
        
        # Normalize to [0, 1] range
        min_uv = np.min(uv_coords, axis=0)
        max_uv = np.max(uv_coords, axis=0)
        size = max_uv - min_uv
        
        if size[0] > 0 and size[1] > 0:
            # Scale to fit in unit square
            scale = min(1.0 / size[0], 1.0 / size[1], self.max_island_size)
            uv_coords = (uv_coords - min_uv) * scale
            
            # Add offset
            uv_coords[:, 0] += offset_x
        
        return uv_coords
    
    def _detect_seams(self, mesh: trimesh.Trimesh, angle_threshold: float) -> List[Tuple[int, int]]:
        """Detect seam edges based on dihedral angle."""
        seams = []
        
        # Get edge information
        edges = mesh.edges
        edge_faces = mesh.edges_face
        
        for i, (edge, faces) in enumerate(zip(edges, edge_faces)):
            if len(faces) == 2:  # Edge shared by two faces
                face1, face2 = faces
                
                # Calculate dihedral angle
                normal1 = mesh.face_normals[face1]
                normal2 = mesh.face_normals[face2]
                
                angle = np.arccos(np.clip(np.dot(normal1, normal2), -1.0, 1.0))
                angle_deg = np.degrees(angle)
                
                if angle_deg > angle_threshold:
                    seams.append((edge[0], edge[1]))
        
        return seams
    
    def _unwrap_with_seams(self, vertices: np.ndarray, faces: np.ndarray, 
                          seams: List[Tuple[int, int]]) -> np.ndarray:
        """Unwrap mesh using detected seams."""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated seam-based unwrapping
        return self._planar_unwrap(trimesh.Trimesh(vertices=vertices, faces=faces))['uv_coordinates']
    
    def _create_islands_from_uv(self, uv_coords: np.ndarray, faces: np.ndarray) -> List[Dict[str, Any]]:
        """Create UV islands from UV coordinates."""
        islands = []
        
        # Simple island detection based on UV connectivity
        # This is a simplified implementation
        island_id = 0
        processed_faces = set()
        
        for face_idx, face in enumerate(faces):
            if face_idx in processed_faces:
                continue
            
            # Create new island
            island_faces = [face_idx]
            processed_faces.add(face_idx)
            
            # Find connected faces
            # This is a simplified connectivity check
            for other_face_idx, other_face in enumerate(faces):
                if other_face_idx in processed_faces:
                    continue
                
                # Check if faces share vertices
                if len(set(face) & set(other_face)) > 0:
                    island_faces.append(other_face_idx)
                    processed_faces.add(other_face_idx)
            
            # Calculate island bounds
            island_uv = []
            for face_idx in island_faces:
                face = faces[face_idx]
                for vertex_idx in face:
                    island_uv.append(uv_coords[vertex_idx])
            
            island_uv = np.array(island_uv)
            bounds = self._calculate_uv_bounds(island_uv)
            
            island = {
                'id': island_id,
                'faces': island_faces,
                'uv_bounds': bounds,
                'area': self._calculate_uv_area(island_uv)
            }
            islands.append(island)
            island_id += 1
        
        return islands
    
    def _optimize_uv_layout(self, uv_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize UV layout by packing islands efficiently."""
        islands = uv_data['islands']
        uv_coords = uv_data['uv_coordinates'].copy()
        
        # Sort islands by area (largest first)
        islands.sort(key=lambda x: x['area'], reverse=True)
        
        # Pack islands using simple bin packing
        packed_islands = self._pack_islands(islands)
        
        # Update UV coordinates based on packed positions
        for island in packed_islands:
            offset_x = island['packed_x']
            offset_y = island['packed_y']
            
            for face_idx in island['faces']:
                face = uv_data.get('faces', [])[face_idx] if 'faces' in uv_data else []
                for vertex_idx in face:
                    uv_coords[vertex_idx, 0] += offset_x
                    uv_coords[vertex_idx, 1] += offset_y
        
        return {
            'uv_coordinates': uv_coords,
            'islands': packed_islands,
            'seams': uv_data.get('seams', [])
        }
    
    def _pack_islands(self, islands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Pack UV islands efficiently."""
        packed_islands = []
        current_x = 0.0
        current_y = 0.0
        max_height = 0.0
        
        for island in islands:
            bounds = island['uv_bounds']
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            
            # Check if island fits on current row
            if current_x + width > 1.0:  # Move to next row
                current_x = 0.0
                current_y += max_height + self.padding
                max_height = 0.0
            
            # Position island
            island['packed_x'] = current_x - bounds[0]
            island['packed_y'] = current_y - bounds[1]
            
            packed_islands.append(island)
            
            # Update position
            current_x += width + self.padding
            max_height = max(max_height, height)
        
        return packed_islands
    
    def _calculate_uv_bounds(self, uv_coords: np.ndarray) -> Tuple[float, float, float, float]:
        """Calculate UV bounds (min_x, min_y, max_x, max_y)."""
        if len(uv_coords) == 0:
            return (0.0, 0.0, 0.0, 0.0)
        
        min_uv = np.min(uv_coords, axis=0)
        max_uv = np.max(uv_coords, axis=0)
        
        return (min_uv[0], min_uv[1], max_uv[0], max_uv[1])
    
    def _calculate_face_area(self, vertices: np.ndarray, faces: np.ndarray) -> float:
        """Calculate total area of faces."""
        total_area = 0.0
        
        for face in faces:
            v0, v1, v2 = vertices[face]
            # Calculate triangle area using cross product
            area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            total_area += area
        
        return total_area
    
    def _calculate_uv_area(self, uv_coords: np.ndarray) -> float:
        """Calculate UV area."""
        if len(uv_coords) < 3:
            return 0.0
        
        # Use shoelace formula for polygon area
        x = uv_coords[:, 0]
        y = uv_coords[:, 1]
        
        area = 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(-1, len(x) - 1)))
        return area
    
    def _calculate_uv_efficiency(self, uv_data: Dict[str, Any]) -> float:
        """Calculate UV layout efficiency."""
        total_uv_area = sum(island['area'] for island in uv_data['islands'])
        
        if total_uv_area == 0:
            return 0.0
        
        # Calculate used UV space
        used_area = 0.0
        for island in uv_data['islands']:
            bounds = island['uv_bounds']
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            used_area += width * height
        
        return total_uv_area / used_area if used_area > 0 else 0.0


def unwrap_mesh_uv(mesh: trimesh.Trimesh, **kwargs) -> Dict[str, Any]:
    """
    Convenience function to unwrap mesh UV coordinates.
    
    Args:
        mesh: Input mesh
        **kwargs: Additional arguments for UVUnwrapper
        
    Returns:
        UV unwrapping results
    """
    unwrapper = UVUnwrapper(**kwargs)
    return unwrapper.unwrap_mesh(mesh)
