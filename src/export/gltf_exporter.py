"""
glTF exporter for Sketch2House3D.
Exports 3D models to glTF/GLB using trimesh for robustness.
"""

import numpy as np
import open3d as o3d
import trimesh
from typing import Dict, Any, List, Tuple, Optional
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class GLTFExporter:
    """Exports 3D models to glTF/GLB using trimesh."""
    
    def __init__(self, 
                 units: str = 'meters',
                 right_handed: bool = True,
                 include_materials: bool = True,
                 include_textures: bool = True,
                 use_draco: bool = True,
                 use_ktx2: bool = True,
                 draco_compression_level: int = 6):
        """
        Initialize the glTF exporter.
        
        Args:
            units: Units for the exported model
            right_handed: Whether to use right-handed coordinate system
            include_materials: Whether to include materials
            include_textures: Whether to include textures
        """
        self.units = units
        self.right_handed = right_handed
        self.include_materials = include_materials
        self.include_textures = include_textures
        self.use_draco = use_draco
        self.use_ktx2 = use_ktx2
        self.draco_compression_level = draco_compression_level
    
    def export_model(self, model_data: Dict[str, Any], output_path: str) -> Dict[str, Any]:
        """Export 3D model to GLB via trimesh, robust single-file output.

        Args:
            model_data: mapping of name -> {'mesh': open3d TriangleMesh, 'material': {...}}
            output_path: target .gltf or .glb path

        Returns:
            dict with success, output_path, file_size
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Aggregate into a scene
            scene = trimesh.Scene()
            mesh_count = 0

            for name, entry in model_data.items():
                if not isinstance(entry, dict) or 'mesh' not in entry:
                    continue
                o3 = entry['mesh']
                if o3 is None or len(o3.vertices) == 0 or len(o3.triangles) == 0:
                    continue
                v = np.asarray(o3.vertices)
                f = np.asarray(o3.triangles)
                tm = trimesh.Trimesh(vertices=v, faces=f, process=False)
                # Optional: simple color
                color = entry.get('material', {}).get('base_color', [0.8, 0.8, 0.8, 1.0])
                rgba = np.clip(np.array(color) * 255.0, 0, 255).astype(np.uint8)
                tm.visual.vertex_colors = np.tile(rgba, (len(v), 1))
                scene.add_geometry(tm, node_name=name)
                mesh_count += 1

            # Choose GLB for single-file robustness
            target = output_path
            if output_path.suffix.lower() == '.gltf':
                target = output_path.with_suffix('.glb')

            # Export with compression options
            export_kwargs = {'file_type': 'glb'}
            
            if self.use_draco:
                export_kwargs['draco_compression'] = True
                export_kwargs['draco_compression_level'] = self.draco_compression_level
            
            if self.use_ktx2:
                export_kwargs['ktx2_compression'] = True
            
            scene.export(target, **export_kwargs)

            return {
                'success': True,
                'output_path': str(target),
                'mesh_count': mesh_count,
                'file_size': target.stat().st_size if target.exists() else 0,
                'output_files': [str(target)],
            }
        except Exception as e:
            logger.error(f"Error exporting GLB: {e}")
            return {'success': False, 'error': str(e), 'output_path': str(output_path)}
    
    # Legacy placeholders kept for API compatibility; now unused.
    
    def _create_gltf_material(self, material_data: Dict[str, Any], name: str):
        return None
    
    def _encode_base64(self, data):
        return ""
    
    def export_with_textures(self, model_data: Dict[str, Any], 
                           output_path: str, 
                           texture_dir: str) -> Dict[str, Any]:
        """
        Export model with textures.
        
        Args:
            model_data: Model data
            output_path: Output file path
            texture_dir: Directory containing textures
            
        Returns:
            Export results
        """
        # Textures not handled in trimesh path; export geometry only.
        return self.export_model(model_data, output_path)


def export_to_gltf(model_data: Dict[str, Any], 
                  output_path: str, **kwargs) -> Dict[str, Any]:
    """
    Convenience function to export model to glTF.
    
    Args:
        model_data: Model data to export
        output_path: Output file path
        **kwargs: Additional arguments for GLTFExporter
        
    Returns:
        Export results
    """
    exporter = GLTFExporter(**kwargs)
    return exporter.export_model(model_data, output_path)
