"""
glTF exporter for Sketch2House3D.
Exports 3D models to glTF 2.0 format with proper materials and textures.
"""

import numpy as np
import open3d as o3d
import trimesh
from typing import Dict, Any, List, Tuple, Optional
import logging
import json
import os
from pathlib import Path
import pygltflib

logger = logging.getLogger(__name__)


class GLTFExporter:
    """Exports 3D models to glTF 2.0 format."""
    
    def __init__(self, 
                 units: str = 'meters',
                 right_handed: bool = True,
                 include_materials: bool = True,
                 include_textures: bool = True):
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
    
    def export_model(self, model_data: Dict[str, Any], 
                    output_path: str) -> Dict[str, Any]:
        """
        Export 3D model to glTF format.
        
        Args:
            model_data: Model data containing meshes and materials
            output_path: Output file path
            
        Returns:
            Export results dictionary
        """
        try:
            # Create glTF document
            gltf = pygltflib.GLTF2()
            
            # Set scene
            gltf.scene = 0
            gltf.scenes = [pygltflib.Scene(nodes=[0])]
            
            # Create nodes
            nodes = []
            meshes = []
            materials = []
            textures = []
            images = []
            
            # Process each mesh
            mesh_index = 0
            for mesh_name, mesh_data in model_data.items():
                if isinstance(mesh_data, dict) and 'mesh' in mesh_data:
                    mesh = mesh_data['mesh']
                    material = mesh_data.get('material', {})
                    
                    # Convert mesh to glTF format
                    gltf_mesh = self._convert_mesh_to_gltf(mesh, mesh_name)
                    if gltf_mesh:
                        meshes.append(gltf_mesh)
                        
                        # Create material
                        if self.include_materials:
                            gltf_material = self._create_gltf_material(material, mesh_name)
                            if gltf_material:
                                materials.append(gltf_material)
                        
                        # Create node
                        node = pygltflib.Node(mesh=mesh_index)
                        nodes.append(node)
                        
                        mesh_index += 1
            
            # Set glTF data
            gltf.nodes = nodes
            gltf.meshes = meshes
            if materials:
                gltf.materials = materials
            if textures:
                gltf.textures = textures
            if images:
                gltf.images = images
            
            # Save glTF file
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            gltf.save(output_path)
            
            return {
                'success': True,
                'output_path': str(output_path),
                'mesh_count': len(meshes),
                'material_count': len(materials),
                'texture_count': len(textures),
                'file_size': output_path.stat().st_size if output_path.exists() else 0
            }
            
        except Exception as e:
            logger.error(f"Error exporting glTF: {e}")
            return {
                'success': False,
                'error': str(e),
                'output_path': output_path
            }
    
    def _convert_mesh_to_gltf(self, mesh: o3d.geometry.TriangleMesh, 
                             name: str) -> Optional[pygltflib.Mesh]:
        """Convert Open3D mesh to glTF mesh."""
        try:
            # Get mesh data
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            normals = np.asarray(mesh.vertex_normals) if mesh.has_vertex_normals() else None
            
            # Create accessors
            accessors = []
            buffer_views = []
            
            # Vertex positions
            pos_accessor = pygltflib.Accessor(
                bufferView=0,
                componentType=pygltflib.FLOAT,
                count=len(vertices),
                type=pygltflib.VEC3,
                min=vertices.min(axis=0).tolist(),
                max=vertices.max(axis=0).tolist()
            )
            accessors.append(pos_accessor)
            
            # Vertex normals
            if normals is not None:
                norm_accessor = pygltflib.Accessor(
                    bufferView=1,
                    componentType=pygltflib.FLOAT,
                    count=len(normals),
                    type=pygltflib.VEC3
                )
                accessors.append(norm_accessor)
            
            # Triangle indices
            indices_accessor = pygltflib.Accessor(
                bufferView=2,
                componentType=pygltflib.UNSIGNED_INT,
                count=len(triangles) * 3,
                type=pygltflib.SCALAR
            )
            accessors.append(indices_accessor)
            
            # Create buffer views
            pos_buffer_view = pygltflib.BufferView(
                buffer=0,
                byteOffset=0,
                byteLength=len(vertices) * 3 * 4  # 3 floats * 4 bytes
            )
            buffer_views.append(pos_buffer_view)
            
            if normals is not None:
                norm_buffer_view = pygltflib.BufferView(
                    buffer=0,
                    byteOffset=len(vertices) * 3 * 4,
                    byteLength=len(normals) * 3 * 4
                )
                buffer_views.append(norm_buffer_view)
            
            indices_buffer_view = pygltflib.BufferView(
                buffer=0,
                byteOffset=len(vertices) * 3 * 4 + (len(normals) * 3 * 4 if normals is not None else 0),
                byteLength=len(triangles) * 3 * 4
            )
            buffer_views.append(indices_buffer_view)
            
            # Create buffer
            buffer_data = []
            buffer_data.extend(vertices.flatten())
            if normals is not None:
                buffer_data.extend(normals.flatten())
            buffer_data.extend(triangles.flatten())
            
            buffer = pygltflib.Buffer(
                byteLength=len(buffer_data) * 4,
                uri=f"data:application/octet-stream;base64,{self._encode_base64(buffer_data)}"
            )
            
            # Create primitive
            primitive = pygltflib.Primitive(
                attributes=pygltflib.Attributes(POSITION=0, NORMAL=1 if normals is not None else None),
                indices=2,
                material=0 if self.include_materials else None
            )
            
            # Create mesh
            gltf_mesh = pygltflib.Mesh(
                name=name,
                primitives=[primitive]
            )
            
            return gltf_mesh
            
        except Exception as e:
            logger.error(f"Error converting mesh to glTF: {e}")
            return None
    
    def _create_gltf_material(self, material_data: Dict[str, Any], 
                             name: str) -> Optional[pygltflib.Material]:
        """Create glTF material."""
        try:
            # Default material properties
            base_color = material_data.get('base_color', [0.8, 0.8, 0.8, 1.0])
            metallic = material_data.get('metallic', 0.0)
            roughness = material_data.get('roughness', 0.5)
            
            # Create PBR material
            pbr = pygltflib.PbrMetallicRoughness(
                baseColorFactor=base_color,
                metallicFactor=metallic,
                roughnessFactor=roughness
            )
            
            material = pygltflib.Material(
                name=name,
                pbrMetallicRoughness=pbr
            )
            
            return material
            
        except Exception as e:
            logger.error(f"Error creating glTF material: {e}")
            return None
    
    def _encode_base64(self, data: List[float]) -> str:
        """Encode data as base64."""
        import base64
        import struct
        
        # Convert to bytes
        byte_data = struct.pack('f' * len(data), *data)
        
        # Encode as base64
        return base64.b64encode(byte_data).decode('utf-8')
    
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
        # This is a simplified implementation
        # In practice, you'd need to handle texture mapping, UV coordinates, etc.
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
