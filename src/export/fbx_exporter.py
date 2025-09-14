"""
FBX exporter for Sketch2House3D.
Exports 3D models to FBX format for use in various 3D software.
"""

import numpy as np
import open3d as o3d
import trimesh
from typing import Dict, Any, List, Tuple, Optional
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class FBXExporter:
    """Exports 3D models to FBX format."""
    
    def __init__(self, 
                 units: str = 'meters',
                 scale_factor: float = 1.0,
                 include_materials: bool = True):
        """
        Initialize the FBX exporter.
        
        Args:
            units: Units for the exported model
            scale_factor: Scale factor for the model
            include_materials: Whether to include materials
        """
        self.units = units
        self.scale_factor = scale_factor
        self.include_materials = include_materials
    
    def export_model(self, model_data: Dict[str, Any], 
                    output_path: str) -> Dict[str, Any]:
        """
        Export 3D model to FBX format.
        
        Args:
            model_data: Model data containing meshes and materials
            output_path: Output file path
            
        Returns:
            Export results dictionary
        """
        try:
            # Create output directory
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert model data to FBX format
            fbx_data = self._convert_to_fbx(model_data)
            
            # Save FBX file
            with open(output_path, 'w') as f:
                f.write(fbx_data)
            
            return {
                'success': True,
                'output_path': str(output_path),
                'file_size': output_path.stat().st_size if output_path.exists() else 0
            }
            
        except Exception as e:
            logger.error(f"Error exporting FBX: {e}")
            return {
                'success': False,
                'error': str(e),
                'output_path': output_path
            }
    
    def _convert_to_fbx(self, model_data: Dict[str, Any]) -> str:
        """Convert model data to FBX format."""
        # This is a simplified FBX exporter
        # In practice, you'd use a proper FBX library like fbx-sdk or pyfbx
        
        fbx_content = []
        
        # FBX header
        fbx_content.append("; FBX 7.4.0 project file")
        fbx_content.append("; Created by Sketch2House3D")
        fbx_content.append("")
        
        # FBX version
        fbx_content.append("FBXHeaderExtension:  {")
        fbx_content.append("    FBXHeaderVersion: 1003")
        fbx_content.append("    FBXVersion: 7400")
        fbx_content.append("}")
        fbx_content.append("")
        
        # Global settings
        fbx_content.append("GlobalSettings:  {")
        fbx_content.append(f"    Version: 1000")
        fbx_content.append(f"    UpAxis: \"Y\"")
        fbx_content.append(f"    UpAxisSign: 1")
        fbx_content.append(f"    FrontAxis: \"Z\"")
        fbx_content.append(f"    FrontAxisSign: 1")
        fbx_content.append(f"    CoordAxis: \"X\"")
        fbx_content.append(f"    CoordAxisSign: 1")
        fbx_content.append(f"    UnitScaleFactor: {self.scale_factor}")
        fbx_content.append(f"    OriginalUnitScaleFactor: 1")
        fbx_content.append("}")
        fbx_content.append("")
        
        # Process meshes
        for mesh_name, mesh_data in model_data.items():
            if isinstance(mesh_data, dict) and 'mesh' in mesh_data:
                mesh = mesh_data['mesh']
                material = mesh_data.get('material', {})
                
                # Add mesh to FBX
                mesh_fbx = self._mesh_to_fbx(mesh, mesh_name, material)
                fbx_content.extend(mesh_fbx)
        
        # FBX footer
        fbx_content.append("}")
        
        return "\n".join(fbx_content)
    
    def _mesh_to_fbx(self, mesh: o3d.geometry.TriangleMesh, 
                    name: str, material: Dict[str, Any]) -> List[str]:
        """Convert mesh to FBX format."""
        fbx_lines = []
        
        # Get mesh data
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        normals = np.asarray(mesh.vertex_normals) if mesh.has_vertex_normals() else None
        
        # Apply scale factor
        vertices = vertices * self.scale_factor
        
        # Mesh definition
        fbx_lines.append(f"Geometry: {name} {{")
        fbx_lines.append(f"    Version: 124")
        fbx_lines.append(f"    Vertices: *{len(vertices) * 3} {{")
        
        # Add vertices
        for vertex in vertices:
            fbx_lines.append(f"        a: {vertex[0]:.6f},{vertex[1]:.6f},{vertex[2]:.6f}")
        
        fbx_lines.append("    }")
        
        # Add polygon vertex indices
        fbx_lines.append(f"    PolygonVertexIndex: *{len(triangles) * 3} {{")
        for triangle in triangles:
            fbx_lines.append(f"        a: {triangle[0]},{triangle[1]},{triangle[2]}")
        fbx_lines.append("    }")
        
        # Add normals if available
        if normals is not None:
            fbx_lines.append(f"    LayerElementNormal: 0 {{")
            fbx_lines.append(f"        Version: 101")
            fbx_lines.append(f"        Name: \"\"")
            fbx_lines.append(f"        MappingInformationType: \"ByVertice\"")
            fbx_lines.append(f"        ReferenceInformationType: \"Direct\"")
            fbx_lines.append(f"        Normals: *{len(normals) * 3} {{")
            for normal in normals:
                fbx_lines.append(f"            a: {normal[0]:.6f},{normal[1]:.6f},{normal[2]:.6f}")
            fbx_lines.append("        }")
            fbx_lines.append("    }")
        
        # Add material if available
        if self.include_materials and material:
            fbx_lines.append(f"    LayerElementMaterial: 0 {{")
            fbx_lines.append(f"        Version: 101")
            fbx_lines.append(f"        Name: \"\"")
            fbx_lines.append(f"        MappingInformationType: \"AllSame\"")
            fbx_lines.append(f"        ReferenceInformationType: \"IndexToDirect\"")
            fbx_lines.append(f"        Materials: *1 {{")
            fbx_lines.append(f"            a: 0")
            fbx_lines.append(f"        }}")
            fbx_lines.append(f"    }}")
        
        fbx_lines.append("}")
        
        # Add material definition
        if self.include_materials and material:
            material_fbx = self._material_to_fbx(material, f"{name}_material")
            fbx_lines.extend(material_fbx)
        
        return fbx_lines
    
    def _material_to_fbx(self, material: Dict[str, Any], name: str) -> List[str]:
        """Convert material to FBX format."""
        fbx_lines = []
        
        # Material definition
        fbx_lines.append(f"Material: {name} {{")
        fbx_lines.append(f"    Version: 102")
        fbx_lines.append(f"    ShadingModel: \"lambert\"")
        fbx_lines.append(f"    MultiLayer: 0")
        fbx_lines.append(f"    Properties70:  {{")
        
        # Base color
        base_color = material.get('base_color', [0.8, 0.8, 0.8, 1.0])
        fbx_lines.append(f"        P: \"DiffuseColor\", \"Vector3D\", \"Vector\", \"\",{base_color[0]:.6f},{base_color[1]:.6f},{base_color[2]:.6f}")
        
        # Metallic
        metallic = material.get('metallic', 0.0)
        fbx_lines.append(f"        P: \"Metallic\", \"double\", \"Number\", \"\",{metallic:.6f}")
        
        # Roughness
        roughness = material.get('roughness', 0.5)
        fbx_lines.append(f"        P: \"Roughness\", \"double\", \"Number\", \"\",{roughness:.6f}")
        
        fbx_lines.append(f"    }}")
        fbx_lines.append(f"}}")
        
        return fbx_lines


def export_to_fbx(model_data: Dict[str, Any], 
                 output_path: str, **kwargs) -> Dict[str, Any]:
    """
    Convenience function to export model to FBX.
    
    Args:
        model_data: Model data to export
        output_path: Output file path
        **kwargs: Additional arguments for FBXExporter
        
    Returns:
        Export results
    """
    exporter = FBXExporter(**kwargs)
    return exporter.export_model(model_data, output_path)
