"""
PBR material presets for architectural elements.
Provides realistic material definitions for walls, roofs, doors, and windows.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PBRMaterialPresets:
    """PBR material presets for architectural elements."""
    
    def __init__(self):
        """Initialize material presets."""
        self.presets = self._create_material_presets()
    
    def _create_material_presets(self) -> Dict[str, Dict[str, Any]]:
        """Create PBR material presets."""
        return {
            'wall_plaster': {
                'name': 'Wall Plaster',
                'base_color': [0.9, 0.9, 0.85, 1.0],
                'metallic': 0.0,
                'roughness': 0.7,
                'emissive': [0.0, 0.0, 0.0, 1.0],
                'normal_scale': 1.0,
                'occlusion_strength': 1.0,
                'description': 'Standard interior/exterior wall plaster'
            },
            'wall_brick': {
                'name': 'Brick Wall',
                'base_color': [0.7, 0.4, 0.3, 1.0],
                'metallic': 0.0,
                'roughness': 0.8,
                'emissive': [0.0, 0.0, 0.0, 1.0],
                'normal_scale': 1.2,
                'occlusion_strength': 1.0,
                'description': 'Red brick wall material'
            },
            'wall_concrete': {
                'name': 'Concrete Wall',
                'base_color': [0.6, 0.6, 0.6, 1.0],
                'metallic': 0.0,
                'roughness': 0.9,
                'emissive': [0.0, 0.0, 0.0, 1.0],
                'normal_scale': 1.0,
                'occlusion_strength': 1.0,
                'description': 'Concrete wall material'
            },
            'roof_tile': {
                'name': 'Roof Tile',
                'base_color': [0.4, 0.2, 0.1, 1.0],
                'metallic': 0.0,
                'roughness': 0.6,
                'emissive': [0.0, 0.0, 0.0, 1.0],
                'normal_scale': 1.5,
                'occlusion_strength': 1.0,
                'description': 'Terracotta roof tile'
            },
            'roof_metal': {
                'name': 'Metal Roof',
                'base_color': [0.3, 0.3, 0.35, 1.0],
                'metallic': 0.8,
                'roughness': 0.3,
                'emissive': [0.0, 0.0, 0.0, 1.0],
                'normal_scale': 0.8,
                'occlusion_strength': 1.0,
                'description': 'Metal roof material'
            },
            'roof_shingle': {
                'name': 'Roof Shingle',
                'base_color': [0.2, 0.2, 0.2, 1.0],
                'metallic': 0.0,
                'roughness': 0.8,
                'emissive': [0.0, 0.0, 0.0, 1.0],
                'normal_scale': 1.3,
                'occlusion_strength': 1.0,
                'description': 'Asphalt shingle roof'
            },
            'door_wood': {
                'name': 'Wooden Door',
                'base_color': [0.6, 0.4, 0.2, 1.0],
                'metallic': 0.0,
                'roughness': 0.6,
                'emissive': [0.0, 0.0, 0.0, 1.0],
                'normal_scale': 1.2,
                'occlusion_strength': 1.0,
                'description': 'Wooden door material'
            },
            'door_metal': {
                'name': 'Metal Door',
                'base_color': [0.4, 0.4, 0.45, 1.0],
                'metallic': 0.9,
                'roughness': 0.2,
                'emissive': [0.0, 0.0, 0.0, 1.0],
                'normal_scale': 0.5,
                'occlusion_strength': 1.0,
                'description': 'Metal door material'
            },
            'window_glass': {
                'name': 'Window Glass',
                'base_color': [0.8, 0.9, 1.0, 0.3],
                'metallic': 0.0,
                'roughness': 0.0,
                'emissive': [0.0, 0.0, 0.0, 1.0],
                'normal_scale': 0.1,
                'occlusion_strength': 0.5,
                'description': 'Window glass material'
            },
            'window_frame': {
                'name': 'Window Frame',
                'base_color': [0.3, 0.3, 0.3, 1.0],
                'metallic': 0.0,
                'roughness': 0.7,
                'emissive': [0.0, 0.0, 0.0, 1.0],
                'normal_scale': 1.0,
                'occlusion_strength': 1.0,
                'description': 'Window frame material'
            },
            'foundation_concrete': {
                'name': 'Concrete Foundation',
                'base_color': [0.5, 0.5, 0.5, 1.0],
                'metallic': 0.0,
                'roughness': 0.9,
                'emissive': [0.0, 0.0, 0.0, 1.0],
                'normal_scale': 1.0,
                'occlusion_strength': 1.0,
                'description': 'Concrete foundation material'
            },
            'floor_wood': {
                'name': 'Wooden Floor',
                'base_color': [0.7, 0.5, 0.3, 1.0],
                'metallic': 0.0,
                'roughness': 0.5,
                'emissive': [0.0, 0.0, 0.0, 1.0],
                'normal_scale': 1.4,
                'occlusion_strength': 1.0,
                'description': 'Wooden floor material'
            }
        }
    
    def get_material(self, material_type: str) -> Dict[str, Any]:
        """Get material preset by type."""
        if material_type in self.presets:
            return self.presets[material_type].copy()
        else:
            logger.warning(f"Unknown material type: {material_type}")
            return self.presets['wall_plaster'].copy()
    
    def get_material_for_element(self, element_type: str, 
                               style: str = 'default') -> Dict[str, Any]:
        """Get appropriate material for architectural element."""
        material_mapping = {
            'wall': {
                'default': 'wall_plaster',
                'brick': 'wall_brick',
                'concrete': 'wall_concrete'
            },
            'roof': {
                'default': 'roof_tile',
                'tile': 'roof_tile',
                'metal': 'roof_metal',
                'shingle': 'roof_shingle'
            },
            'door': {
                'default': 'door_wood',
                'wood': 'door_wood',
                'metal': 'door_metal'
            },
            'window': {
                'default': 'window_glass',
                'glass': 'window_glass',
                'frame': 'window_frame'
            },
            'foundation': {
                'default': 'foundation_concrete'
            },
            'floor': {
                'default': 'floor_wood',
                'wood': 'floor_wood'
            }
        }
        
        if element_type in material_mapping:
            style_options = material_mapping[element_type]
            if style in style_options:
                material_name = style_options[style]
            else:
                material_name = style_options['default']
        else:
            material_name = 'wall_plaster'  # Default fallback
        
        return self.get_material(material_name)
    
    def create_gltf_material(self, material_preset: Dict[str, Any]) -> Dict[str, Any]:
        """Create glTF material from preset."""
        return {
            'name': material_preset['name'],
            'pbrMetallicRoughness': {
                'baseColorFactor': material_preset['base_color'],
                'metallicFactor': material_preset['metallic'],
                'roughnessFactor': material_preset['roughness']
            },
            'emissiveFactor': material_preset['emissive'],
            'normalTexture': {
                'scale': material_preset['normal_scale']
            },
            'occlusionTexture': {
                'strength': material_preset['occlusion_strength']
            }
        }
    
    def get_all_materials(self) -> List[Dict[str, Any]]:
        """Get all available material presets."""
        return list(self.presets.values())
    
    def get_materials_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get materials by category."""
        category_materials = []
        for material in self.presets.values():
            if material['name'].lower().startswith(category.lower()):
                category_materials.append(material)
        return category_materials


class MaterialAssigner:
    """Assigns appropriate materials to 3D model elements."""
    
    def __init__(self, presets: Optional[PBRMaterialPresets] = None):
        """Initialize material assigner."""
        self.presets = presets or PBRMaterialPresets()
    
    def assign_materials_to_model(self, model_data: Dict[str, Any], 
                                 style_preferences: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Assign materials to model elements.
        
        Args:
            model_data: Model data containing meshes
            style_preferences: Optional style preferences for materials
            
        Returns:
            Model data with assigned materials
        """
        if style_preferences is None:
            style_preferences = {}
        
        updated_model_data = model_data.copy()
        
        # Assign materials to different elements
        for element_name, element_data in model_data.items():
            if isinstance(element_data, dict) and 'mesh' in element_data:
                # Determine element type
                element_type = self._determine_element_type(element_name)
                
                # Get style preference
                style = style_preferences.get(element_type, 'default')
                
                # Get material
                material = self.presets.get_material_for_element(element_type, style)
                
                # Create glTF material
                gltf_material = self.presets.create_gltf_material(material)
                
                # Update element data
                updated_model_data[element_name]['material'] = gltf_material
                updated_model_data[element_name]['material_preset'] = material
        
        return updated_model_data
    
    def _determine_element_type(self, element_name: str) -> str:
        """Determine element type from name."""
        element_name_lower = element_name.lower()
        
        if 'wall' in element_name_lower:
            return 'wall'
        elif 'roof' in element_name_lower:
            return 'roof'
        elif 'door' in element_name_lower:
            return 'door'
        elif 'window' in element_name_lower:
            return 'window'
        elif 'foundation' in element_name_lower:
            return 'foundation'
        elif 'floor' in element_name_lower:
            return 'floor'
        else:
            return 'wall'  # Default fallback
    
    def create_material_variations(self, base_material: str, 
                                  variations: List[str]) -> List[Dict[str, Any]]:
        """Create material variations from base material."""
        base = self.presets.get_material(base_material)
        variations_list = []
        
        for variation in variations:
            if variation == 'darker':
                var_material = base.copy()
                var_material['base_color'] = [c * 0.7 for c in base['base_color'][:3]] + [base['base_color'][3]]
                var_material['name'] = base['name'] + ' (Darker)'
            elif variation == 'lighter':
                var_material = base.copy()
                var_material['base_color'] = [min(1.0, c * 1.3) for c in base['base_color'][:3]] + [base['base_color'][3]]
                var_material['name'] = base['name'] + ' (Lighter)'
            elif variation == 'rougher':
                var_material = base.copy()
                var_material['roughness'] = min(1.0, base['roughness'] * 1.2)
                var_material['name'] = base['name'] + ' (Rougher)'
            elif variation == 'smoother':
                var_material = base.copy()
                var_material['roughness'] = max(0.0, base['roughness'] * 0.8)
                var_material['name'] = base['name'] + ' (Smoother)'
            else:
                continue
            
            variations_list.append(var_material)
        
        return variations_list


def assign_materials_to_model(model_data: Dict[str, Any], 
                            style_preferences: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Convenience function to assign materials to model.
    
    Args:
        model_data: Model data
        style_preferences: Style preferences
        
    Returns:
        Model data with materials
    """
    assigner = MaterialAssigner()
    return assigner.assign_materials_to_model(model_data, style_preferences)
