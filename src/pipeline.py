"""
End-to-end pipeline for Sketch2House3D.
Orchestrates the complete process from sketch to 3D model.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import cv2
import os
import yaml
import cv2

from .preprocessing import ImageNormalizer, EdgeDetector, ShapeDetector
from .models import create_segmentation_model
from .reconstruction.layout import FacadeSolver, OpeningSolver, ConstraintValidator, ConstraintRepair
from .reconstruction.geometry import WallsBuilder, create_openings
from .reconstruction.geometry.roof_builder import (
    build_gable_roof, build_hip_roof, build_shed_roof, build_flat_roof
)
from .export import export_to_gltf, export_to_fbx

logger = logging.getLogger(__name__)


class Sketch2HousePipeline:
    """End-to-end pipeline for converting sketches to 3D houses."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or self._default_config()
        self.model = None
        self.preprocessing_modules = {}
        self.layout_modules = {}
        self.geometry_modules = {}
        
    def _default_config(self) -> Dict[str, Any]:
        """Get default pipeline configuration."""
        return {
            'preprocessing': {
                'target_size': (256, 256),
                'enhance_contrast': True,
                'reduce_noise': True,
                'correct_perspective': True
            },
            'segmentation': {
                'architecture': 'deeplabv3plus',
                'encoder': 'mobilenetv3_large_100',
                'num_classes': 4,
                'input_size': [256, 256],
                'pretrained': True,
                'in_channels': 3,
                'weights_path': None
            },
            'layout': {
                'min_wall_thickness': 0.12,
                'door_height_m': 2.0,
                'window_height_ratio': [0.3, 0.8],
                'manhattan_tolerance': 15.0
            },
            'geometry': {
                'wall_thickness': 0.2,
                'wall_height': 3.0,
                'corner_radius': 0.05,
                'min_wall_length': 0.5
            },
            'export': {
                'units': 'meters',
                'right_handed': True,
                'include_materials': True,
                'include_textures': True
            }
        }
    
    def initialize(self):
        """Initialize the pipeline components."""
        try:
            logger.info("Initializing Sketch2House3D pipeline...")
            
            # Initialize preprocessing modules
            self.preprocessing_modules = {
                'image_normalizer': ImageNormalizer(
                    target_size=self.config['preprocessing']['target_size']
                ),
                'edge_detector': EdgeDetector(),
                'shape_detector': ShapeDetector()
            }
            
            # Initialize layout modules
            self.layout_modules = {
                'facade_solver': FacadeSolver(
                    min_wall_thickness=self.config['layout']['min_wall_thickness'],
                    door_height_m=self.config['layout']['door_height_m'],
                    window_height_ratio=tuple(self.config['layout']['window_height_ratio']),
                    manhattan_tolerance=self.config['layout']['manhattan_tolerance']
                ),
                'opening_solver': OpeningSolver(),
                'constraint_validator': ConstraintValidator(),
                'constraint_repair': ConstraintRepair()
            }
            
            # Initialize geometry modules
            self.geometry_modules = {
                'walls_builder': WallsBuilder(
                    wall_thickness=self.config['geometry']['wall_thickness'],
                    wall_height=self.config['geometry']['wall_height'],
                    corner_radius=self.config['geometry']['corner_radius'],
                    min_wall_length=self.config['geometry']['min_wall_length']
                )
            }
            
            # Load segmentation model
            self._load_model()
            
            logger.info("Pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    def process(self, image_path: str, output_dir: str, 
               backend: str = "open3d", export_format: str = "gltf",
               include_textures: bool = True, quality: str = "medium") -> Dict[str, Any]:
        """
        Process a sketch image to generate a 3D house model.
        
        Args:
            image_path: Path to the input sketch image
            output_dir: Directory to save output files
            backend: 3D backend to use
            export_format: Export format for the model
            include_textures: Whether to include textures
            quality: Quality level for generation
            
        Returns:
            Processing results dictionary
        """
        start_time = time.time()
        results = {
            'success': False,
            'input_path': image_path,
            'output_dir': output_dir,
            'processing_time': 0.0,
            'steps': [],
            'errors': [],
            'warnings': [],
            'output_files': []
        }
        
        try:
            logger.info(f"Starting processing of {image_path}")
            
            # Step 1: Preprocessing
            logger.info("Step 1: Preprocessing image...")
            preprocessing_result = self._preprocess_image(image_path)
            results['steps'].append({
                'name': 'preprocessing',
                'status': 'completed',
                'time': time.time() - start_time
            })
            
            # Step 2: Segmentation
            logger.info("Step 2: Running segmentation...")
            segmentation_result = self._run_segmentation(preprocessing_result)
            results['steps'].append({
                'name': 'segmentation',
                'status': 'completed',
                'time': time.time() - start_time
            })
            try:
                out_dir = Path(output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                mask = self._compose_label_mask(segmentation_result)
                cv2.imwrite(str(out_dir / 'mask.png'), mask)
            except Exception as _:
                pass
            
            # Step 3: Layout extraction
            logger.info("Step 3: Extracting layout...")
            layout_result = self._extract_layout(segmentation_result)
            results['steps'].append({
                'name': 'layout_extraction',
                'status': 'completed',
                'time': time.time() - start_time
            })
            
            # Step 4: Constraint validation and repair
            logger.info("Step 4: Validating and repairing constraints...")
            constraint_result = self._validate_and_repair_constraints(layout_result)
            results['steps'].append({
                'name': 'constraint_validation',
                'status': 'completed',
                'time': time.time() - start_time
            })
            
            # Step 5: 3D geometry generation
            logger.info("Step 5: Generating 3D geometry...")
            geometry_result = self._generate_geometry(constraint_result, backend)
            results['steps'].append({
                'name': 'geometry_generation',
                'status': 'completed',
                'time': time.time() - start_time
            })
            
            # Step 6: Export
            logger.info("Step 6: Exporting model...")
            export_result = self._export_model(geometry_result, output_dir, export_format)
            results['steps'].append({
                'name': 'export',
                'status': 'completed',
                'time': time.time() - start_time
            })
            
            # Update results
            results['success'] = True
            results['processing_time'] = time.time() - start_time
            results['output_files'] = export_result.get('output_files', [])
            results['model_url'] = export_result.get('model_url')
            
            logger.info(f"Processing completed successfully in {results['processing_time']:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error in pipeline processing: {e}")
            results['errors'].append(str(e))
            results['processing_time'] = time.time() - start_time
            
        return results
    
    def _preprocess_image(self, image_path: str) -> Dict[str, Any]:
        """Preprocess the input image."""
        try:
            # Load and normalize image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")

            normalizer = self.preprocessing_modules['image_normalizer']
            normalization_result = normalizer.normalize(image)
            
            # Detect edges
            edge_detector = self.preprocessing_modules['edge_detector']
            edge_result = edge_detector.detect_edges(normalization_result['normalized_image'])
            
            # Detect shapes
            shape_detector = self.preprocessing_modules['shape_detector']
            shape_result = shape_detector.detect_architectural_shapes(normalization_result['normalized_image'])
            
            return {
                'normalized_image': normalization_result['normalized_image'],
                'edges': edge_result['edges'],
                'shapes': shape_result['shapes'],
                'metadata': {
                    'original_size': normalization_result['original_size'],
                    'scale_factor': normalization_result['scale_factor']
                }
            }
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            raise
    
    def _run_segmentation(self, preprocessing_result: Dict[str, Any]) -> Dict[str, Any]:
        """Run segmentation on the preprocessed image."""
        try:
            # This is a simplified implementation
            # In practice, you'd load a trained model and run inference
            
            # For now, return mock segmentation results
            image = preprocessing_result['normalized_image']
            height, width = image.shape[:2]
            
            # Create mock segmentation masks
            wall_mask = self._create_mock_mask(height, width, 'wall')
            opening_mask = self._create_mock_mask(height, width, 'opening')
            roof_mask = self._create_mock_mask(height, width, 'roof')
            
            return {
                'wall_mask': wall_mask,
                'opening_mask': opening_mask,
                'roof_mask': roof_mask,
                'metadata': {
                    'image_size': (width, height),
                    'confidence': 0.85
                }
            }
            
        except Exception as e:
            logger.error(f"Error in segmentation: {e}")
            raise
    
    def _extract_layout(self, segmentation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract 2D layout parameters from segmentation."""
        try:
            facade_solver = self.layout_modules['facade_solver']
            opening_solver = self.layout_modules['opening_solver']
            
            # Extract facade layout
            facade_result = facade_solver.solve_facade(
                segmentation_result, 
                segmentation_result['metadata']['image_size']
            )
            
            # Extract openings
            openings_result = opening_solver.solve_openings(
                segmentation_result,
                facade_result['facade_polygon'],
                facade_result['scale_m_per_px']
            )
            
            # Combine results
            layout_params = {
                'scale_m_per_px': facade_result['scale_m_per_px'],
                'facade_polygon': facade_result['facade_polygon'],
                'openings': openings_result['doors'] + openings_result['windows'],
                'roof': {
                    'type': 'gable',
                    'pitch_deg': 30.0,
                    'overhang_m': 0.5
                },
                'metadata': {
                    'image_size': segmentation_result['metadata']['image_size'],
                    'confidence': segmentation_result['metadata']['confidence']
                }
            }
            
            return layout_params
            
        except Exception as e:
            logger.error(f"Error in layout extraction: {e}")
            raise
    
    def _validate_and_repair_constraints(self, layout_params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and repair layout constraints."""
        try:
            constraint_validator = self.layout_modules['constraint_validator']
            constraint_repair = self.layout_modules['constraint_repair']
            
            # Validate constraints
            validation_result = constraint_validator.validate_layout(layout_params)
            
            if not validation_result['valid']:
                logger.warning(f"Layout constraints violated: {validation_result['violations']}")
                
                # Repair constraints
                repaired_layout = constraint_repair.repair_layout(
                    layout_params, 
                    validation_result['violations']
                )
                
                return repaired_layout
            else:
                return layout_params
                
        except Exception as e:
            logger.error(f"Error in constraint validation: {e}")
            raise
    
    def _generate_geometry(self, layout_params: Dict[str, Any], backend: str) -> Dict[str, Any]:
        """Generate 3D geometry from layout parameters."""
        try:
            walls_builder = self.geometry_modules['walls_builder']
            
            # Build walls
            walls_result = walls_builder.build_walls(layout_params)
            
            # Create openings
            openings_result = create_openings(walls_result['wall_mesh'], layout_params)
            
            # Build roof
            roof_type = layout_params['roof']['type']
            if roof_type == 'gable':
                roof_result = build_gable_roof(layout_params)
            elif roof_type == 'hip':
                roof_result = build_hip_roof(layout_params)
            elif roof_type == 'shed':
                roof_result = build_shed_roof(layout_params)
            elif roof_type == 'flat':
                roof_result = build_flat_roof(layout_params)
            else:
                roof_result = build_gable_roof(layout_params)  # Default
            
            return {
                'walls': walls_result,
                'openings': openings_result,
                'roof': roof_result,
                'metadata': {
                    'backend': backend,
                    'total_vertices': 0,  # Calculate from meshes
                    'total_faces': 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error in geometry generation: {e}")
            raise
    
    def _export_model(self, geometry_result: Dict[str, Any], 
                     output_dir: str, export_format: str) -> Dict[str, Any]:
        """Export the 3D model."""
        try:
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Prepare model data for export
            model_data = {
                'walls': {
                    'mesh': geometry_result['walls']['wall_mesh'],
                    'material': {'base_color': [0.8, 0.8, 0.8, 1.0]}
                },
                'roof': {
                    'mesh': geometry_result['roof']['roof_mesh'],
                    'material': {'base_color': [0.6, 0.4, 0.2, 1.0]}
                }
            }
            
            # Export based on format
            if export_format == 'gltf':
                output_file = output_path / "house.gltf"
                export_result = export_to_gltf(model_data, str(output_file))
            elif export_format == 'fbx':
                output_file = output_path / "house.fbx"
                export_result = export_to_fbx(model_data, str(output_file))
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
            
            return {
                'success': export_result.get('success', False),
                'output_files': [str(output_file)],
                'model_url': str(output_file)
            }
            
        except Exception as e:
            logger.error(f"Error in export: {e}")
            raise
    
    def _load_model(self):
        """Load the segmentation model."""
        try:
            # This is a placeholder for model loading
            # In practice, you'd load a trained model
            logger.info("Loading segmentation model...")
            seg_cfg = self.config.get('segmentation', {})
            self.model = create_segmentation_model(seg_cfg)
            weights_path = seg_cfg.get('weights_path')
            if weights_path and __import__('os').path.exists(weights_path):
                import torch
                state = torch.load(weights_path, map_location='cpu')
                if isinstance(state, dict) and 'state_dict' in state:
                    self.model.load_state_dict(state['state_dict'])
                elif isinstance(state, dict):
                    self.model.load_state_dict(state)
                logger.info(f"Loaded weights from {weights_path}")
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _create_mock_mask(self, height: int, width: int, class_name: str):
        """Create a mock segmentation mask for testing."""
        import numpy as np
        
        mask = np.zeros((height, width), dtype=np.uint8)
        
        if class_name == 'wall':
            # Create a rectangular wall mask
            mask[height//4:3*height//4, width//4:3*width//4] = 1
        elif class_name == 'opening':
            # Create opening masks
            mask[height//2-20:height//2+20, width//3-15:width//3+15] = 1  # Door
            mask[height//3-15:height//3+15, 2*width//3-20:2*width//3+20] = 1  # Window
        elif class_name == 'roof':
            # Create triangular roof mask
            for y in range(height//4):
                for x in range(width//2 - y, width//2 + y + 1):
                    if 0 <= x < width:
                        mask[y, x] = 1
        
        return mask


def run_pipeline(image_path: str, output_dir: str, **kwargs) -> Dict[str, Any]:
    """
    Convenience function to run the complete pipeline.
    
    Args:
        image_path: Path to input sketch image
        output_dir: Output directory for results
        **kwargs: Additional pipeline arguments
        
    Returns:
        Processing results
    """
    pipeline = Sketch2HousePipeline()
    pipeline.initialize()
    return pipeline.process(image_path, output_dir, **kwargs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Sketch2House3D Pipeline')
    parser.add_argument('--input', required=True, help='Input sketch image path')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--backend', default='open3d', help='3D backend')
    parser.add_argument('--export', default='gltf', help='Export format')
    parser.add_argument('--config', default=None, help='YAML config path')
    args = parser.parse_args()

    cfg = None
    import os, yaml, json
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
    pipe = Sketch2HousePipeline(cfg)
    pipe.initialize()
    res = pipe.process(args.input, args.output, backend=args.backend, export_format=args.export)
    print(json.dumps(res, indent=2))
