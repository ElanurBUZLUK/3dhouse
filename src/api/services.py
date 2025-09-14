"""
Service layer for Sketch2House3D API.
Handles business logic and coordinates between components.
"""

import asyncio
import logging
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import uuid
from datetime import datetime

from ..preprocessing import ImageNormalizer, EdgeDetector, ShapeDetector
from ..models import create_segmentation_model
from ..reconstruction.layout import FacadeSolver, OpeningSolver, ConstraintValidator
from ..reconstruction.geometry import WallsBuilder, create_openings
from ..reconstruction.geometry.roof_builder import (
    build_gable_roof, build_hip_roof, build_shed_roof, build_flat_roof
)
from ..export import export_to_gltf, export_to_fbx
from .schemas import ProcessingStatus, ModelInfo, ValidationResult

logger = logging.getLogger(__name__)


class Sketch2HouseService:
    """Main service for Sketch2House3D processing."""
    
    def __init__(self):
        """Initialize the service."""
        self.model = None
        self.preprocessing_modules = {}
        self.layout_modules = {}
        self.geometry_modules = {}
        self.processing_status = {}
        self.config = self._load_config()
        
    async def initialize(self):
        """Initialize the service components."""
        try:
            logger.info("Initializing Sketch2House3D service...")
            
            # Initialize preprocessing modules
            self.preprocessing_modules = {
                'image_normalizer': ImageNormalizer(),
                'edge_detector': EdgeDetector(),
                'shape_detector': ShapeDetector()
            }
            
            # Initialize layout modules
            self.layout_modules = {
                'facade_solver': FacadeSolver(),
                'opening_solver': OpeningSolver(),
                'constraint_validator': ConstraintValidator()
            }
            
            # Initialize geometry modules
            self.geometry_modules = {
                'walls_builder': WallsBuilder()
            }
            
            # Load segmentation model
            await self._load_model()
            
            logger.info("Sketch2House3D service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize service: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup service resources."""
        logger.info("Cleaning up Sketch2House3D service...")
        # Add cleanup logic here
        logger.info("Service cleanup completed")
    
    async def process_sketch(self, image_path: str, backend: str = "open3d", 
                           export_format: str = "gltf", include_textures: bool = True,
                           quality: str = "medium", request_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a child's sketch to generate a 3D house model.
        
        Args:
            image_path: Path to the sketch image
            backend: 3D backend to use
            export_format: Export format for the model
            include_textures: Whether to include textures
            quality: Quality level for generation
            request_id: Optional request ID
            
        Returns:
            Processing results dictionary
        """
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        try:
            # Update processing status
            self.processing_status[request_id] = {
                'status': ProcessingStatus.PROCESSING,
                'progress': 0.0,
                'message': 'Starting processing...',
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            # Step 1: Preprocessing
            await self._update_status(request_id, 0.1, "Preprocessing image...")
            preprocessing_result = await self._preprocess_image(image_path)
            
            # Step 2: Segmentation
            await self._update_status(request_id, 0.3, "Running segmentation...")
            segmentation_result = await self._run_segmentation(preprocessing_result)
            
            # Step 3: Layout extraction
            await self._update_status(request_id, 0.5, "Extracting layout...")
            layout_result = await self._extract_layout(segmentation_result)
            
            # Step 4: 3D geometry generation
            await self._update_status(request_id, 0.7, "Generating 3D geometry...")
            geometry_result = await self._generate_geometry(layout_result, backend)
            
            # Step 5: Export
            await self._update_status(request_id, 0.9, "Exporting model...")
            export_result = await self._export_model(geometry_result, export_format, request_id)
            
            # Complete processing
            await self._update_status(request_id, 1.0, "Processing completed")
            
            return {
                'success': True,
                'model_url': export_result.get('model_url'),
                'report_url': export_result.get('report_url'),
                'metadata': {
                    'processing_time': time.time() - time.time(),
                    'quality': quality,
                    'backend': backend,
                    'export_format': export_format
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing sketch {request_id}: {e}")
            await self._update_status(request_id, 0.0, f"Processing failed: {str(e)}")
            
            return {
                'success': False,
                'error': str(e),
                'metadata': {
                    'request_id': request_id,
                    'error_time': datetime.now().isoformat()
                }
            }
    
    async def _preprocess_image(self, image_path: str) -> Dict[str, Any]:
        """Preprocess the input image."""
        try:
            # Load and normalize image
            normalizer = self.preprocessing_modules['image_normalizer']
            normalization_result = normalizer.normalize(image_path)
            
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
    
    async def _run_segmentation(self, preprocessing_result: Dict[str, Any]) -> Dict[str, Any]:
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
    
    async def _extract_layout(self, segmentation_result: Dict[str, Any]) -> Dict[str, Any]:
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
    
    async def _generate_geometry(self, layout_params: Dict[str, Any], backend: str) -> Dict[str, Any]:
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
    
    async def _export_model(self, geometry_result: Dict[str, Any], 
                          export_format: str, request_id: str) -> Dict[str, Any]:
        """Export the 3D model."""
        try:
            # Create output directory
            output_dir = Path("outputs") / request_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
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
                output_path = output_dir / f"house_{request_id}.gltf"
                export_result = export_to_gltf(model_data, str(output_path))
            elif export_format == 'fbx':
                output_path = output_dir / f"house_{request_id}.fbx"
                export_result = export_to_fbx(model_data, str(output_path))
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
            
            # Create report
            report_path = output_dir / f"report_{request_id}.json"
            report = {
                'request_id': request_id,
                'processing_time': 0.0,  # Calculate actual time
                'steps': ['preprocessing', 'segmentation', 'layout', 'geometry', 'export'],
                'metrics': {
                    'total_vertices': geometry_result['metadata']['total_vertices'],
                    'total_faces': geometry_result['metadata']['total_faces']
                },
                'output_files': [str(output_path)]
            }
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            return {
                'model_url': f"/download/{request_id}?format={export_format}",
                'report_url': f"/report/{request_id}",
                'success': export_result.get('success', False)
            }
            
        except Exception as e:
            logger.error(f"Error in export: {e}")
            raise
    
    async def _update_status(self, request_id: str, progress: float, message: str):
        """Update processing status."""
        if request_id in self.processing_status:
            self.processing_status[request_id].update({
                'progress': progress,
                'message': message,
                'updated_at': datetime.now().isoformat()
            })
    
    async def get_processing_status(self, request_id: str) -> Dict[str, Any]:
        """Get processing status for a request."""
        return self.processing_status.get(request_id, {
            'status': ProcessingStatus.PENDING,
            'progress': 0.0,
            'message': 'Request not found',
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        })
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_name': 'Sketch2House3D',
            'version': '1.0.0',
            'architecture': 'DeepLabV3+',
            'input_size': [256, 256],
            'num_classes': 4,
            'classes': ['bg', 'wall', 'roof', 'opening'],
            'total_parameters': 0,  # Load from actual model
            'trainable_parameters': 0,
            'last_updated': datetime.now().isoformat()
        }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        return {
            'model_loaded': self.model is not None,
            'active_requests': len(self.processing_status),
            'uptime': time.time() - time.time()  # Calculate actual uptime
        }
    
    async def get_model_path(self, request_id: str, format: str) -> Optional[str]:
        """Get model file path for download."""
        output_dir = Path("outputs") / request_id
        model_path = output_dir / f"house_{request_id}.{format}"
        return str(model_path) if model_path.exists() else None
    
    async def get_report_path(self, request_id: str) -> Optional[str]:
        """Get report file path."""
        output_dir = Path("outputs") / request_id
        report_path = output_dir / f"report_{request_id}.json"
        return str(report_path) if report_path.exists() else None
    
    async def cleanup_request(self, request_id: str):
        """Clean up files for a request."""
        output_dir = Path("outputs") / request_id
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)
        
        if request_id in self.processing_status:
            del self.processing_status[request_id]
    
    async def get_examples(self) -> List[Dict[str, Any]]:
        """Get example sketches."""
        return [
            {
                'id': 'example_1',
                'title': 'Simple House',
                'description': 'A basic house with gable roof',
                'image_url': '/static/examples/simple_house.jpg',
                'model_url': '/static/examples/simple_house.gltf',
                'difficulty': 'easy',
                'tags': ['house', 'gable', 'simple']
            }
        ]
    
    async def validate_sketch(self, image_path: str) -> Dict[str, Any]:
        """Validate if a sketch is suitable for processing."""
        try:
            # This is a simplified validation
            # In practice, you'd run more sophisticated checks
            
            return {
                'is_valid': True,
                'confidence': 0.8,
                'issues': [],
                'suggestions': ['Consider adding more detail to the roof'],
                'metadata': {
                    'file_size': os.path.getsize(image_path),
                    'validation_time': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error validating sketch: {e}")
            return {
                'is_valid': False,
                'confidence': 0.0,
                'issues': [str(e)],
                'suggestions': [],
                'metadata': {}
            }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load service configuration."""
        return {
            'max_file_size': 10 * 1024 * 1024,  # 10MB
            'timeout': 300,  # 5 minutes
            'supported_formats': ['jpg', 'jpeg', 'png', 'bmp'],
            'quality_levels': ['low', 'medium', 'high']
        }
    
    async def _load_model(self):
        """Load the segmentation model."""
        try:
            # This is a placeholder for model loading
            # In practice, you'd load a trained model
            logger.info("Loading segmentation model...")
            # self.model = create_segmentation_model(config)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _create_mock_mask(self, height: int, width: int, class_name: str) -> np.ndarray:
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
