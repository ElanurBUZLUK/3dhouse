"""
ONNX export and optimization for segmentation models.
Provides model conversion, quantization, and TensorRT support.
"""

import torch
import onnx
import onnxruntime as ort
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class ONNXExporter:
    """Handles ONNX export and optimization for segmentation models."""
    
    def __init__(self, 
                 input_size: Tuple[int, int] = (256, 256),
                 batch_size: int = 1,
                 opset_version: int = 11):
        """
        Initialize ONNX exporter.
        
        Args:
            input_size: Input image size (height, width)
            batch_size: Batch size for export
            opset_version: ONNX opset version
        """
        self.input_size = input_size
        self.batch_size = batch_size
        self.opset_version = opset_version
    
    def export_model(self, 
                    model: torch.nn.Module,
                    output_path: str,
                    input_names: Optional[list] = None,
                    output_names: Optional[list] = None,
                    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None) -> Dict[str, Any]:
        """
        Export PyTorch model to ONNX format.
        
        Args:
            model: PyTorch model to export
            output_path: Output path for ONNX model
            input_names: Input tensor names
            output_names: Output tensor names
            dynamic_axes: Dynamic axes configuration
            
        Returns:
            Export results and metadata
        """
        try:
            # Set model to evaluation mode
            model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(self.batch_size, 3, *self.input_size)
            
            # Default names
            if input_names is None:
                input_names = ['input']
            if output_names is None:
                output_names = ['output']
            
            # Default dynamic axes
            if dynamic_axes is None:
                dynamic_axes = {
                    'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                    'output': {0: 'batch_size', 2: 'height', 3: 'width'}
                }
            
            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=self.opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=False
            )
            
            # Verify ONNX model
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            
            # Get model info
            model_size = os.path.getsize(output_path)
            
            return {
                'success': True,
                'output_path': output_path,
                'model_size_mb': model_size / (1024 * 1024),
                'opset_version': self.opset_version,
                'input_size': self.input_size,
                'batch_size': self.batch_size
            }
            
        except Exception as e:
            logger.error(f"Error exporting ONNX model: {e}")
            return {
                'success': False,
                'error': str(e),
                'output_path': output_path
            }
    
    def optimize_model(self, 
                      onnx_path: str,
                      output_path: str,
                      optimization_level: str = 'basic') -> Dict[str, Any]:
        """
        Optimize ONNX model.
        
        Args:
            onnx_path: Path to input ONNX model
            output_path: Path to output optimized model
            optimization_level: Optimization level ('basic', 'extended', 'all')
            
        Returns:
            Optimization results
        """
        try:
            # Load ONNX model
            model = onnx.load(onnx_path)
            
            # Apply optimizations
            if optimization_level == 'basic':
                # Basic optimizations
                from onnx import optimizer
                passes = ['eliminate_identity', 'eliminate_nop_transpose', 'fuse_consecutive_transposes']
                optimized_model = optimizer.optimize(model, passes)
            elif optimization_level == 'extended':
                # Extended optimizations
                from onnx import optimizer
                passes = [
                    'eliminate_identity', 'eliminate_nop_transpose', 'fuse_consecutive_transposes',
                    'fuse_transpose_into_gemm', 'fuse_add_bias_into_conv', 'fuse_pad_into_conv'
                ]
                optimized_model = optimizer.optimize(model, passes)
            elif optimization_level == 'all':
                # All optimizations
                from onnx import optimizer
                passes = optimizer.get_available_passes()
                optimized_model = optimizer.optimize(model, passes)
            else:
                optimized_model = model
            
            # Save optimized model
            onnx.save(optimized_model, output_path)
            
            # Get size comparison
            original_size = os.path.getsize(onnx_path)
            optimized_size = os.path.getsize(output_path)
            compression_ratio = optimized_size / original_size
            
            return {
                'success': True,
                'output_path': output_path,
                'original_size_mb': original_size / (1024 * 1024),
                'optimized_size_mb': optimized_size / (1024 * 1024),
                'compression_ratio': compression_ratio,
                'optimization_level': optimization_level
            }
            
        except Exception as e:
            logger.error(f"Error optimizing ONNX model: {e}")
            return {
                'success': False,
                'error': str(e),
                'output_path': output_path
            }
    
    def quantize_model(self, 
                      onnx_path: str,
                      output_path: str,
                      quantization_type: str = 'int8') -> Dict[str, Any]:
        """
        Quantize ONNX model for faster inference.
        
        Args:
            onnx_path: Path to input ONNX model
            output_path: Path to output quantized model
            quantization_type: Quantization type ('int8', 'uint8', 'fp16')
            
        Returns:
            Quantization results
        """
        try:
            # Load ONNX model
            model = onnx.load(onnx_path)
            
            # Apply quantization
            if quantization_type == 'int8':
                from onnxruntime.quantization import quantize_dynamic
                quantize_dynamic(onnx_path, output_path, weight_type=quantization_type)
            elif quantization_type == 'uint8':
                from onnxruntime.quantization import quantize_dynamic
                quantize_dynamic(onnx_path, output_path, weight_type=quantization_type)
            elif quantization_type == 'fp16':
                from onnx import version_converter
                # Convert to FP16
                model_fp16 = version_converter.convert_version(model, 11)
                onnx.save(model_fp16, output_path)
            else:
                raise ValueError(f"Unsupported quantization type: {quantization_type}")
            
            # Get size comparison
            original_size = os.path.getsize(onnx_path)
            quantized_size = os.path.getsize(output_path)
            compression_ratio = quantized_size / original_size
            
            return {
                'success': True,
                'output_path': output_path,
                'original_size_mb': original_size / (1024 * 1024),
                'quantized_size_mb': quantized_size / (1024 * 1024),
                'compression_ratio': compression_ratio,
                'quantization_type': quantization_type
            }
            
        except Exception as e:
            logger.error(f"Error quantizing ONNX model: {e}")
            return {
                'success': False,
                'error': str(e),
                'output_path': output_path
            }
    
    def test_onnx_model(self, 
                       onnx_path: str,
                       test_input: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Test ONNX model inference.
        
        Args:
            onnx_path: Path to ONNX model
            test_input: Test input array (optional)
            
        Returns:
            Test results and performance metrics
        """
        try:
            # Create test input if not provided
            if test_input is None:
                test_input = np.random.randn(self.batch_size, 3, *self.input_size).astype(np.float32)
            
            # Create ONNX Runtime session
            session = ort.InferenceSession(onnx_path)
            
            # Get input/output names
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            
            # Run inference
            import time
            start_time = time.time()
            
            outputs = session.run([output_name], {input_name: test_input})
            
            inference_time = time.time() - start_time
            
            # Get output shape
            output_shape = outputs[0].shape
            
            return {
                'success': True,
                'inference_time_ms': inference_time * 1000,
                'output_shape': output_shape,
                'input_shape': test_input.shape,
                'model_path': onnx_path
            }
            
        except Exception as e:
            logger.error(f"Error testing ONNX model: {e}")
            return {
                'success': False,
                'error': str(e),
                'model_path': onnx_path
            }


class TensorRTExporter:
    """Handles TensorRT export and optimization."""
    
    def __init__(self, 
                 input_size: Tuple[int, int] = (256, 256),
                 batch_size: int = 1,
                 precision: str = 'fp16'):
        """
        Initialize TensorRT exporter.
        
        Args:
            input_size: Input image size
            batch_size: Batch size
            precision: Precision mode ('fp32', 'fp16', 'int8')
        """
        self.input_size = input_size
        self.batch_size = batch_size
        self.precision = precision
    
    def export_from_onnx(self, 
                        onnx_path: str,
                        output_path: str,
                        max_workspace_size: int = 1 << 30) -> Dict[str, Any]:
        """
        Export ONNX model to TensorRT engine.
        
        Args:
            onnx_path: Path to ONNX model
            output_path: Path to output TensorRT engine
            max_workspace_size: Maximum workspace size in bytes
            
        Returns:
            Export results
        """
        try:
            import tensorrt as trt
            
            # Create TensorRT logger
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
            # Create builder
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            # Parse ONNX model
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    error_msgs = []
                    for error in range(parser.num_errors):
                        error_msgs.append(parser.get_error(error).desc())
                    raise RuntimeError(f"Failed to parse ONNX model: {error_msgs}")
            
            # Configure builder
            builder.max_workspace_size = max_workspace_size
            
            if self.precision == 'fp16':
                builder.fp16_mode = True
            elif self.precision == 'int8':
                builder.int8_mode = True
                # Note: INT8 calibration would be needed for production use
            
            # Build engine
            engine = builder.build_cuda_engine(network)
            
            if engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
            
            # Save engine
            with open(output_path, 'wb') as f:
                f.write(engine.serialize())
            
            # Get engine info
            engine_size = os.path.getsize(output_path)
            
            return {
                'success': True,
                'output_path': output_path,
                'engine_size_mb': engine_size / (1024 * 1024),
                'precision': self.precision,
                'max_workspace_size_mb': max_workspace_size / (1024 * 1024)
            }
            
        except ImportError:
            logger.warning("TensorRT not available. Install tensorrt package for TensorRT support.")
            return {
                'success': False,
                'error': 'TensorRT not available',
                'output_path': output_path
            }
        except Exception as e:
            logger.error(f"Error exporting TensorRT engine: {e}")
            return {
                'success': False,
                'error': str(e),
                'output_path': output_path
            }


def export_model_to_onnx(model: torch.nn.Module, 
                        output_path: str,
                        **kwargs) -> Dict[str, Any]:
    """
    Convenience function to export model to ONNX.
    
    Args:
        model: PyTorch model
        output_path: Output path
        **kwargs: Additional arguments for ONNXExporter
        
    Returns:
        Export results
    """
    exporter = ONNXExporter(**kwargs)
    return exporter.export_model(model, output_path)


def optimize_onnx_model(onnx_path: str, 
                       output_path: str,
                       **kwargs) -> Dict[str, Any]:
    """
    Convenience function to optimize ONNX model.
    
    Args:
        onnx_path: Input ONNX model path
        output_path: Output optimized model path
        **kwargs: Additional arguments for optimization
        
    Returns:
        Optimization results
    """
    exporter = ONNXExporter()
    return exporter.optimize_model(onnx_path, output_path, **kwargs)
