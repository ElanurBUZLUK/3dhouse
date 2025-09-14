"""
Pydantic schemas for Sketch2House3D API.
Defines request and response models.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum


class QualityLevel(str, Enum):
    """Quality levels for 3D generation."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class BackendType(str, Enum):
    """3D backend types."""
    OPEN3D = "open3d"
    BLENDER = "blender"


class ExportFormat(str, Enum):
    """Export formats."""
    GLTF = "gltf"
    FBX = "fbx"
    USDZ = "usdz"


class ProcessingStatus(str, Enum):
    """Processing status values."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class InferenceRequest(BaseModel):
    """Request model for 3D house inference."""
    image_path: str = Field(..., description="Path to the sketch image")
    backend: BackendType = Field(default=BackendType.OPEN3D, description="3D backend to use")
    export_format: ExportFormat = Field(default=ExportFormat.GLTF, description="Export format")
    include_textures: bool = Field(default=True, description="Include textures")
    quality: QualityLevel = Field(default=QualityLevel.MEDIUM, description="Quality level")
    request_id: Optional[str] = Field(None, description="Optional request ID")


class InferenceResponse(BaseModel):
    """Response model for 3D house inference."""
    request_id: str = Field(..., description="Unique request identifier")
    success: bool = Field(..., description="Whether the inference was successful")
    model_url: Optional[str] = Field(None, description="URL to download the 3D model")
    report_url: Optional[str] = Field(None, description="URL to download the processing report")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    error: Optional[str] = Field(None, description="Error message if failed")


class BatchInferenceRequest(BaseModel):
    """Request model for batch 3D house inference."""
    image_paths: List[str] = Field(..., description="Paths to the sketch images")
    backend: BackendType = Field(default=BackendType.OPEN3D, description="3D backend to use")
    export_format: ExportFormat = Field(default=ExportFormat.GLTF, description="Export format")
    include_textures: bool = Field(default=True, description="Include textures")
    quality: QualityLevel = Field(default=QualityLevel.MEDIUM, description="Quality level")


class BatchInferenceResponse(BaseModel):
    """Response model for batch 3D house inference."""
    batch_id: str = Field(..., description="Unique batch identifier")
    total_images: int = Field(..., description="Total number of images processed")
    successful: int = Field(..., description="Number of successful inferences")
    failed: int = Field(..., description="Number of failed inferences")
    results: List[Dict[str, Any]] = Field(..., description="Individual results")


class ProcessingStatusResponse(BaseModel):
    """Response model for processing status."""
    request_id: str = Field(..., description="Unique request identifier")
    status: ProcessingStatus = Field(..., description="Current processing status")
    progress: float = Field(..., description="Processing progress (0.0 to 1.0)")
    message: Optional[str] = Field(None, description="Status message")
    estimated_time: Optional[int] = Field(None, description="Estimated time remaining in seconds")
    created_at: str = Field(..., description="Request creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")


class ModelInfo(BaseModel):
    """Model information response."""
    model_name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    architecture: str = Field(..., description="Model architecture")
    input_size: List[int] = Field(..., description="Input image size")
    num_classes: int = Field(..., description="Number of segmentation classes")
    classes: List[str] = Field(..., description="Class names")
    total_parameters: int = Field(..., description="Total number of parameters")
    trainable_parameters: int = Field(..., description="Number of trainable parameters")
    last_updated: str = Field(..., description="Last model update timestamp")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")


class ValidationResult(BaseModel):
    """Sketch validation result."""
    is_valid: bool = Field(..., description="Whether the sketch is valid")
    confidence: float = Field(..., description="Validation confidence score")
    issues: List[str] = Field(default_factory=list, description="List of issues found")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions for improvement")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ExampleSketch(BaseModel):
    """Example sketch information."""
    id: str = Field(..., description="Example ID")
    title: str = Field(..., description="Example title")
    description: str = Field(..., description="Example description")
    image_url: str = Field(..., description="URL to example image")
    model_url: Optional[str] = Field(None, description="URL to generated 3D model")
    difficulty: str = Field(..., description="Difficulty level")
    tags: List[str] = Field(default_factory=list, description="Example tags")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    code: str = Field(..., description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request ID if available")


class ProcessingReport(BaseModel):
    """Processing report model."""
    request_id: str = Field(..., description="Request ID")
    processing_time: float = Field(..., description="Total processing time in seconds")
    steps: List[Dict[str, Any]] = Field(..., description="Processing steps")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Processing metrics")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
    warnings: List[str] = Field(default_factory=list, description="Any warnings")
    output_files: List[str] = Field(..., description="Generated output files")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ConfigurationUpdate(BaseModel):
    """Configuration update request."""
    backend: Optional[BackendType] = Field(None, description="3D backend to use")
    export_format: Optional[ExportFormat] = Field(None, description="Default export format")
    quality: Optional[QualityLevel] = Field(None, description="Default quality level")
    include_textures: Optional[bool] = Field(None, description="Default texture inclusion")
    max_file_size: Optional[int] = Field(None, description="Maximum file size in bytes")
    timeout: Optional[int] = Field(None, description="Processing timeout in seconds")


class ConfigurationResponse(BaseModel):
    """Configuration response."""
    backend: BackendType = Field(..., description="Current 3D backend")
    export_format: ExportFormat = Field(..., description="Current default export format")
    quality: QualityLevel = Field(..., description="Current default quality level")
    include_textures: bool = Field(..., description="Current default texture inclusion")
    max_file_size: int = Field(..., description="Current maximum file size")
    timeout: int = Field(..., description="Current processing timeout")
    last_updated: str = Field(..., description="Last configuration update")
