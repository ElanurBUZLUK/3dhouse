"""
API routes for Sketch2House3D.
Handles HTTP endpoints for the application.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional, Dict, Any
import logging
import os
import tempfile
import uuid
from pathlib import Path
import json

from .schemas import (
    InferenceRequest, 
    InferenceResponse, 
    ProcessingStatus,
    BatchInferenceRequest,
    BatchInferenceResponse
)
from .services import Sketch2HouseService

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Initialize service
sketch_service = Sketch2HouseService()

@router.post("/infer", response_model=InferenceResponse)
async def infer_3d_house(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(..., description="Child's sketch image"),
    backend: str = Form("open3d", description="3D backend to use"),
    export_format: str = Form("gltf", description="Export format"),
    include_textures: bool = Form(True, description="Include textures"),
    quality: str = Form("medium", description="Quality level")
):
    """
    Convert a child's sketch to a 3D house model.
    
    Args:
        image: Uploaded sketch image
        backend: 3D backend to use (open3d, blender)
        export_format: Export format (gltf, fbx, usdz)
        include_textures: Whether to include textures
        quality: Quality level (low, medium, high)
        
    Returns:
        Inference response with 3D model and metadata
    """
    try:
        # Validate file type
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Save uploaded image
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        image_path = upload_dir / f"{request_id}_{image.filename}"
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Process the image
        result = await sketch_service.process_sketch(
            image_path=str(image_path),
            backend=backend,
            export_format=export_format,
            include_textures=include_textures,
            quality=quality,
            request_id=request_id
        )
        
        # Clean up uploaded file
        background_tasks.add_task(cleanup_file, image_path)
        
        return InferenceResponse(
            request_id=request_id,
            success=result['success'],
            model_url=result.get('model_url'),
            report_url=result.get('report_url'),
            metadata=result.get('metadata', {}),
            error=result.get('error')
        )
        
    except Exception as e:
        logger.error(f"Error processing sketch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-infer", response_model=BatchInferenceResponse)
async def batch_infer_3d_houses(
    background_tasks: BackgroundTasks,
    images: list[UploadFile] = File(..., description="Multiple sketch images"),
    backend: str = Form("open3d", description="3D backend to use"),
    export_format: str = Form("gltf", description="Export format"),
    include_textures: bool = Form(True, description="Include textures"),
    quality: str = Form("medium", description="Quality level")
):
    """
    Convert multiple child sketches to 3D house models.
    
    Args:
        images: List of uploaded sketch images
        backend: 3D backend to use
        export_format: Export format
        include_textures: Whether to include textures
        quality: Quality level
        
    Returns:
        Batch inference response with results
    """
    try:
        # Validate file types
        for image in images:
            if not image.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="All files must be images")
        
        # Generate batch ID
        batch_id = str(uuid.uuid4())
        
        # Process images
        results = []
        for i, image in enumerate(images):
            try:
                # Generate unique request ID
                request_id = f"{batch_id}_{i}"
                
                # Save uploaded image
                upload_dir = Path("uploads")
                upload_dir.mkdir(exist_ok=True)
                
                image_path = upload_dir / f"{request_id}_{image.filename}"
                with open(image_path, "wb") as buffer:
                    shutil.copyfileobj(image.file, buffer)
                
                # Process the image
                result = await sketch_service.process_sketch(
                    image_path=str(image_path),
                    backend=backend,
                    export_format=export_format,
                    include_textures=include_textures,
                    quality=quality,
                    request_id=request_id
                )
                
                # Clean up uploaded file
                background_tasks.add_task(cleanup_file, image_path)
                
                results.append({
                    'request_id': request_id,
                    'filename': image.filename,
                    'success': result['success'],
                    'model_url': result.get('model_url'),
                    'report_url': result.get('report_url'),
                    'metadata': result.get('metadata', {}),
                    'error': result.get('error')
                })
                
            except Exception as e:
                logger.error(f"Error processing image {i}: {e}")
                results.append({
                    'request_id': f"{batch_id}_{i}",
                    'filename': image.filename,
                    'success': False,
                    'error': str(e)
                })
        
        return BatchInferenceResponse(
            batch_id=batch_id,
            total_images=len(images),
            successful=sum(1 for r in results if r['success']),
            failed=sum(1 for r in results if not r['success']),
            results=results
        )
        
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{request_id}", response_model=ProcessingStatus)
async def get_processing_status(request_id: str):
    """
    Get the processing status of a request.
    
    Args:
        request_id: Unique request identifier
        
    Returns:
        Processing status
    """
    try:
        status = await sketch_service.get_processing_status(request_id)
        return ProcessingStatus(**status)
    except Exception as e:
        logger.error(f"Error getting status for {request_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download/{request_id}")
async def download_model(request_id: str, format: str = "gltf"):
    """
    Download the generated 3D model.
    
    Args:
        request_id: Unique request identifier
        format: Model format (gltf, fbx, usdz)
        
    Returns:
        Model file
    """
    try:
        model_path = await sketch_service.get_model_path(request_id, format)
        
        if not model_path or not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model not found")
        
        return FileResponse(
            path=model_path,
            filename=f"house_{request_id}.{format}",
            media_type="application/octet-stream"
        )
        
    except Exception as e:
        logger.error(f"Error downloading model {request_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/report/{request_id}")
async def get_processing_report(request_id: str):
    """
    Get the processing report for a request.
    
    Args:
        request_id: Unique request identifier
        
    Returns:
        Processing report
    """
    try:
        report_path = await sketch_service.get_report_path(request_id)
        
        if not report_path or not os.path.exists(report_path):
            raise HTTPException(status_code=404, detail="Report not found")
        
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        return JSONResponse(content=report)
        
    except Exception as e:
        logger.error(f"Error getting report for {request_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/cleanup/{request_id}")
async def cleanup_request(request_id: str):
    """
    Clean up files for a specific request.
    
    Args:
        request_id: Unique request identifier
        
    Returns:
        Cleanup status
    """
    try:
        await sketch_service.cleanup_request(request_id)
        return {"message": "Cleanup completed", "request_id": request_id}
        
    except Exception as e:
        logger.error(f"Error cleaning up {request_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/examples")
async def get_examples():
    """
    Get example sketches and their results.
    
    Returns:
        List of example sketches
    """
    try:
        examples = await sketch_service.get_examples()
        return examples
        
    except Exception as e:
        logger.error(f"Error getting examples: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/validate")
async def validate_sketch(
    image: UploadFile = File(..., description="Sketch image to validate")
):
    """
    Validate if a sketch is suitable for 3D generation.
    
    Args:
        image: Uploaded sketch image
        
    Returns:
        Validation results
    """
    try:
        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            shutil.copyfileobj(image.file, tmp_file)
            tmp_path = tmp_file.name
        
        # Validate the sketch
        validation_result = await sketch_service.validate_sketch(tmp_path)
        
        # Clean up
        os.unlink(tmp_path)
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Error validating sketch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def cleanup_file(file_path: Path):
    """Clean up a file asynchronously."""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        logger.error(f"Error cleaning up file {file_path}: {e}")
