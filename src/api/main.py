"""
FastAPI main application for Sketch2House3D.
Provides REST API for 3D house generation from child sketches.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any
import json
import tempfile
import shutil

from .routes import router
from .schemas import (
    InferenceRequest, 
    InferenceResponse, 
    ModelInfo, 
    HealthResponse
)
from .services import Sketch2HouseService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Sketch2House3D API",
    description="Convert child sketches to 3D house models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")

# Mount static files
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Initialize service
sketch_service = Sketch2HouseService()

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting Sketch2House3D API...")
    
    # Initialize the service
    await sketch_service.initialize()
    
    # Create necessary directories
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    logger.info("Sketch2House3D API started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down Sketch2House3D API...")
    await sketch_service.cleanup()
    logger.info("Sketch2House3D API shutdown complete")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Sketch2House3D API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check if service is ready
        service_status = await sketch_service.get_status()
        
        return HealthResponse(
            status="healthy",
            service="sketch2house3d",
            version="1.0.0",
            details=service_status
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

@app.get("/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information."""
    try:
        info = await sketch_service.get_model_info()
        return ModelInfo(**info)
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model info")

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
