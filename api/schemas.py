"""
Pydantic schemas for API requests and responses
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class DetectionRequest(BaseModel):
    """Request schema for detection"""
    return_heatmap: bool = Field(False, description="Generate attention heatmap")
    frame_interval: int = Field(30, description="Analyze every Nth frame (video only)")


class DetectionResponse(BaseModel):
    """Response schema for detection"""
    prediction: str = Field(..., description="Prediction result: 'real', 'fake', or 'error'")
    confidence: float = Field(..., description="Confidence score (0-1)")
    raw_score: float = Field(..., description="Raw model output score")
    face_detected: Optional[bool] = Field(None, description="Whether a face was detected")
    file_type: str = Field(..., description="Type of file: 'image', 'video', or 'audio'")
    filename: str = Field(..., description="Original filename")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    message: Optional[str] = Field(None, description="Additional information or error message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "fake",
                "confidence": 0.87,
                "raw_score": 0.87,
                "face_detected": True,
                "file_type": "image",
                "filename": "example.jpg",
                "metadata": {}
            }
        }


class BatchDetectionResponse(BaseModel):
    """Response schema for batch detection"""
    total_files: int = Field(..., description="Total number of files processed")
    results: List[Dict[str, Any]] = Field(..., description="List of detection results")
    summary: Dict[str, int] = Field(..., description="Summary statistics")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_files": 3,
                "results": [
                    {
                        "filename": "image1.jpg",
                        "prediction": "fake",
                        "confidence": 0.92,
                        "face_detected": True
                    },
                    {
                        "filename": "image2.jpg",
                        "prediction": "real",
                        "confidence": 0.78,
                        "face_detected": True
                    }
                ],
                "summary": {
                    "fake_count": 1,
                    "real_count": 1,
                    "error_count": 0
                }
            }
        }


class ModelInfo(BaseModel):
    """Model information schema"""
    model_path: str = Field(..., description="Path to the model file")
    input_size: List[int] = Field(..., description="Expected input size")
    confidence_threshold: float = Field(..., description="Confidence threshold")
    total_parameters: int = Field(..., description="Total number of model parameters")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_path": "models/efficientnet_b0.h5",
                "input_size": [224, 224],
                "confidence_threshold": 0.7,
                "total_parameters": 4049571
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error details")
    status_code: int = Field(..., description="HTTP status code")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "detail": "Invalid file type",
                "status_code": 400
            }
        }


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "service": "deepfake-detection-api"
            }
        }
