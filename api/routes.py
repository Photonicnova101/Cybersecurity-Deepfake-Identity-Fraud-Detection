"""
API routes for deepfake detection
"""

import os
import uuid
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
import logging
import shutil

from api.schemas import DetectionResponse, BatchDetectionResponse, ModelInfo
from api.middleware import verify_api_key, rate_limit
from src.detector import DeepfakeDetector
from src.audio.audio_detector import AudioDeepfakeDetector

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize detectors
image_detector = DeepfakeDetector()
audio_detector = AudioDeepfakeDetector()

# Upload folder
UPLOAD_FOLDER = Path(os.getenv("UPLOAD_FOLDER", "uploads"))
UPLOAD_FOLDER.mkdir(exist_ok=True)

ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}
ALLOWED_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.ogg', '.flac'}


def validate_file_extension(filename: str, allowed_extensions: set) -> bool:
    """Validate file extension"""
    return Path(filename).suffix.lower() in allowed_extensions


def save_upload_file(upload_file: UploadFile) -> Path:
    """Save uploaded file temporarily"""
    file_id = str(uuid.uuid4())
    file_extension = Path(upload_file.filename).suffix
    file_path = UPLOAD_FOLDER / f"{file_id}{file_extension}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    
    return file_path


def cleanup_file(file_path: Path):
    """Delete temporary file"""
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {e}")


@router.post("/detect/image", response_model=DetectionResponse)
async def detect_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    return_heatmap: bool = False,
    api_key: str = Depends(verify_api_key),
    _rate_limit: None = Depends(rate_limit)
):
    """
    Detect deepfake in image
    
    - **file**: Image file (JPG, PNG)
    - **return_heatmap**: Generate attention heatmap
    """
    # Validate file
    if not validate_file_extension(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {ALLOWED_IMAGE_EXTENSIONS}"
        )
    
    # Save file
    file_path = save_upload_file(file)
    
    # Schedule cleanup
    background_tasks.add_task(cleanup_file, file_path)
    
    try:
        # Detect
        result = image_detector.predict_image(str(file_path), return_heatmap=return_heatmap)
        
        return DetectionResponse(
            prediction=result['prediction'],
            confidence=result['confidence'],
            raw_score=result.get('raw_score', result['confidence']),
            face_detected=result.get('face_detected', True),
            file_type="image",
            filename=file.filename
        )
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect/video", response_model=DetectionResponse)
async def detect_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    frame_interval: int = 30,
    api_key: str = Depends(verify_api_key),
    _rate_limit: None = Depends(rate_limit)
):
    """
    Detect deepfake in video
    
    - **file**: Video file (MP4, AVI, MOV)
    - **frame_interval**: Analyze every Nth frame
    """
    # Validate file
    if not validate_file_extension(file.filename, ALLOWED_VIDEO_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {ALLOWED_VIDEO_EXTENSIONS}"
        )
    
    # Save file
    file_path = save_upload_file(file)
    
    # Schedule cleanup
    background_tasks.add_task(cleanup_file, file_path)
    
    try:
        # Detect
        result = image_detector.predict_video(str(file_path), frame_interval=frame_interval)
        
        return DetectionResponse(
            prediction=result['prediction'],
            confidence=result['confidence'],
            raw_score=result.get('raw_score', result['confidence']),
            file_type="video",
            filename=file.filename,
            metadata={
                'fake_frame_percentage': result.get('fake_frame_percentage'),
                'total_frames_analyzed': result.get('total_frames_analyzed'),
                'fake_frames': result.get('fake_frames'),
                'real_frames': result.get('real_frames')
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect/audio", response_model=DetectionResponse)
async def detect_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key),
    _rate_limit: None = Depends(rate_limit)
):
    """
    Detect deepfake in audio
    
    - **file**: Audio file (WAV, MP3, OGG)
    """
    # Validate file
    if not validate_file_extension(file.filename, ALLOWED_AUDIO_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {ALLOWED_AUDIO_EXTENSIONS}"
        )
    
    # Save file
    file_path = save_upload_file(file)
    
    # Schedule cleanup
    background_tasks.add_task(cleanup_file, file_path)
    
    try:
        # Detect
        result = audio_detector.predict_audio(str(file_path))
        
        return DetectionResponse(
            prediction=result['prediction'],
            confidence=result['confidence'],
            raw_score=result.get('raw_score', result['confidence']),
            file_type="audio",
            filename=file.filename,
            metadata=result.get('voice_features', {})
        )
        
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect/batch", response_model=BatchDetectionResponse)
async def detect_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    api_key: str = Depends(verify_api_key)
):
    """
    Batch detect deepfakes in multiple files
    
    - **files**: Multiple image files
    """
    if len(files) > 50:
        raise HTTPException(
            status_code=400,
            detail="Maximum 50 files per batch request"
        )
    
    results = []
    file_paths = []
    
    try:
        for file in files:
            # Validate and save
            if not validate_file_extension(file.filename, ALLOWED_IMAGE_EXTENSIONS):
                results.append({
                    'filename': file.filename,
                    'error': 'Invalid file type'
                })
                continue
            
            file_path = save_upload_file(file)
            file_paths.append(file_path)
            
            # Detect
            result = image_detector.predict_image(str(file_path))
            
            results.append({
                'filename': file.filename,
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'face_detected': result.get('face_detected', True)
            })
        
        # Schedule cleanup
        for fp in file_paths:
            background_tasks.add_task(cleanup_file, fp)
        
        # Calculate summary
        fake_count = sum(1 for r in results if r.get('prediction') == 'fake')
        real_count = sum(1 for r in results if r.get('prediction') == 'real')
        
        return BatchDetectionResponse(
            total_files=len(files),
            results=results,
            summary={
                'fake_count': fake_count,
                'real_count': real_count,
                'error_count': len(files) - fake_count - real_count
            }
        )
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=ModelInfo)
async def get_models(api_key: str = Depends(verify_api_key)):
    """
    Get information about available models
    """
    info = image_detector.get_model_info()
    
    return ModelInfo(
        model_path=info['model_path'],
        input_size=info['input_size'],
        confidence_threshold=info['confidence_threshold'],
        total_parameters=info['total_parameters']
    )


@router.get("/status")
async def status():
    """
    Get API status
    """
    return {
        "status": "online",
        "detectors": {
            "image": "ready",
            "video": "ready",
            "audio": "ready"
        }
    }
