"""
Video processing utilities
"""

import cv2
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)


def extract_frames(video_path: str, 
                   interval: int = 30,
                   max_frames: int = None) -> List[np.ndarray]:
    """
    Extract frames from video at specified interval
    
    Args:
        video_path: Path to video file
        interval: Extract every Nth frame
        max_frames: Maximum number of frames to extract
        
    Returns:
        List of frame arrays
    """
    frames = []
    
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return frames
        
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if frame_count % interval == 0:
                frames.append(frame)
                extracted_count += 1
                
                if max_frames and extracted_count >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        logger.info(f"Extracted {len(frames)} frames from {video_path}")
        
    except Exception as e:
        logger.error(f"Error extracting frames: {e}")
    
    return frames


def get_video_info(video_path: str) -> dict:
    """
    Get video metadata
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video information
    """
    try:
        cap = cv2.VideoCapture(video_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            'fps': fps,
            'total_frames': frame_count,
            'width': width,
            'height': height,
            'duration_seconds': duration
        }
        
    except Exception as e:
        logger.error(f"Error getting video info: {e}")
        return {}


def save_video_results(frames: List[np.ndarray],
                      predictions: List[dict],
                      output_path: str,
                      fps: int = 30):
    """
    Save video with detection results overlay
    
    Args:
        frames: List of video frames
        predictions: List of prediction results
        output_path: Path to save output video
        fps: Frames per second
    """
    try:
        if len(frames) == 0:
            return
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame, pred in zip(frames, predictions):
            # Add prediction text overlay
            label = f"{pred['prediction'].upper()}: {pred['confidence']:.2%}"
            color = (0, 0, 255) if pred['prediction'] == 'fake' else (0, 255, 0)
            
            cv2.putText(frame, label, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            out.write(frame)
        
        out.release()
        logger.info(f"Saved annotated video to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving video results: {e}")
