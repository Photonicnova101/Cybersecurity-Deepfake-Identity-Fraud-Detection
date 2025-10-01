"""
REST API for deepfake detection system
"""

__version__ = "1.0.0"

from api.routes import router
from api.middleware import setup_middleware
from api.schemas import DetectionRequest, DetectionResponse

__all__ = [
    'router',
    'setup_middleware',
    'DetectionRequest',
    'DetectionResponse',
]
