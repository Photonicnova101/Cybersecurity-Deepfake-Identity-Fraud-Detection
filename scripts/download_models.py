"""
Script to download pre-trained models
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model URLs (replace with actual URLs when models are hosted)
MODEL_URLS = {
    'efficientnet_b0.h5': 'https://example.com/models/efficientnet_b0.h5',
    'efficientnet_b4.h5': 'https://example.com/models/efficientnet_b4.h5',
    'xception.h5': 'https://example.com/models/xception.h5',
    'audio_detector.h5': 'https://example.com/models/audio_detector.h5',
}

def download_file(url: str, destination: Path):
    """
    Download file with progress bar
    
    Args:
        url: URL to download from
        destination: Destination path
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f, tqdm(
            desc=destination.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        logger.info(f"Downloaded {destination.name}")
        
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        if destination.exists():
            destination.unlink()
        raise

def main():
    """Download all models"""
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    logger.info("Starting model downloads...")
    
    for model_name, url in MODEL_URLS.items():
        model_path = models_dir / model_name
        
        if model_path.exists():
            logger.info(f"Model {model_name} already exists, skipping")
            continue
        
        logger.info(f"Downloading {model_name}...")
        
        try:
            download_file(url, model_path)
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            logger.info("You can manually download models and place them in the models/ directory")
    
    logger.info("\nModel download complete!")
    logger.info("\nNote: If downloads failed, you need to:")
    logger.info("1. Train your own models using train.py")
    logger.info("2. Or obtain pre-trained models from a trusted source")
    logger.info("3. Place .h5 files in the models/ directory")

if __name__ == '__main__':
    main()
