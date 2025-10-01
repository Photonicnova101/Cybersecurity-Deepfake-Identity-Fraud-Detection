#!/usr/bin/env python3
"""
Command-line interface for deepfake detection
"""

import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm
import logging

from src.detector import DeepfakeDetector
from src.audio.audio_detector import AudioDeepfakeDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_single(args):
    """Detect deepfake in single file"""
    
    # Determine detector type
    if args.type == 'audio':
        detector = AudioDeepfakeDetector()
    else:
        detector = DeepfakeDetector(
            model_path=args.model,
            confidence_threshold=args.threshold
        )
    
    # Process file
    print(f"\n🔍 Analyzing: {args.input}")
    
    if args.type == 'video':
        result = detector.predict_video(args.input, return_detailed=args.detailed)
    elif args.type == 'audio':
        result = detector.predict_audio(args.input)
    else:
        result = detector.predict_image(args.input, return_heatmap=args.heatmap)
    
    # Display results
    print("\n" + "="*60)
    print(f"📊 RESULTS")
    print("="*60)
    print(f"Prediction: {result['prediction'].upper()}")
    print(f"Confidence: {result['confidence']:.2%}")
    
    if 'message' in result:
        print(
