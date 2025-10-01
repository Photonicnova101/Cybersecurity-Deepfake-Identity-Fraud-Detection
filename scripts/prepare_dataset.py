"""
Prepare dataset for training
Organizes images into train/val/test splits
"""

import argparse
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_dataset(real_dir: str, 
                   fake_dir: str,
                   output_dir: str,
                   train_ratio: float = 0.7,
                   val_ratio: float = 0.15,
                   test_ratio: float = 0.15):
    """
    Prepare and split dataset
    
    Args:
        real_dir: Directory containing real images
        fake_dir: Directory containing fake images
        output_dir: Output directory for organized dataset
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
    """
    # Validate ratios
    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError("Ratios must sum to 1.0")
    
    # Create output directories
    output_path = Path(output_dir)
    splits = ['train', 'validation', 'test']
    classes = ['real', 'fake']
    
    for split in splits:
        for cls in classes:
            (output_path / split / cls).mkdir(parents=True, exist_ok=True)
    
    # Get file lists
    real_files = list(Path(real_dir).glob('**/*'))
    real_files = [f for f in real_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    fake_files = list(Path(fake_dir).glob('**/*'))
    fake_files = [f for f in fake_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    logger.info(f"Found {len(real_files)} real images")
    logger.info(f"Found {len(fake_files)} fake images")
    
    # Split datasets
    def split_files(files, class_name):
        # First split: train and temp (val + test)
        train_files, temp_files = train_test_split(
            files, 
            test_size=(val_ratio + test_ratio),
            random_state=42
        )
        
        # Second split: val and test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_files, test_files = train_test_split(
            temp_files,
            test_size=(1 - val_size),
            random_state=42
        )
        
        return train_files, val_files, test_files
    
    # Split real files
    real_train, real_val, real_test = split_files(real_files, 'real')
    
    # Split fake files
    fake_train, fake_val, fake_test = split_files(fake_files, 'fake')
    
    # Copy files
    def copy_files(files, split, class_name):
        dest_dir = output_path / split / class_name
        for src_file in tqdm(files, desc=f"Copying {split}/{class_name}"):
            dest_file = dest_dir / src_file.name
            shutil.copy2(src_file, dest_file)
    
    logger.info("Copying files...")
    
    copy_files(real_train, 'train', 'real')
    copy_files(real_val, 'validation', 'real')
    copy_files(real_test, 'test', 'real')
    
    copy_files(fake_train, 'train', 'fake')
    copy_files(fake_val, 'validation', 'fake')
    copy_files(fake_test, 'test', 'fake')
    
    # Print statistics
    logger.info("\n" + "="*60)
    logger.info("DATASET STATISTICS")
    logger.info("="*60)
    logger.info(f"Training set:")
    logger.info(f"  Real: {len(real_train)}")
    logger.info(f"  Fake: {len(fake_train)}")
    logger.info(f"  Total: {len(real_train) + len(fake_train)}")
    
    logger.info(f"\nValidation set:")
    logger.info(f"  Real: {len(real_val)}")
    logger.info(f"  Fake: {len(fake_val)}")
    logger.info(f"  Total: {len(real_val) + len(fake_val)}")
    
    logger.info(f"\nTest set:")
    logger.info(f"  Real: {len(real_test)}")
    logger.info(f"  Fake: {len(fake_test)}")
    logger.info(f"  Total: {len(real_test) + len(fake_test)}")
    
    logger.info(f"\n✅ Dataset prepared successfully at: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for training")
    
    parser.add_argument('--real-dir', type=str, required=True,
                       help='Directory containing real images')
    parser.add_argument('--fake-dir', type=str, required=True,
                       help='Directory containing fake images')
    parser.add_argument('--output', type=str, default='data/processed',
                       help='Output directory')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                       help='Test set ratio')
    
    args = parser.parse_args()
    
    # Verify input directories exist
    if not Path(args.real_dir).exists():
        raise ValueError(f"Real directory not found: {args.real_dir}")
    if not Path(args.fake_dir).exists():
        raise ValueError(f"Fake directory not found: {args.fake_dir}")
    
    prepare_dataset(
        args.real_dir,
        args.fake_dir,
        args.output,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio
    )


if __name__ == '__main__':
    main()
