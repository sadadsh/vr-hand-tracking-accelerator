

"""
Dataset Downloader for Realtime VR Hand Tracking
Downloads the minimal dataset (~50GB) from Hugging Face and saves it.
Focuses on the 10 most VR-relevant gesture classes from the dataset.

Engineer: Sadad Haidari
"""

import os
import sys
import json
import hashlib
import requests
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import shutil
import time
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetDownloader:
    # VR-relevant gestures optimized for <5ms inference
    VR_GESTURES = [
        'palm',        # Menu selection, open hand interactions
        'fist',        # Object grabbing, closed hand actions
        'peace',       # Mode switching, dual-finger gestures
        'like',        # Thumbs up, confirmation actions
        'ok',          # OK sign, action confirmation
        'one',         # Pointing, cursor control (mapped from 'point' if available)
        'mute',        # Mute/unmute functionality
        'three',       # Number gestures, multi-finger actions
        'call',        # Call gestures, communication
        'stop',        # Stop/halt actions, palm-forward
        'no_gesture'   # Negative samples for robust detection
    ]
    
    # Gesture name mappings
    GESTURE_MAP = {
        'palm': 'palm',
        'fist': 'fist', 
        'peace': 'peace',
        'like': 'like',
        'ok': 'ok',
        'one': 'one',          # or 'point' in some versions
        'mute': 'mute',
        'three': 'three',
        'call': 'call',
        'stop': 'stop',
        'no_gesture': 'no_gesture'
    }
    
    def __init__(self, save_path: str, target_images_per_gesture: int = 10000):
        self.save_path = Path(save_path)
        self.target_images_per_gesture = target_images_per_gesture
        
        # Create directory structure
        self.dataset_dir = self.save_path / 'hagrid_dataset'
        self.annotations_dir = self.save_path / 'hagrid_annotations'
        self.temp_dir = self.save_path / 'temp'
        
        for dir_path in [self.dataset_dir, self.annotations_dir, self.temp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Create gesture subdirectories
        for gesture in self.VR_GESTURES:
            gesture_dir = self.dataset_dir / gesture
            gesture_dir.mkdir(exist_ok=True)
            
        logger.info(f"Initialized downloader for HuggingFace 50GB dataset")
        logger.info(f"Save path: {self.save_path}")
        logger.info(f"Target: {self.target_images_per_gesture} images per gesture")
        
    def check_dependencies(self) -> bool:
        """Check if required dependencies are installed."""
        required_packages = ['datasets', 'PIL', 'numpy']
        missing_packages = []
        
        for package in required_packages:
            try:
                if package == 'datasets':
                    import datasets
                elif package == 'PIL':
                    from PIL import Image
                elif package == 'numpy':
                    import numpy
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing required packages: {', '.join(missing_packages)}")
            logger.error("Please install with: pip install datasets pillow numpy")
            return False
        
        return True
    
    def download_and_process(self) -> bool:
        """
        Download and process dataset.
        
        Returns:
            True if successful
        """
        if not self.check_dependencies():
            return False
            
        try:
            from datasets import load_dataset
            logger.info("Loading dataset...")
            logger.info("This may take several minutes for the 50GB dataset...")
            
            # Load the HuggingFace dataset
            # Note: This loads the 384p version which is perfect for our <5ms target
            dataset = load_dataset(
                "cj-mills/hagrid-sample-500k-384p", 
                cache_dir=str(self.temp_dir),
                trust_remote_code=True
            )
            
            logger.info(f"Dataset loaded successfully!")
            # Handle different dataset types for keys access
            try:
                available_splits = list(dataset.keys())
            except (AttributeError, TypeError):
                available_splits = ['train', 'validation', 'test']  # Default splits
            logger.info(f"Available splits: {available_splits}")
            
            # Process each split
            total_processed = 0
            gesture_counts = {gesture: 0 for gesture in self.VR_GESTURES}
            
            for split_name in ['train', 'validation', 'test']:
                if split_name not in dataset:
                    logger.warning(f"Split '{split_name}' not found in dataset")
                    continue
                
                logger.info(f"Processing {split_name} split...")
                split_data = dataset[split_name]
                
                processed_count = self._process_split(
                    split_data, split_name, gesture_counts
                )
                total_processed += processed_count
                
                logger.info(f"Processed {processed_count} images from {split_name} split")
            
            # Generate final statistics
            self._generate_download_report(gesture_counts, total_processed)
            
            return True
            
        except Exception as e:
            logger.error(f"Download and processing failed: {e}")
            return False
    
    def _process_split(self, split_data, split_name: str, gesture_counts: Dict[str, int]) -> int:
        """
        Process a single data split and extract VR-relevant images.
        
        Args:
            split_data: HuggingFace dataset split
            split_name: Name of the split (train/validation/test)
            gesture_counts: Running count of images per gesture
            
        Returns:
            Number of processed images
        """
        # Create split annotation directory
        split_dir = self.annotations_dir / split_name
        split_dir.mkdir(exist_ok=True)
        
        split_annotations = {}
        processed_count = 0
        
        logger.info(f"Split {split_name} contains {len(split_data)} total samples")
        
        # Process samples with progress bar
        for idx, sample in enumerate(tqdm(split_data, desc=f"Processing {split_name}")):
            try:
                # Extract labels from sample
                labels = sample.get('labels', [])
                if isinstance(labels, str):
                    labels = [labels]
                
                # Check if sample contains VR-relevant gestures
                vr_gestures_found = []
                for label in labels:
                    # Map gesture names to our VR gesture names
                    for vr_gesture, hagrid_gesture in self.GESTURE_MAP.items():
                        if hagrid_gesture.lower() in label.lower():
                            vr_gestures_found.append(vr_gesture)
                            break
                
                if not vr_gestures_found:
                    continue
                
                # Check if we need more samples of these gestures
                needed_gestures = [
                    g for g in vr_gestures_found 
                    if gesture_counts.get(g, 0) < self.target_images_per_gesture
                ]
                
                if not needed_gestures:
                    continue  # We have enough samples of these gestures
                
                # Extract and save image
                image = sample['image']
                if image is None:
                    continue
                
                # Generate unique image ID
                image_id = f"{split_name}_{idx:08d}.jpg"
                
                # Save image to primary gesture directory (use first gesture)
                primary_gesture = needed_gestures[0]
                image_path = self.dataset_dir / primary_gesture / image_id
                
                # Convert and save image
                if hasattr(image, 'save'):
                    image.save(image_path, 'JPEG', quality=95, optimize=True)
                else:
                    # Handle numpy array or other formats
                    if isinstance(image, np.ndarray):
                        Image.fromarray(image).save(image_path, 'JPEG', quality=95, optimize=True)
                    else:
                        logger.warning(f"Unknown image format for sample {idx}")
                        continue
                
                # Create annotation entry
                annotation = {
                    'image_id': image_id,
                    'image_path': str(image_path.relative_to(self.save_path)),
                    'labels': vr_gestures_found,
                    'primary_gesture': primary_gesture,
                    'split': split_name,
                    'original_idx': idx,
                    'bboxes': sample.get('bboxes', []),
                    'user_id': sample.get('user_id', f'unknown_{idx}'),
                    'meta': sample.get('meta', {}),
                    'image_size': [image.width, image.height] if hasattr(image, 'width') and hasattr(image, 'height') else [384, 384]
                }
                
                split_annotations[image_id] = annotation
                
                # Update gesture counts
                for gesture in vr_gestures_found:
                    gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
                
                processed_count += 1
                
                # Log progress every 1000 images
                if processed_count % 1000 == 0:
                    logger.info(f"Processed {processed_count} images, current counts: {dict(list(gesture_counts.items())[:3])}...")
                
            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                continue
        
        # Save split annotations
        if split_annotations:
            annotation_file = split_dir / 'annotations.json'
            with open(annotation_file, 'w') as f:
                json.dump(split_annotations, f, indent=2)
            
            logger.info(f"Saved {len(split_annotations)} annotations to {annotation_file}")
        
        return processed_count
    
    def _generate_download_report(self, gesture_counts: Dict[str, int], total_processed: int):
        """Generate comprehensive download report."""
        
        # Calculate dataset statistics
        total_size_mb = 0
        file_count = 0
        
        for gesture_dir in self.dataset_dir.iterdir():
            if gesture_dir.is_dir() and gesture_dir.name in self.VR_GESTURES:
                for image_file in gesture_dir.glob('*.jpg'):
                    total_size_mb += image_file.stat().st_size / (1024 * 1024)
                    file_count += 1
        
        # Create comprehensive report
        report = {
            'download_info': {
                'source': 'HuggingFace 384p (50GB original)',
                'target_gestures': self.VR_GESTURES,
                'target_per_gesture': self.target_images_per_gesture,
                'download_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'dataset_statistics': {
                'total_images_processed': total_processed,
                'total_images_saved': file_count,
                'total_size_mb': round(total_size_mb, 2),
                'average_image_size_kb': round((total_size_mb * 1024) / file_count, 2) if file_count > 0 else 0
            },
            'gesture_distribution': gesture_counts,
            'quality_metrics': {
                'completion_percentage': {
                    gesture: min(100, (count / self.target_images_per_gesture) * 100)
                    for gesture, count in gesture_counts.items()
                },
                'overall_completion': min(100, (sum(gesture_counts.values()) / (len(self.VR_GESTURES) * self.target_images_per_gesture)) * 100)
            },
            'file_structure': {
                'dataset_dir': str(self.dataset_dir),
                'annotations_dir': str(self.annotations_dir),
                'gesture_subdirectories': [str(self.dataset_dir / gesture) for gesture in self.VR_GESTURES]
            }
        }
        
        # Save report
        report_path = self.save_path / 'download_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Display summary
        self._display_download_summary(report)
        
        logger.info(f"Detailed report saved to: {report_path}")
    
    def _display_download_summary(self, report: Dict):
        """Display a user-friendly download summary."""
        
        print("\n" + "="*80)
        print("DATASET DOWNLOAD COMPLETED!")
        print("="*80)
        
        stats = report['dataset_statistics']
        print(f"Dataset Statistics:")
        print(f"   -> Total images saved: {stats['total_images_saved']:,}")
        print(f"   -> Dataset size: {stats['total_size_mb']:.1f} MB")
        print(f"   -> Average image size: {stats['average_image_size_kb']:.1f} KB")
        print(f"   -> Image resolution: 384×384 pixels (optimized for <5ms inference)")
        
        print(f"\nVR Gesture Distribution:")
        gesture_counts = report['gesture_distribution']
        completion = report['quality_metrics']['completion_percentage']
        
        for gesture in self.VR_GESTURES:
            count = gesture_counts.get(gesture, 0)
            percent = completion.get(gesture, 0)
            status = "✅" if percent >= 90 else "⚠️" if percent >= 50 else "❌"
            print(f"   {status} {gesture:12}: {count:,} images ({percent:.1f}%)")
        
        overall_completion = report['quality_metrics']['overall_completion']
        print(f"\n📈 Overall Completion: {overall_completion:.1f}%")
        
        print(f"\n📁 Files saved to:")
        print(f"   • Images: {report['file_structure']['dataset_dir']}")
        print(f"   • Annotations: {report['file_structure']['annotations_dir']}")
        
        print(f"\n🚀 Ready for VR Hand Tracking Pipeline!")
        print(f"   Next steps:")
        print(f"   1. Run hagrid_preprocessor.py to optimize for <5ms inference")
        print(f"   2. Run vr_gesture_mapper.py to configure VR mappings")
        print(f"   3. Train ultra_fast_cnn.py model")
        
        print("="*80 + "\n")
    
    def cleanup_temp_files(self):
        """Clean up temporary download files."""
        if self.temp_dir.exists():
            logger.info("Cleaning up temporary files...")
            try:
                shutil.rmtree(self.temp_dir)
                logger.info("Temporary files cleaned up successfully")
            except Exception as e:
                logger.warning(f"Could not clean up temp files: {e}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Download dataset for VR hand tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python download_dataset.py --save_path ./data --target_per_gesture 8000

This downloads the HuggingFace 384p version (~50GB) and extracts
~80,000 images (8K per gesture) optimized for <5ms VR inference.

VR Gesture Classes:
  palm, fist, peace, like, ok, one, mute, three, call, stop, no_gesture
        """
    )
    
    parser.add_argument(
        '--save_path', '-p',
        type=str,
        required=True,
        help='Directory to save downloaded dataset'
    )
    
    parser.add_argument(
        '--target_per_gesture', '-t',
        type=int,
        default=10000,
        help='Target number of images per gesture class (default: 10000)'
    )
    
    parser.add_argument(
        '--no-cleanup',
        action='store_true',
        help='Do not clean up temporary files after download'
    )
    
    args = parser.parse_args()
    
    # Display information
    print("Dataset Downloader | Minimal 50GB Version")
    print("="*55)
    print("Source: HaGRID 384p dataset (~50GB) from HuggingFace")
    print("Target: 10 VR-relevant gesture classes")
    print(f"Images per gesture: {args.target_per_gesture}")
    print(f"Expected total: ~{args.target_per_gesture * 11:,} images")
    print(f"Optimized for: <5ms VR hand tracking inference")
    print()
    
    # Confirm download
    response = input("Start download? This will take 10-30 minutes depending on your connection (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Download cancelled.")
        return
    
    # Create downloader and start
    start_time = time.time()
    downloader = DatasetDownloader(args.save_path, args.target_per_gesture)
    
    logger.info("Starting download from HuggingFace...")
    success = downloader.download_and_process()
    
    if success:
        elapsed_time = time.time() - start_time
        logger.info(f"Download completed in {elapsed_time/60:.1f} minutes")
        
        # Clean up if requested
        if not args.no_cleanup:
            downloader.cleanup_temp_files()
        
        print("\nSuccess! Your VR hand tracking dataset is ready!")
        print(f"Next: Run 'python gesture_preprocessor.py --input_path {args.save_path}' to prepare for training")
        
    else:
        print("\nDownload failed! Check the logs above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()