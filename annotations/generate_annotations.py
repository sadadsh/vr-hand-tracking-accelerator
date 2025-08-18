"""
HaGRID Annotation Generator for VR Hand Tracking
Creates training annotations from existing organized dataset

Engineer: Sadad Haidari
Optimized for: All 18 HaGRID gesture classes on Zybo Z7-20 FPGA
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import time
import numpy as np
from PIL import Image
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HaGRIDAnnotationGenerator:
    """
    Generate training annotations from existing HaGRID dataset structure.
    Works with files in place - no copying or processing needed.
    """
    
    def __init__(self, dataset_path: str, output_path: str, num_workers: int | None = None):  # pyright: ignore[reportMissingSuperCall]
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        
        # Use CPU cores for metadata extraction
        self.num_workers = num_workers or min(8, mp.cpu_count() or 1)
        logger.info(f"Using {self.num_workers} workers for metadata extraction")
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # All 18 HaGRID gesture classes
        self.all_gestures = [
            'call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one', 
            'palm', 'peace_inverted', 'peace', 'rock', 'stop_inverted', 
            'stop', 'three', 'three2', 'two_up_inverted', 'two_up'
        ]
        
        # VR action mappings for all gestures
        self.vr_action_map = {
            'palm': 'menu_open',
            'fist': 'grab_object', 
            'ok': 'confirm_action',
            'peace': 'two_finger_select',
            'peace_inverted': 'two_finger_select_alt',
            'call': 'phone_gesture',
            'stop': 'stop_cancel',
            'stop_inverted': 'stop_cancel_alt',
            'like': 'thumbs_up',
            'dislike': 'thumbs_down',
            'mute': 'silence_audio',
            'one': 'select_one',
            'two_up': 'select_two',
            'two_up_inverted': 'select_two_alt',
            'three': 'select_three',
            'three2': 'select_three_alt',
            'four': 'select_four',
            'rock': 'rock_sign'
        }
        
        self.annotations_data = {}
        
        logger.info(f"Initialized annotation generator for all {len(self.all_gestures)} gestures")
    
    def scan_dataset_structure(self) -> Dict[str, List[Path]]:
        """Scan existing dataset structure and count images."""
        print("\n🔍 Scanning existing dataset structure...")
        
        gesture_files = {}
        total_images = 0
        
        for gesture in self.all_gestures:
            gesture_dir = self.dataset_path / f"train_val_{gesture}"
            
            if not gesture_dir.exists():
                print(f"⚠ Missing directory: {gesture_dir}")
                gesture_files[gesture] = []
                continue
            
            # Find all image files
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend(list(gesture_dir.glob(ext)))
            
            gesture_files[gesture] = image_files
            count = len(image_files)
            total_images += count
            
            # Status indicator
            status = "✓" if count > 1000 else "⚠" if count > 100 else "❌" if count > 0 else "✗"
            print(f"  {status} {gesture:15}: {count:,} images")
        
        print(f"\n✓ Total images found: {total_images:,}")
        print(f"✓ Gesture classes with data: {len([g for g in gesture_files if gesture_files[g]])}")
        
        return gesture_files
    
    def load_existing_annotations(self) -> bool:
        """Load existing annotation files if available."""
        print("\n📋 Loading existing annotations...")
        
        annotation_files = list(self.dataset_path.glob("ann_*.json"))
        
        if not annotation_files:
            print("  No annotation files found - will use basic metadata")
            return False
        
        for ann_file in annotation_files:
            try:
                print(f"  Loading: {ann_file.name}")
                
                with open(ann_file, 'r') as f:
                    ann_data = json.load(f)
                
                if isinstance(ann_data, dict):
                    self.annotations_data.update(ann_data)
                    
            except Exception as e:
                print(f"  ❌ Failed to load {ann_file.name}: {e}")
                continue
        
        print(f"✓ Loaded annotations for {len(self.annotations_data):,} images")
        return len(self.annotations_data) > 0
    
    def extract_image_metadata(self, image_path: Path) -> Dict | None:
        """Extract metadata from a single image file."""
        try:
            # Get basic file info
            stat = image_path.stat()
            
            # Get image dimensions
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    mode = img.mode
            except Exception:
                width, height, mode = 0, 0, 'unknown'
            
            # Extract gesture from path
            gesture = image_path.parent.name.replace('train_val_', '')
            
            # Check for existing annotation
            image_filename = image_path.name
            existing_ann = self.annotations_data.get(image_filename, {})
            
            metadata = {
                'image_id': image_path.name,
                'image_path': str(image_path.relative_to(self.dataset_path)),
                'absolute_path': str(image_path),
                'gesture_name': gesture,
                'vr_action': self.vr_action_map.get(gesture, f'gesture_{gesture}'),
                'file_size_bytes': stat.st_size,
                'file_size_mb': round(stat.st_size / (1024 * 1024), 3),
                'image_width': width,
                'image_height': height,
                'image_mode': mode,
                'modification_time': stat.st_mtime,
                'labels': existing_ann.get('labels', [gesture]),
                'bboxes': existing_ann.get('bboxes', []),
                'user_id': existing_ann.get('user_id', 'unknown'),
                'created_timestamp': time.time()
            }
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Failed to extract metadata from {image_path}: {e}")
            return None
    
    def generate_annotations_parallel(self, gesture_files: Dict[str, List[Path]]) -> Dict[str, Dict]:
        """Generate annotations for all images in parallel."""
        print(f"\n🚀 Generating annotations with {self.num_workers} workers...")
        
        # Collect all image paths
        all_image_paths = []
        for _, paths in gesture_files.items():
            all_image_paths.extend(paths)
        
        total_images = len(all_image_paths)
        print(f"📊 Processing metadata for {total_images:,} images...")
        
        all_metadata = {}
        processed_count = 0
        start_time = time.time()
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self.extract_image_metadata, path): path 
                for path in all_image_paths
            }
            
            # Process results with progress bar
            with tqdm(total=len(all_image_paths), desc="Extracting metadata", unit="img") as pbar:
                for future in as_completed(future_to_path):
                    result = future.result()
                    
                    if result is not None:
                        all_metadata[result['image_id']] = result
                        processed_count += 1
                    
                    pbar.update(1)
                    
                    # Update speed in progress bar
                    if processed_count % 1000 == 0:
                        elapsed = time.time() - start_time
                        speed = processed_count / elapsed if elapsed > 0 else 0
                        pbar.set_description(f"Extracting metadata ({speed:.0f}/sec)")
        
        # Summary
        total_time = time.time() - start_time
        final_speed = processed_count / total_time if total_time > 0 else 0
        
        print(f"✅ Metadata extraction completed!")
        print(f"   • Successfully processed: {processed_count:,}/{total_images:,}")
        print(f"   • Processing time: {total_time:.1f} seconds")
        print(f"   • Average speed: {final_speed:.0f} images/second")
        
        return all_metadata
    
    def create_training_splits(self, all_metadata: Dict[str, Dict]) -> Dict[str, List[str]]:
        """Create balanced training/validation/test splits."""
        print("\n📚 Creating training splits...")
        
        # Group by gesture
        gesture_groups = defaultdict(list)
        for image_id, metadata in all_metadata.items():
            gesture_name = metadata['gesture_name']
            gesture_groups[gesture_name].append(image_id)
        
        # Create splits (70% train, 20% val, 10% test)
        splits = {
            'train': [],
            'validation': [],
            'test': []
        }
        
        print("Split distribution per gesture:")
        
        for gesture_name in sorted(gesture_groups.keys()):
            image_ids = gesture_groups[gesture_name]
            n_images = len(image_ids)
            
            if n_images == 0:
                continue
                
            n_train = int(0.7 * n_images)
            n_val = int(0.2 * n_images)
            
            # Shuffle for random split (reproducible)
            import random
            random.seed(42)
            shuffled_ids = image_ids.copy()
            random.shuffle(shuffled_ids)
            
            train_ids = shuffled_ids[:n_train]
            val_ids = shuffled_ids[n_train:n_train + n_val]
            test_ids = shuffled_ids[n_train + n_val:]
            
            splits['train'].extend(train_ids)
            splits['validation'].extend(val_ids)
            splits['test'].extend(test_ids)
            
            print(f"  {gesture_name:15}: {len(train_ids):,} train, {len(val_ids):,} val, {len(test_ids):,} test")
        
        total_train = len(splits['train'])
        total_val = len(splits['validation'])
        total_test = len(splits['test'])
        total_all = total_train + total_val + total_test
        
        print(f"\nOverall splits:")
        print(f"  Train:      {total_train:,} images ({total_train/total_all*100:.1f}%)")
        print(f"  Validation: {total_val:,} images ({total_val/total_all*100:.1f}%)")
        print(f"  Test:       {total_test:,} images ({total_test/total_all*100:.1f}%)")
        
        return splits
    
    def calculate_dataset_statistics(self, all_metadata: Dict[str, Dict]) -> Dict:
        """Calculate comprehensive dataset statistics."""
        print("\nCalculating dataset statistics...")
        
        # Group by gesture
        gesture_stats = defaultdict(list)
        total_size_bytes = 0
        
        for metadata in all_metadata.values():
            gesture_name = metadata['gesture_name']
            gesture_stats[gesture_name].append(metadata)
            total_size_bytes += metadata['file_size_bytes']
        
        # Calculate statistics
        statistics = {
            'overview': {
                'total_images': len(all_metadata),
                'total_gestures': len(gesture_stats),
                'total_size_mb': round(total_size_bytes / (1024 * 1024), 2),
                'total_size_gb': round(total_size_bytes / (1024 * 1024 * 1024), 3),
                'average_file_size_kb': round((total_size_bytes / len(all_metadata)) / 1024, 2) if all_metadata else 0
            },
            'gesture_breakdown': {},
            'image_dimensions': defaultdict(int),
            'file_size_distribution': {
                'under_100kb': 0,
                '100kb_to_500kb': 0,
                '500kb_to_1mb': 0,
                'over_1mb': 0
            }
        }
        
        # Per-gesture statistics
        for gesture_name, images in gesture_stats.items():
            if not images:
                continue
                
            file_sizes = [img['file_size_bytes'] for img in images]
            widths = [img['image_width'] for img in images if img['image_width'] > 0]
            heights = [img['image_height'] for img in images if img['image_height'] > 0]
            
            statistics['gesture_breakdown'][gesture_name] = {
                'image_count': len(images),
                'total_size_mb': round(sum(file_sizes) / (1024 * 1024), 2),
                'avg_file_size_kb': round(np.mean(file_sizes) / 1024, 2),
                'avg_width': int(np.mean(widths)) if widths else 0,
                'avg_height': int(np.mean(heights)) if heights else 0,
                'vr_action': self.vr_action_map.get(gesture_name, f'gesture_{gesture_name}')
            }
        
        # Image dimension distribution
        for metadata in all_metadata.values():
            if metadata['image_width'] > 0 and metadata['image_height'] > 0:
                dim_key = f"{metadata['image_width']}x{metadata['image_height']}"
                statistics['image_dimensions'][dim_key] += 1
        
        # File size distribution
        for metadata in all_metadata.values():
            size_kb = metadata['file_size_bytes'] / 1024
            if size_kb < 100:
                statistics['file_size_distribution']['under_100kb'] += 1
            elif size_kb < 500:
                statistics['file_size_distribution']['100kb_to_500kb'] += 1
            elif size_kb < 1024:
                statistics['file_size_distribution']['500kb_to_1mb'] += 1
            else:
                statistics['file_size_distribution']['over_1mb'] += 1
        
        return statistics
    
    def save_all_annotations(self, all_metadata: Dict[str, Dict], splits: Dict[str, List[str]], statistics: Dict):
        """Save comprehensive annotation files."""
        print("\nSaving annotation files...")
        
        # Main dataset annotations
        dataset_annotations = {
            'dataset_info': {
                'name': 'HaGRID All Gestures for VR Hand Tracking',
                'source_path': str(self.dataset_path),
                'gesture_classes': self.all_gestures,
                'total_gestures': len(self.all_gestures),
                'created_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'created_by': 'HaGRID Annotation Generator'
            },
            'vr_mappings': self.vr_action_map,
            'statistics': statistics,
            'images': all_metadata
        }
        
        # Save main annotations
        main_file = self.output_path / 'hagrid_all_gestures_annotations.json'
        with open(main_file, 'w') as f:
            json.dump(dataset_annotations, f, indent=2, default=str)
        print(f"✓ Main annotations: {main_file}")
        
        # Save training splits
        splits_file = self.output_path / 'train_val_test_splits.json'
        with open(splits_file, 'w') as f:
            json.dump(splits, f, indent=2)
        print(f"✓ Training splits: {splits_file}")
        
        # Save VR training configuration
        vr_config = {
            'model_config': {
                'input_size': [224, 224],  # Standard size, can be resized during training
                'num_classes': len(self.all_gestures),
                'class_names': self.all_gestures,
                'target_device': 'Zybo Z7-20 FPGA',
                'target_latency_ms': 5
            },
            'training_config': {
                'batch_size': 32,
                'learning_rate': 0.001,
                'epochs': 50,
                'early_stopping_patience': 10,
                'data_augmentation': True
            },
            'vr_inference_config': {
                'confidence_threshold': 0.8,
                'max_latency_ms': 5,
                'target_fps': 200,
                'gesture_to_action_map': self.vr_action_map
            },
            'fpga_optimization': {
                'quantization': 'INT4',
                'target_parameters': 8000,
                'target_memory_mb': 2,
                'pipeline_stages': 10
            }
        }
        
        vr_config_file = self.output_path / 'vr_training_config.json'
        with open(vr_config_file, 'w') as f:
            json.dump(vr_config, f, indent=2)
        print(f"✓ VR config: {vr_config_file}")
        
        # Save gesture-specific files
        gesture_dir = self.output_path / 'gesture_annotations'
        gesture_dir.mkdir(exist_ok=True)
        
        for gesture in self.all_gestures:
            gesture_images = {
                img_id: metadata 
                for img_id, metadata in all_metadata.items()
                if metadata['gesture_name'] == gesture
            }
            
            if gesture_images:
                gesture_file = gesture_dir / f'{gesture}_annotations.json'
                with open(gesture_file, 'w') as f:
                    json.dump(gesture_images, f, indent=2, default=str)
        
        print(f"✓ Individual gesture annotations: {gesture_dir}")
        
        # Save class mapping for training
        class_mapping = {
            'class_to_idx': {gesture: idx for idx, gesture in enumerate(self.all_gestures)},
            'idx_to_class': {idx: gesture for idx, gesture in enumerate(self.all_gestures)},
            'num_classes': len(self.all_gestures)
        }
        
        class_file = self.output_path / 'class_mapping.json'
        with open(class_file, 'w') as f:
            json.dump(class_mapping, f, indent=2)
        print(f"✓ Class mapping: {class_file}")
    
    def display_final_report(self, statistics: Dict):
        """Display comprehensive final report."""
        print("\n" + "="*80)
        print("ANNOTATIONS GENERATED!")
        print("="*80)
        
        overview = statistics['overview']
        print(f"Dataset Overview:")
        print(f"   • Total images: {overview['total_images']:,}")
        print(f"   • Total gestures: {overview['total_gestures']}")
        print(f"   • Dataset size: {overview['total_size_gb']:.2f} GB ({overview['total_size_mb']:.1f} MB)")
        print(f"   • Average file size: {overview['average_file_size_kb']:.1f} KB")
        
        print(f"\nAll {len(self.all_gestures)} Gesture Classes:")
        gesture_breakdown = statistics['gesture_breakdown']
        
        # Sort by image count
        sorted_gestures = sorted(
            gesture_breakdown.items(),
            key=lambda x: x[1]['image_count'],
            reverse=True
        )
        
        for gesture_name, stats in sorted_gestures:
            vr_action = stats['vr_action']
            count = stats['image_count']
            size_mb = stats['total_size_mb']
            avg_size = stats['avg_file_size_kb']
            
            status = "EXCELLENT" if count > 2000 else "GOOD" if count > 1000 else "OK" if count > 500 else "LOW"
            print(f"   {gesture_name:15}: {count:,} images ({size_mb:5.1f}MB, {avg_size:3.0f}KB avg) -> {vr_action} ({status})")
        
        print(f"\nFile Distribution:")
        file_dist = statistics['file_size_distribution']
        total_images = overview['total_images']
        for size_range, count in file_dist.items():
            percentage = (count / total_images) * 100 if total_images > 0 else 0
            print(f"   {size_range:15}: {count:,} images ({percentage:.1f}%)")
        
        print(f"\nGenerated Files:")
        print(f"   • Main annotations: hagrid_all_gestures_annotations.json")
        print(f"   • Training splits: train_val_test_splits.json")
        print(f"   • VR configuration: vr_training_config.json")
        print(f"   • Class mapping: class_mapping.json")
        print(f"   • Per-gesture annotations: gesture_annotations/")
        
        print(f"\nReady for model training:")
        print(f"   1. Use annotations for model training (all {len(self.all_gestures)} classes)")
        print(f"   2. Train CNN for FPGA")
        print(f"   3. Apply INT4 quantization")
        print(f"   4. Deploy to Zybo Z7-20 for <5ms inference")
        
        print("="*80)
    
    def generate_annotations(self) -> bool:
        """Main annotation generation pipeline."""
        try:
            # Step 1: Scan existing structure
            gesture_files = self.scan_dataset_structure()
            
            if not any(gesture_files.values()):
                print("No image files found in dataset")
                return False
            
            # Step 2: Load existing annotations if available
            self.load_existing_annotations()
            
            # Step 3: Generate metadata for all images
            all_metadata = self.generate_annotations_parallel(gesture_files)
            
            if not all_metadata:
                print("No metadata generated")
                return False
            
            # Step 4: Create training splits
            splits = self.create_training_splits(all_metadata)
            
            # Step 5: Calculate statistics
            statistics = self.calculate_dataset_statistics(all_metadata)
            
            # Step 6: Save everything
            self.save_all_annotations(all_metadata, splits, statistics)
            
            # Step 7: Display report
            self.display_final_report(statistics)
            
            return True
            
        except Exception as e:
            logger.error(f"Annotation generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function for annotation generation."""
    parser = argparse.ArgumentParser(
        description="Generate training annotations from existing HaGRID dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python generate_annotations.py --dataset_path data/local_dataset --output_path data/annotations --workers 8

Expected dataset structure:
  data/local_dataset/
  ├── ann_train_val.json (optional)
  ├── train_val_call/
  ├── train_val_dislike/
  ├── train_val_fist/
  └── ... (all 18 gesture folders)

This generates training-ready annotations for all 18 gesture classes:
call, dislike, fist, four, like, mute, ok, one, palm, peace_inverted, peace, 
rock, stop_inverted, stop, three, three2, two_up_inverted, two_up

No image processing/copying, it works with existing files in place!
        """
    )
    
    parser.add_argument(
        '--dataset_path', '-d',
        type=str,
        default='data/local_dataset',
        help='Path to existing HaGRID dataset (default: data/local_dataset)'
    )
    
    parser.add_argument(
        '--output_path', '-o',
        type=str,
        default='data/annotations',
        help='Directory to save annotations (default: data/annotations)'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=None,
        help='Number of worker threads (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    # Display startup information
    print("HaGRID Annotation Generator for VR Hand Tracking")
    print("="*60)
    print(f"Dataset path: {args.dataset_path}")
    print(f"Output path: {args.output_path}")
    print(f"Workers: {args.workers or 'auto-detect'}")
    print(f"Target: All 18 HaGRID gesture classes")
    print(f"Mode: Annotation generation (no image processing)")
    print()
    
    # Confirm generation
    response = input("Generate annotations? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Annotation generation cancelled.")
        return
    
    # Create generator and start
    start_time = time.time()
    generator = HaGRIDAnnotationGenerator(
        args.dataset_path,
        args.output_path,
        args.workers
    )
    
    success = generator.generate_annotations()
    
    if success:
        elapsed_time = time.time() - start_time
        print(f"\n🎉 SUCCESS! Annotations generated in {elapsed_time:.1f} seconds")

    else:
        print("\n❌ Annotation generation failed! Check the logs above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()