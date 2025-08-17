"""
Quick test script for dataset download verification
"""

import subprocess
import sys
import time
from pathlib import Path

def test_download():
    print("Testing dataset download...")

    test_path = "./test_data"

    cmd = [
        sys.executable, "download_dataset.py",
        "--save_path", test_path,
        "--target_per_gesture", "100", # Just 100 images per gesture for testing
        "--no-cleanup" # Keep the files for debugging if needed
    ]

    print("Running:", " ".join(cmd))
    
    start_time = time.time()

    try:
        # Run the download script
        result = subprocess.run(cmd, capture_output=True, text=True, input="y\n")
        elapsed = time.time() - start_time

        if result.returncode == 0:
            # Verify structure
            test_path_obj = Path(test_path)
            dataset_dir = test_path_obj / "dataset"

            if dataset_dir.exists():
                gesture_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
                image_count = sum(len(list(d.glob("*.jpg"))) for d in gesture_dirs)

                print(f"TEST PASSED in {elapsed:1f}s")
                print(f"   - Found {len(gesture_dirs)} gesture directories")
                print(f"   - Downloaded {image_count} test images")
                return True
            else:
                print(f"TEST FAILED: No dataset directory created")
                return False
        else:
            print(f"TEST FAILED: Script returned error:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"TEST FAILED: Exception: {e}")
        return False
    
    def cleanup_test():
        test_path = Path("./test_data")
        if test_path.exists():
            import shutil
            shutil.rmtree(test_path)
            print("Test data cleaned up")
    if __name__ == "__main__":
        success = test_download()

        if success:
            print("\nDownload script works! Good to move on.")
            print("\nNext steps:")
            print("1. Run full download: python download_dataset.py --save_path ./data")
            print("2. Move to preprocessing phase")
        
            cleanup_choice = input("\nClean up test files? (Y/n): ").strip().lower()
            if cleanup_choice != 'n':
                cleanup_test()
        else:
            print("\n🔧 Fix the issues above and try again")