"""Test script to diagnose video loading issues."""
import sys
from pathlib import Path
from napari_video.napari_video import VideoReaderNP

def test_video_reader(path):
    """Test VideoReaderNP with a video file."""
    print(f"Testing video file: {path}")
    print(f"File exists: {Path(path).exists()}")
    
    try:
        print("Creating VideoReaderNP...")
        vr = VideoReaderNP(str(path), remove_leading_singleton=True)
        print(f"✓ VideoReaderNP created successfully")
        
        print(f"Shape: {vr.shape}")
        print(f"Dtype: {vr.dtype}")
        print(f"Number of frames: {len(vr)}")
        print(f"ndim: {vr.ndim}")
        
        # Try to access first frame
        print("Accessing first frame...")
        first_frame = vr[0]
        print(f"✓ First frame shape: {first_frame.shape}")
        print(f"✓ First frame dtype: {first_frame.dtype}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_video_reader(sys.argv[1])
    else:
        print("Usage: python test_video_load.py <video_path>")

