import os
import urllib.request
import sys

def download_with_progress(url, filename):
    """Download a file and show progress"""
    def progress_callback(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\rDownloading {filename}: {percent}%")
        sys.stdout.flush()
    
    print(f"Starting download of {filename}...")
    try:
        urllib.request.urlretrieve(
            url, 
            filename,
            reporthook=progress_callback if sys.stdout.isatty() else None
        )
        print("\nDownload completed successfully!")
        return True
    except Exception as e:
        print(f"\nError downloading {filename}: {str(e)}")
        return False

def main():
    # Create yolo directory if it doesn't exist
    os.makedirs('yolo', exist_ok=True)
    
    # YOLOv3 weights from a mirror
    weights_url = 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov3.weights'
    weights_path = os.path.join('yolo', 'yolov3.weights')
    
    # Download the weights file
    if not os.path.exists(weights_path) or os.path.getsize(weights_path) < 200 * 1024 * 1024:  # Less than 200MB
        if not download_with_progress(weights_url, weights_path):
            print("Failed to download weights. Please try again or download manually.")
            print(f"You can download the weights from: {weights_url}")
            print("And place it in the 'yolo' folder.")
            return False
    
    print("\nWeights file is ready!")
    file_size = os.path.getsize(weights_path) / (1024 * 1024)
    print(f"File size: {file_size:.2f} MB")
    print("You can now run train_video_processor.py")
    return True

if __name__ == "__main__":
    main()
