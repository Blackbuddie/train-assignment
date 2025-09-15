import os
import urllib.request
import sys

def download_file(url, filename):
    print(f"Downloading {filename}...")
    try:
        # Create a request with a user-agent header
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req) as response, open(filename, 'wb') as out_file:
            # Show progress
            file_size = int(response.headers.get('Content-Length', 0))
            if file_size > 0:
                print(f"File size: {file_size/1024/1024:.2f} MB")
            
            # Download in chunks to show progress
            block_size = 8192
            downloaded = 0
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                out_file.write(buffer)
                downloaded += len(buffer)
                if file_size > 0:
                    percent = min(100, int(downloaded * 100 / file_size))
                    print(f"\rProgress: {percent}% ({downloaded/1024/1024:.1f} MB / {file_size/1024/1024:.1f} MB)", end='')
            print()
            
        print(f"Successfully downloaded {filename}")
        return True
    except Exception as e:
        print(f"\nError downloading {filename}: {str(e)}")
        return False

def main():
    # Create yolo directory if it doesn't exist
    os.makedirs('yolo', exist_ok=True)
    
    # YOLO files to download (using alternative mirror for weights)
    files = {
        'yolov3.cfg': 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg',
        'yolov3.weights': 'https://github.com/pjreddie/darknet/releases/download/0.1/yolov3.weights',
        'classes.txt': 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
    }
    
    # Download each file
    success = True
    for filename, url in files.items():
        filepath = os.path.join('yolo', filename)
        if not os.path.exists(filepath):
            if not download_file(url, filepath):
                success = False
    
    if success:
        print("\nAll files downloaded successfully!")
        print("You can now run train_video_processor.py")
    else:
        print("\nSome files failed to download. Please check your internet connection and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
