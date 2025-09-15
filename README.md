# Train Coach Detection and Analysis System

## Project Overview
This project is designed to process train videos, detect individual coaches, and generate a comprehensive analysis report. It uses computer vision techniques to analyze train composition and generate detailed reports.

## Key Features

- **Video Processing**: Splits input train video into individual coach segments
- **Object Detection**: Uses YOLO (You Only Look Once) for object detection
- **Frame Extraction**: Extracts and analyzes frames at regular intervals
- **HTML Reporting**: Generates detailed HTML reports with summary statistics
- **Multi-threaded Processing**: Utilizes concurrent processing for improved performance

## Technologies Used

- **Python 3.x**
- **OpenCV** - For video processing and image manipulation
- **YOLO (v3)** - For object detection
- **Jinja2** - For HTML report templating
- **MoviePy** - For video processing
- **NumPy** - For numerical operations

## Setup Instructions

1. **Prerequisites**:
   - Python 3.7 or higher
   - pip (Python package manager)

2. **Install Dependencies**:
   ```bash
   pip install opencv-python numpy moviepy jinja2
   ```

3. **Download YOLO Weights and Config**:
   Run the following scripts to download required YOLO files:
   ```bash
   python download_yolo.py
   python download_weights.py
   ```

4. **Project Structure**:
   ```
   project/
   ├── yolo/
   │   ├── yolov3.cfg
   │   ├── yolov3.weights
   │   └── classes.txt
   ├── input_video.mp4
   ├── train_video_processor.py
   ├── download_yolo.py
   ├── download_weights.py
   └── output/
   ```

## Usage

1. Place your input video file as `input_video.mp4` in the project root
2. Run the main processing script:
   ```bash
   python train_video_processor.py
   ```
3. The processed output will be saved in the `output/` directory
4. View the generated HTML report: `output/12309_coverage_report.html`

## Report Features

- **Summary Section**:
  - Train number and processing date
  - Total coaches processed
  - Number of engines detected
  - Coaches with tail sections
  - Total frames processed

- **Detailed View**:
  - Individual coach images
  - Object detection visualizations
  - Coach identification numbers

## Limitations

1. Requires a clear video of the train for accurate coach detection
2. Performance may vary based on video quality and lighting conditions
3. Currently optimized for a specific train configuration (2 engines)
4. Requires sufficient disk space for processing large videos

## Video Submission

Please find the demonstration videos at:
- Project Features: [Your Google Drive Link Here]
- Technical Walkthrough: [Your Google Drive Link Here]

## Future Improvements

1. Add support for real-time processing
2. Implement machine learning for better coach detection
3. Add support for multiple train configurations
4. Create a web interface for easier interaction
5. Add more detailed analytics and reporting

## License

This project is for educational purposes only.

## Contact

For any queries, please contact [Your Email]
