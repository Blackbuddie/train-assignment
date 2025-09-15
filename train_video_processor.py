import cv2
import os
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from jinja2 import Template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_PATH = os.path.join(BASE_DIR, 'input_video.mp4')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
TRAIN_NUMBER = '12309'
YOLO_CFG = os.path.join(BASE_DIR, 'yolo', 'yolov3.cfg')
YOLO_WEIGHTS = os.path.join(BASE_DIR, 'yolo', 'yolov3.weights')
YOLO_CLASSES = os.path.join(BASE_DIR, 'yolo', 'classes.txt')
COACH_INTERVAL_SEC = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading YOLO model...")
if not all(os.path.exists(f) for f in [YOLO_WEIGHTS, YOLO_CFG, YOLO_CLASSES]):
    raise FileNotFoundError("YOLO model files not found. Please ensure you have the following files in the 'yolo' directory:\n"
                         f"- {YOLO_CFG}\n- {YOLO_WEIGHTS}\n- {YOLO_CLASSES}")

try:
    net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CFG)
    with open(YOLO_CLASSES, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    layer_names = net.getLayerNames()
    output_layers_indices = net.getUnconnectedOutLayers()
    output_layers = []
    
    if len(output_layers_indices) == 0:
        output_layers = [layer_names[-1]]
    else:
        for i in output_layers_indices:
            idx = i.item() if hasattr(i, 'item') else i
            if isinstance(idx, (list, np.ndarray)):
                idx = idx[0]
            try:
                layer_name = layer_names[idx - 1] if idx > 0 else layer_names[idx]
                output_layers.append(layer_name)
            except (IndexError, TypeError):
                continue
    
    if not output_layers and layer_names:
        output_layers = [layer_names[-1]]
    
    print("YOLO model loaded successfully!")
    print(f"Output layers: {output_layers}")
    print(f"Available classes: {len(classes)}")
except Exception as e:
    raise RuntimeError(f"Failed to load YOLO model: {str(e)}")

def detect_objects_yolo(img):
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int((detection[0] * width) - w/2)
                y = int((detection[1] * height) - h/2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return [(classes[class_ids[i]], confidences[i], boxes[i]) for i in indexes] if len(indexes) > 0 else []

def split_video_into_coaches(video_path, output_dir, train_num, interval_sec=COACH_INTERVAL_SEC):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    
    segments = int(duration_sec // interval_sec)
    segment_paths = []
    
    for i in range(segments):
        start_time = i * interval_sec
        cap.release()
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        coach_folder = os.path.join(output_dir, f"{train_num}_{i+1}")
        os.makedirs(coach_folder, exist_ok=True)
        segment_path = os.path.join(coach_folder, f"{train_num}_{i+1}.mp4")
        
        out = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if current_time > start_time + interval_sec:
                break
            if out is None:
                height, width, _ = frame.shape
                out = cv2.VideoWriter(segment_path, fourcc, fps, (width, height))
            out.write(frame)
        if out:
            out.release()
        segment_paths.append(segment_path)
    
    cap.release()
    return segment_paths

def extract_frames(video_path, output_folder, train_num, coach_counter, frame_interval=5):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = 0
    saved_frames = []
    
    print(f"Processing {total_frames} frames (every {frame_interval} frames)...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_num += 1
        if frame_num % frame_interval != 0:
            continue
            
        if frame_num % (frame_interval * 10) == 0:
            progress = (frame_num / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_num}/{total_frames})", end='\r')
        
        img_name = f"{train_num}_{coach_counter}_{frame_num:04d}.jpg"
        img_path = os.path.join(output_folder, img_name)
        cv2.imwrite(img_path, frame)
        saved_frames.append(img_path)
    
    print("\nFinished processing frames.")
    cap.release()
    return saved_frames

def annotate_frames(frame_paths):
    total_frames = len(frame_paths)
    print(f"Annotating {total_frames} frames...")
    
    for idx, img_path in enumerate(frame_paths, 1):
        if idx % 5 == 0 or idx == total_frames:
            progress = (idx / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({idx}/{total_frames})", end='\r')
        
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            detections = detect_objects_yolo(img)
            
            # Only process frames with detections
            if detections:
                for label, conf, (x,y,w,h) in detections:
                    color = (0, 255, 0) if label == 'door_closed' else (0, 0, 255)
                    cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
                    cv2.putText(img, f"{label} {conf:.2f}", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.imwrite(img_path, img)
        except Exception as e:
            print(f"\nError processing {img_path}: {str(e)}")

def process_tail_coach(video_path, output_dir, train_num, coach_id):
    tail_coach_dir = os.path.join(output_dir, f"{train_num}_{coach_id}_tail")
    os.makedirs(tail_coach_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 150))
    
    ret, frame = cap.read()
    if ret:
        img_path = os.path.join(tail_coach_dir, f"{train_num}_{coach_id}_tail.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Saved tail coach image: {img_path}")

def generate_html_report(train_num: str, coaches: int, output_dir: str):
    from datetime import datetime
    
    report_path = os.path.join(output_dir, f"{train_num}_coverage_report.html")
    processing_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    summary = {
        'train_number': train_num,
        'total_coaches': coaches,
        'engines_count': 2,  # Based on assignment hint
        'processing_date': processing_date,
        'coaches_with_tail': 0,
        'total_frames_processed': 0
    }
    
    coach_images = []
    for coach_id in range(1, coaches + 1):
        coach_folder = os.path.join(output_dir, f"{train_num}_{coach_id}")
        if not os.path.exists(coach_folder):
            continue
            
        images = sorted([f for f in os.listdir(coach_folder) if f.endswith('.jpg')])
        if not images:
            continue
            
        middle_idx = len(images) // 2
        selected_img = images[middle_idx]
        full_path = os.path.join(f"{train_num}_{coach_id}", selected_img)
        coach_images.append((coach_id, full_path))
        
        tail_coach_dir = os.path.join(output_dir, f"{train_num}_{coach_id}_tail")
        if os.path.exists(tail_coach_dir):
            tail_images = [f for f in os.listdir(tail_coach_dir) if f.endswith('.jpg')]
            if tail_images:
                summary['coaches_with_tail'] += 1
                tail_path = os.path.join(f"{train_num}_{coach_id}_tail", tail_images[0])
                coach_images.append((f"{coach_id} (Tail)", tail_path))
                
        summary['total_frames_processed'] += len(images)
    
    template_str = """
    <html>
    <head>
        <title>Coverage Report - Train {{ train_number }}</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; }
            .summary-card {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 30px;
            }
            .summary-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-top: 15px;
            }
            .summary-item {
                background: white;
                padding: 15px;
                border-radius: 6px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .summary-item h3 {
                margin-top: 0;
                color: #2c3e50;
                font-size: 1.1em;
            }
            .summary-item p {
                margin: 5px 0 0;
                font-size: 1.5em;
                font-weight: bold;
                color: #3498db;
            }
            .coach-container { 
                margin: 20px 0; 
                padding: 20px; 
                border: 1px solid #e0e0e0; 
                border-radius: 8px;
                background: #fff;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            }
            .coach-image { 
                max-width: 100%; 
                height: auto; 
                display: block;
                margin: 15px 0;
                border: 1px solid #eee;
                border-radius: 4px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            h1 { 
                color: #2c3e50; 
                margin-bottom: 5px;
            }
            .subtitle {
                color: #7f8c8d;
                margin-top: 0;
                font-weight: normal;
            }
            h2 {
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 8px;
                margin-top: 30px;
            }
        </style>
    </head>
    <body>
      <h1>Train Coverage Report</h1>
      <p class="subtitle">Generated on {{ processing_date }}</p>
      
      <div class="summary-card">
        <h2>Summary</h2>
        <div class="summary-grid">
          <div class="summary-item">
            <h3>Train Number</h3>
            <p>{{ train_number }}</p>
          </div>
          <div class="summary-item">
            <h3>Total Coaches</h3>
            <p>{{ total_coaches }}</p>
          </div>
          <div class="summary-item">
            <h3>Engines</h3>
            <p>{{ engines_count }}</p>
          </div>
          <div class="summary-item">
            <h3>Coaches with Tail</h3>
            <p>{{ coaches_with_tail }}</p>
          </div>
          <div class="summary-item">
            <h3>Frames Processed</h3>
            <p>{{ "{:,}".format(total_frames_processed) }}</p>
          </div>
        </div>
      </div>
      
      <h2>Coach Images</h2>
      
      <div class="coach-images">
        {% for coach_id, img in coach_images %}
        <div class="coach-container">
            <h3>Coach {{ coach_id }}</h3>
            <img src="{{ img }}" class="coach-image" alt="Coach {{ coach_id }}" />
        </div>
        {% endfor %}
      </div>
    </body>
    </html>
    """
    template = Template(template_str)
    html_content = template.render(
        train_number=summary['train_number'],
        total_coaches=summary['total_coaches'],
        engines_count=summary['engines_count'],
        coaches_with_tail=summary['coaches_with_tail'],
        total_frames_processed=summary['total_frames_processed'],
        processing_date=summary['processing_date'],
        coach_images=coach_images
    )
    with open(report_path, 'w') as f:
        f.write(html_content)
    print(f"Report generated: {report_path}")

import concurrent.futures
import time

def process_coach(coach_info):
    idx, coach_video = coach_info
    coach_num = idx + 1
    print(f"\nProcessing coach {coach_num}...")
    
    coach_folder = os.path.dirname(coach_video)
    frames_dir = os.path.join(coach_folder, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    frame_paths = extract_frames(coach_video, frames_dir, TRAIN_NUMBER, coach_num)
    annotate_frames(frame_paths)
    
    if coach_num == 1:
        process_tail_coach(coach_video, os.path.dirname(coach_folder), TRAIN_NUMBER, coach_num)
    
    return coach_num

def main():
    print("Starting video processing pipeline...")
    start_time = time.time()
    
    coach_videos = split_video_into_coaches(VIDEO_PATH, OUTPUT_DIR, TRAIN_NUMBER)
    print(f"\nDetected and split into {len(coach_videos)} coach videos.")
    
    # Process coaches in parallel (using max_workers based on CPU cores)
    max_workers = min(4, os.cpu_count() or 2)  # Use up to 4 workers
    print(f"Processing coaches in parallel using {max_workers} workers...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all coach processing tasks
        future_to_coach = {
            executor.submit(process_coach, (idx+1, coach_video)): idx+1 
            for idx, coach_video in enumerate(coach_videos)
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_coach):
            coach_num = future_to_coach[future]
            try:
                future.result()
            except Exception as exc:
                print(f'Coach {coach_num} generated an exception: {exc}')
    
    # Process tail coach (last 5 seconds of the video)
    if coach_videos:
        last_coach_video = coach_videos[-1]
        process_tail_coach(last_coach_video, OUTPUT_DIR, TRAIN_NUMBER, len(coach_videos))
    
    # Generate final report
    print("\nGenerating final report...")
    generate_html_report(TRAIN_NUMBER, len(coach_videos), OUTPUT_DIR)
    
    total_duration = time.time() - start_time
    print(f"\n--- Processing completed in {total_duration/60:.1f} minutes ---")

if __name__ == "__main__":
    main()
