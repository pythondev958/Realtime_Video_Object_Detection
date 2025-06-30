# Realtime Video Object Detection with YOLOv8

This project demonstrates real-time object detection in videos using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics). It processes all video files from a folder and outputs annotated videos showing detected objects like people, cars, and more.

## Features

- Batch processing: automatically detects and processes all videos in the `videos/` folder.
- Real-time object detection using YOLOv8 (`yolov8n.pt` model).
- Outputs saved with bounding boxes in the `outputs/` folder.
- Live visualization while processing (press `q` to quit early).

---
##  Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/pythondev958/Realtime_Video_Object_Detection.git
cd Realtime_Video_Object_Detection


python -m venv venvvenv\Scripts\activate #

On Windows# source venv/bin/activate # On Linux/macOS

pip install -r requirements.txt

python video_detector.py

## Model Used

yolov8n.pt (YOLOv8 Nano) – lightweight and fast for real-time applications.

Downloaded automatically via ultralytics package.
