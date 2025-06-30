import cv2
import os
from ultralytics import YOLO

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')

def process_video(video_path, output_path, model):
    print(f"Processing {video_path}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, imgsz=640, conf=0.4)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        cv2.imshow("YOLOv8 Video Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Detection stopped by user.")
            break

    cap.release()
    out.release()
    print(f"Output saved to {output_path}")

def main():
    videos_dir = 'videos'
    output_dir = 'outputs'
    model = YOLO('yolov8n.pt')

    video_files = [f for f in os.listdir(videos_dir) if f.lower().endswith(VIDEO_EXTENSIONS)]
    if not video_files:
        print(f"No videos found in {videos_dir} folder.")
        return

    for video_file in video_files:
        input_path = os.path.join(videos_dir, video_file)
        output_path = os.path.join(output_dir, f"annotated_{video_file}")
        process_video(input_path, output_path, model)

    cv2.destroyAllWindows()
    print("All videos processed.")

if __name__ == "__main__":
    main()
