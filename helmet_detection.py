import argparse
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import cv2
import os
import threading
import queue
import pyttsx3

# Initialize TTS Queue and Worker
tts_queue = queue.Queue()

def tts_worker():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    while True:
        text = tts_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
        tts_queue.task_done()

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

class CameraSequence:
    def __init__(self, source=0):
        self.camera = cv2.VideoCapture(source)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret = False
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        if self.camera.isOpened():
            self.thread.start()

    def update(self):
        while self.running:
            self.ret, self.frame = self.camera.read()

    def read(self):
        return self.ret, self.frame
        
    def release(self):
        self.running = False
        self.camera.release()
        
    def isOpened(self):
        return self.camera.isOpened()

def download_model():
    model_path = "yolov8m-helmet-best.pt"
    if not os.path.exists(model_path):
        print("Downloading YOLOv8 Helmet Detection model...")
        downloaded_path = hf_hub_download(repo_id='keremberke/yolov8m-hard-hat-detection', filename='best.pt')
        import shutil
        shutil.copy(downloaded_path, model_path)
        print("Model saved to:", model_path)
    return model_path

def main():
    parser = argparse.ArgumentParser(description="Helmet detection using YOLOv8")
    parser.add_argument('--source', type=str, default='0', help='Video source (0 for webcam, or path to video file/image)')
    args = parser.parse_args()

    model_path = download_model()
    
    # Load model
    print("Loading YOLOv8 model...")
    model = YOLO(model_path)
    
    # Process source
    source = args.source
    if source.isdigit():
        source = int(source)
    
    cap = CameraSequence(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    print(f"Starting detection on source {source}. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            cv2.waitKey(10)
            continue
        
        # Run inference (fast mode)
        results = model(frame, imgsz=320, verbose=False)
        
        # Check for helmet detection
        helmet_detected = False
        for r in results:
            for box in r.boxes:
                # Need to use model.names and handle string conversion like app.py
                cls_id = int(box.cls[0])
                label = model.names[cls_id].lower()
                if 'hard hat' in label or 'hard-hat' in label or 'helmet' in label:
                    helmet_detected = True
                    break
                    
        # If helmet detected, speak
        if helmet_detected and tts_queue.empty():
            tts_queue.put("Helmet detected")
        
        # Visualize findings
        annotated_frame = results[0].plot()
        
        # Display the resulting frame
        cv2.imshow('Helmet Detection (Construction & Traffic)', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
