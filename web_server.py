import cv2
import os
import threading
import queue
import pyttsx3
import time
from datetime import datetime
from fastapi import FastAPI, Response, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="Sentinel Helmet Detection Engine")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Initialize TTS Queue and Worker
tts_queue = queue.Queue()

def tts_worker():
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        while True:
            text = tts_queue.get()
            if text is None:
                break
            engine.say(text)
            engine.runAndWait()
            tts_queue.task_done()
    except Exception as e:
        print(f"TTS Worker Error: {e}")

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

class CameraManager:
    def __init__(self, source=0):
        self.camera = cv2.VideoCapture(source)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.ret = False
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        if self.camera.isOpened():
            self.thread.start()

    def update(self):
        while self.running:
            self.ret, self.frame = self.camera.read()
            time.sleep(0.01)  # Reduce CPU usage slightly

    def read(self):
        return self.ret, self.frame
        
    def release(self):
        self.running = False
        self.camera.release()
        
    def isOpened(self):
        return self.camera.isOpened()

def get_model():
    model_path = "yolov8m-helmet-best.pt"
    if not os.path.exists(model_path):
        print("Downloading Sentinel Model...")
        try:
            downloaded_path = hf_hub_download(repo_id='keremberke/yolov8m-hard-hat-detection', filename='best.pt')
            import shutil
            shutil.copy(downloaded_path, model_path)
        except Exception as e:
            print(f"Error downloading model: {e}")
            # Fallback to a default YOLO model if helmet specific fails
            return YOLO("yolov8n.pt")
    return YOLO(model_path)

model = get_model()

# Class mapping and stats
STATS = {
    "detections": 0,
    "last_detection_time": "N/A",
    "status": "Healthy",
    "fps": 0
}

# Rename classes to "Helmet"
for key in model.names:
    label = model.names[key].lower()
    if 'hard hat' in label or 'hard-hat' in label:
        model.names[key] = 'Helmet'
    elif 'head' in label or 'no hard hat' in label:
        model.names[key] = 'No Helmet'

def generate_frames():
    camera = CameraManager(0)
    if not camera.isOpened():
        print("Webcam disconnected or unavailable.")
        return

    prev_time = time.time()
    
    try:
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                continue
            
            # Simple FPS calculation
            curr_time = time.time()
            STATS["fps"] = int(1 / (curr_time - prev_time)) if curr_time - prev_time > 0 else 0
            prev_time = curr_time

            # Perform detection
            results = model(frame, imgsz=320, verbose=False)
            
            helmet_detected = False
            detection_count = 0
            
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    detection_count += 1
                    if model.names[cls_id] == 'Helmet':
                        helmet_detected = True
            
            if helmet_detected and tts_queue.empty():
                tts_queue.put("Safety helmet detected")
                STATS["detections"] += 1
                STATS["last_detection_time"] = datetime.now().strftime("%H:%M:%S")

            # Render detection results
            annotated_frame = results[0].plot()
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        camera.release()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/video_feed")
async def video_feed():
    return Response(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/api/stats")
async def get_stats():
    return JSONResponse(content=STATS)

if __name__ == "__main__":
    print("Initializing Sentinel AI Web Server...")
    # Using 0.0.0.0 to allow access from local network
    uvicorn.run(app, host="0.0.0.0", port=8000)
