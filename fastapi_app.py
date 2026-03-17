import cv2
import os
import threading
import queue
import pyttsx3
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import uvicorn

app = FastAPI(title="Helmet Detection API")
templates = Jinja2Templates(directory="templates")

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

def get_model():
    model_path = "yolov8m-helmet-best.pt"
    if not os.path.exists(model_path):
        print("Downloading YOLOv8 Helmet Detection model...")
        downloaded_path = hf_hub_download(repo_id='keremberke/yolov8m-hard-hat-detection', filename='best.pt')
        import shutil
        shutil.copy(downloaded_path, model_path)
    return YOLO(model_path)

model = get_model()

# Rename classes to "Helmet"
for key in model.names:
    label = model.names[key].lower()
    if 'hard hat' in label or 'hard-hat' in label:
        model.names[key] = 'Helmet'
    elif 'head' in label or 'no hard hat' in label:
        model.names[key] = 'No Helmet'

def generate_frames():
    camera = CameraSequence(0)
    if not camera.isOpened():
        print("Error: Could not access the webcam.")
        return

    try:
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                continue
            
            results = model(frame, imgsz=320, verbose=False)
            helmet_detected = False
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    if model.names[cls_id] == 'Helmet':
                        helmet_detected = True
                        break
            
            if helmet_detected and tts_queue.empty():
                tts_queue.put("Helmet detected")
            
            annotated_frame = results[0].plot()
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        camera.release()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video_feed")
async def video_feed():
    return Response(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("Starting FastAPI Helmet Detection Server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
