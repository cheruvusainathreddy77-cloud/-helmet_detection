from flask import Flask, render_template, Response, jsonify
import cv2
import os
import threading
import queue
import pyttsx3
import time
from datetime import datetime
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

app = Flask(__name__)

# System Statistics
STATS = {
    "detections": 0,
    "last_detection_time": "N/A",
    "status": "Initializing...",
    "fps": 0
}

# Initialize TTS Queue and Worker
tts_queue = queue.Queue()

def tts_worker():
    try:
        # Initialize pyttsx3 inside the thread
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
        print(f"TTS Worker Error (speech might be disabled): {e}")

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

class CameraSequence:
    def __init__(self, source=0):
        self.camera = cv2.VideoCapture(source)
        # Low latency settings
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.ret = False
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        if self.camera.isOpened():
            self.thread.start()
            STATS["status"] = "Camera Connected"
        else:
            STATS["status"] = "Camera Error"

    def update(self):
        while self.running:
            self.ret, self.frame = self.camera.read()
            time.sleep(0.01)

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
        print("Downloading AI weights...")
        try:
            downloaded_path = hf_hub_download(repo_id='keremberke/yolov8m-hard-hat-detection', filename='best.pt')
            import shutil
            shutil.copy(downloaded_path, model_path)
        except Exception as e:
            print(f"Download failed: {e}. Attempting fallback...")
            return YOLO("yolov8n.pt")
    return YOLO(model_path)

# Initialize model
model = get_model()

# Map classes
for key in model.names:
    label = model.names[key].lower()
    if 'hard hat' in label or 'hard-hat' in label:
        model.names[key] = 'Helmet'
    elif 'head' in label or 'no hard hat' in label:
        model.names[key] = 'No Helmet'

def generate_frames():
    camera = CameraSequence(0)
    
    if not camera.isOpened():
        print("Critical Error: Web camera not accessible.")
        return

    prev_time = time.time()
    
    try:
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                continue
            
            # FPS Calculation
            curr_time = time.time()
            STATS["fps"] = int(1 / (curr_time - prev_time)) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            # AI Inference
            results = model(frame, imgsz=320, verbose=False)
            
            helmet_detected = False
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    if model.names[cls_id] == 'Helmet':
                        helmet_detected = True
            
            if helmet_detected and tts_queue.empty():
                try:
                    tts_queue.put("Helmet detected")
                except:
                    pass
                STATS["detections"] += 1
                STATS["last_detection_time"] = datetime.now().strftime("%H:%M:%S")

            # Plotting
            annotated_frame = results[0].plot()
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        camera.release()

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def get_stats():
    return jsonify(STATS)

if __name__ == "__main__":
    print("------------------------------------------")
    print(" SENTINEL FLASK SERVER STARTING...")
    print(" URL: http://127.0.0.1:5000")
    print("------------------------------------------")
    # Using Flask's robust production settings for local dev
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
