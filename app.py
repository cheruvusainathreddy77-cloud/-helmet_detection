from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import os
from huggingface_hub import hf_hub_download
import threading
import queue
import pyttsx3

app = Flask(__name__)

# Initialize TTS Queue and Worker
tts_queue = queue.Queue()

def tts_worker():
    # Initialize pyttsx3 inside the thread
    engine = pyttsx3.init()
    engine.setProperty('rate', 150) # Slightly faster speech
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
        # Try to reduce buffer size if backend supports it
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret = False
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        if self.camera.isOpened():
            self.thread.start()

    def update(self):
        while self.running:
            # Continuously grab frames to clear the buffer
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

# Rename classes to "Helmet" instead of "hard hat" / "hard-hat"
for key in model.names:
    label = model.names[key].lower()
    if 'hard hat' in label or 'hard-hat' in label:
        model.names[key] = 'Helmet'
    elif 'head' in label or 'no hard hat' in label:
        model.names[key] = 'No Helmet'

def generate_frames():
    # Attempt to open webcam using our threaded Camera object
    camera = CameraSequence(0)
    
    if not camera.isOpened():
        print("Error: Could not access the webcam.")
        return

    while True:
        success, frame = camera.read()
        if not success or frame is None:
            # Wait a tiny bit for the camera thread to grab a frame
            cv2.waitKey(10)
            continue
        
        # Perform detection with smaller imgsz for speed
        results = model(frame, imgsz=320, verbose=False)
        
        # Check for helmet detection
        helmet_detected = False
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if model.names[cls_id] == 'Helmet':
                    helmet_detected = True
                    break
        
        # If helmet detected, push to TTS queue if it's not currently speaking
        if helmet_detected and tts_queue.empty():
            tts_queue.put("Helmet detected")
        
        # Annotated frame
        annotated_frame = results[0].plot()
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        
        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
    camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
