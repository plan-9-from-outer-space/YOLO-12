
# Import Required Libraries
from flask import Flask, render_template, Response, jsonify, request
from werkzeug.utils import secure_filename
import os
import cv2
import time
from ultralytics import YOLO
# import numpy as np
from numpy import ones as np_ones, uint32 as np_uint32

# Initialize Flask Application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load YOLO Model
model = YOLO("yolov12n.pt")
cocoClassNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                  "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                  "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                  "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                  "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"]

# Variable to store the uploaded video path
uploaded_video_path = None
#Store the Latest Person Count
latest_person_count = 0

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    global uploaded_video_path

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        uploaded_video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(uploaded_video_path)
        print(f"Uploaded file: {uploaded_video_path}")  # Debugging log

        return jsonify({'message': 'File uploaded successfully!', 'video_path': f"/uploads/{filename}"})


def generate_frames():
    """Processes video frames and runs object detection on them."""
    global uploaded_video_path, latest_person_count

    if not uploaded_video_path or not os.path.exists(uploaded_video_path):
        print("Error: No video uploaded.")
        return

    cap = cv2.VideoCapture(uploaded_video_path)
    ptime = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.15, iou=0.1)
        person_count = 0 #Initialize Person Count
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), [255, 0, 0], 2)
                classNameInt = int(box.cls[0])
                classname = cocoClassNames[classNameInt]
                if classname == "person": #Count Only Person
                    person_count+=1
                conf = round(box.conf[0].item(), 2)
                label = f"{classname}: {conf}"
                text_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                c2 = x1 + text_size[0], y1 - text_size[1] - 3
                cv2.rectangle(frame, (x1, y1), c2, [255, 0, 0], -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 1)
        #Update the Latest Person Count
        latest_person_count = person_count
        # Display person count on the frame
        cv2.putText(frame, f"Persons: {person_count}", (30, 100),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(frame, f"FPS: {int(fps)}", (30, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

def generate_heatmap():
    """Processes video frames and runs object detection on them."""
    global uploaded_video_path

    if not uploaded_video_path or not os.path.exists(uploaded_video_path):
        print("Error: No video uploaded.")
        return

    cap = cv2.VideoCapture(uploaded_video_path)
    # globalImgArray = None

    # Calculate the frame width and frame height
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # print("Frame Width", w, "Frame Height", h)

    globalImgArray = np_ones([int(h), int(w)], dtype=np_uint32)

    while True:
        ret, frame = cap.read()
        if not ret: break

        results = model.predict(frame, conf=0.15, iou=0.1)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), [255, 0, 0], 2)
                classNameInt = int(box.cls[0])
                classname = cocoClassNames[classNameInt]
                conf = round(box.conf[0].item(), 2)
                label = f"{classname}: {conf}"
                text_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                c2 = x1 + text_size[0], y1 - text_size[1] - 3
                cv2.rectangle(frame, (x1, y1), c2, [255, 0, 0], -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 1)
                # Create the Intensity Heatmaps
                globalImgArray[y1:y2, x1:x2] += 1
        
        # Prepare the heatmap for overlay
        globalImgArrayNorm = (globalImgArray - globalImgArray.min()) / (globalImgArray.max() - globalImgArray.min()) * 255
        globalImgArrayNorm = globalImgArrayNorm.astype('uint8')
        globalImgArrayNorm = cv2.GaussianBlur(globalImgArrayNorm, (9, 9), 0)
        heatMapImg = cv2.applyColorMap(globalImgArrayNorm, cv2.COLORMAP_JET)
        superImposedFrame = cv2.addWeighted(heatMapImg, 0.5, frame, 0.5, 0)

        # Prepare the jpeg frame to send to the frontend
        ret, buffer = cv2.imencode('.jpg', superImposedFrame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/video_feed')
def video_feed():
    """Provides the video stream to the frontend if a video has been uploaded."""
    global uploaded_video_path
    if not uploaded_video_path or not os.path.exists(uploaded_video_path):
        return jsonify({'error': 'No video uploaded'}), 400
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/person_count')
def person_count():
    """API to return the latest person count."""
    return jsonify({"count": latest_person_count})

@app.route('/generate_map')
def generate_map():
    """Generates a heatmap."""
    global uploaded_video_path
    if not uploaded_video_path or not os.path.exists(uploaded_video_path):
        return jsonify({'error': 'No video uploaded'}), 400
    return Response(generate_heatmap(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
