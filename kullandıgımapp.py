from flask import Flask, Response, render_template, request, redirect, url_for, send_from_directory, flash
import os
import cv2
from ultralytics import YOLO
from character_ai import weapon_detected_c
from telegram_mes import send_telegram_photo
from detect import detect_objects_in_photo, detect_objects_in_video
import threading
from gradient_ai import weapon_detected_g
app = Flask(__name__)
app.secret_key = 'stringg'  # Needed for flashing messages
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the uploads directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global flag to control webcam capture
global capturing
capturing = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_image', methods=['POST'])
def detect_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        original_path, result_path, detected_weapons, ai_response = detect_objects_in_photo(filepath)
        detected_filename = os.path.basename(result_path)
        if not detected_weapons:
            flash("No weapons are detected in this image.")
        return render_template('result.html', detected_filename=detected_filename, detected_weapons=detected_weapons, ai_response=ai_response)

@app.route('/detect_video', methods=['POST'])
def detect_video():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        result_video_path = detect_objects_in_video(filepath)
        video_filename = os.path.basename(result_video_path)
        return render_template('result_video.html', video_filename=video_filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/webcam')
def webcam():
    global capturing
    return render_template('webcam.html', capturing=capturing)

@app.route('/start_webcam')
def start_webcam():
    global capturing
    capturing = True
    return redirect(url_for('webcam'))

@app.route('/stop_webcam')
def stop_webcam():
    global capturing
    capturing = False
    return redirect(url_for('webcam'))

def process_frame(frame, yolo_model):
    results = yolo_model(frame)
    for result in results:
        classes = result.names
        cls = result.boxes.cls
        conf = result.boxes.conf
        detections = result.boxes.xyxy

        for pos, detection in enumerate(detections):
            if conf[pos] >= 0.6:
                xmin, ymin, xmax, ymax = detection
                label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}"

                # Call weapon_detected_c when a weapon is detected
                threading.Thread(target=weapon_detected_g, args=(classes[int(cls[pos])],)).start()

                # Save the frame with the detected object
                photo_path = os.path.join(app.config['UPLOAD_FOLDER'], "detected_weapon.jpg")
                cv2.imwrite(photo_path, frame)
                threading.Thread(target=send_telegram_photo, args=(photo_path,)).start()

                color = (0, int(cls[pos]), 255)
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame

def gen_frames():
    global capturing
    yolo_model = YOLO('./runs/detect/Normal_Compressed/weights/last.pt')
    video_capture = cv2.VideoCapture(0)  # Use 0 for the default webcam
    video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))

    if not video_capture.isOpened():
        raise RuntimeError("Could not start webcam.")

    frame_count = 0
    process_every_n_frames = 3

    while capturing:
        success, frame = video_capture.read()
        if not success:
            break

        frame_count += 1
        if frame_count % process_every_n_frames == 0:
            frame = process_frame(frame, yolo_model)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video_capture.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
