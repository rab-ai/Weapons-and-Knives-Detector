"""
from flask import Flask, Response, render_template, request, redirect, url_for, send_from_directory, flash, jsonify
import os
from detect import detect_objects_in_photo, detect_objects_in_video #, detect_objects_from_webcam
import cv2
from ultralytics import YOLO
from telegram_mes import send_telegram_message, send_telegram_photo
import json
from character_ai import weapon_detected_c
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
        original_path, result_path, detected_weapons = detect_objects_in_photo(filepath)
        original_filename = os.path.basename(original_path)
        detected_filename = os.path.basename(result_path)
        if not detected_weapons:
            flash("No weapons are detected in this image.")
        return render_template('result.html', original_filename=original_filename, detected_filename=detected_filename, detected_weapons=detected_weapons)

def get_characterai_response(detected_weapons):
    # Your code to get CharacterAI response
    res = weapon_detected_c(detected_weapons)
    return res  # Replace with actual response

@app.route('/retrieve_ai_response_and_send_telegram', methods=['POST'])
def retrieve_ai_response_and_send_telegram():
    data = request.get_json()
    original_filename = data['original_filename']
    detected_filename = data['detected_filename']
    detected_weapons = data['detected_weapons']

    # Retrieve CharacterAI response
    ai_responses = get_characterai_response(detected_weapons)

    # Send the detected image and CharacterAI responses to Telegram
    send_telegram_photo(os.path.join(app.config['UPLOAD_FOLDER'], detected_filename))
    send_telegram_message("Detected objects: " + ", ".join(map(str, detected_weapons)))

    return jsonify({
        "message": "CharacterAI response retrieved and Telegram message sent successfully.",
        "ai_responses": ai_responses
    })

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

def gen_frames():
    global capturing
    yolo_model = YOLO('./runs/detect/Normal_Compressed/weights/best.pt')
    video_capture = cv2.VideoCapture(0)  # Use 0 for the default webcam
    video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))

    if not video_capture.isOpened():
        raise RuntimeError("Could not start webcam.")

    while capturing:
        success, frame = video_capture.read()
        if not success:
            break

        results = yolo_model(frame)
        for result in results:
            classes = result.names
            cls = result.boxes.cls
            conf = result.boxes.conf
            detections = result.boxes.xyxy

            for pos, detection in enumerate(detections):
                if conf[pos] >= 0.7:
                    xmin, ymin, xmax, ymax = detection
                    label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}" 
                    color = (0, int(cls[pos]), 255)
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                    cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

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

"""
from flask import Flask, Response, render_template, request, redirect, url_for, send_from_directory, flash
import os
import cv2
from ultralytics import YOLO
from character_ai import weapon_detected_c
from telegram_mes import send_telegram_photo
import threading
from detect import detect_objects_in_photo, detect_objects_in_video
import time
from concurrent.futures import ThreadPoolExecutor

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
        original_path, result_path, detected_weapons = detect_objects_in_photo(filepath)
        original_filename = os.path.basename(original_path)
        detected_filename = os.path.basename(result_path)
        if not detected_weapons:
            flash("No weapons are detected in this image.")
        return render_template('result.html', original_filename=original_filename, detected_filename=detected_filename, detected_weapons=detected_weapons)

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
                threading.Thread(target=weapon_detected_c, args=(classes[int(cls[pos])],)).start()

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
    

"""
# Initialize a lock for thread-safe updates
lock = threading.Lock()

# Cooldown time in seconds
COOLDOWN_TIME = 60
# Dictionary to keep track of last notification time and position for each detected weapon
last_notification = {}

# Thread pool for managing threads efficiently
executor = ThreadPoolExecutor(max_workers=4)

def is_within_cooldown(weapon, current_time, threshold=COOLDOWN_TIME):
    with lock:
        if weapon in last_notification:
            if (current_time - last_notification[weapon]['time']) < threshold:
                return True
    return False

def update_notification(weapon, coordinates, current_time):
    with lock:
        last_notification[weapon] = {'time': current_time, 'coordinates': coordinates}

def coordinates_within_range(coord1, coord2, threshold=50):
    # Check if two sets of coordinates are within a certain range
    x1, y1, x2, y2 = coord1
    a1, b1, a2, b2 = coord2
    return abs(x1 - a1) < threshold and abs(y1 - b1) < threshold and abs(x2 - a2) < threshold and abs(y2 - b2) < threshold

def process_frame(frame, yolo_model):
    results = yolo_model(frame)
    current_time = time.time()

    for result in results:
        classes = result.names
        cls = result.boxes.cls
        conf = result.boxes.conf
        detections = result.boxes.xyxy

        for pos, detection in enumerate(detections):
            if conf[pos] >= 0.7:
                xmin, ymin, xmax, ymax = detection
                weapon = classes[int(cls[pos])]
                coordinates = (int(xmin), int(ymin), int(xmax), int(ymax))

                # Check if the cooldown period has passed and the weapon has moved significantly
                if is_within_cooldown(weapon, current_time) and coordinates_within_range(last_notification[weapon]['coordinates'], coordinates):
                    continue  # Skip this detection if within cooldown and same position

                # Update the last notification time and coordinates for this weapon
                update_notification(weapon, coordinates, current_time)

                label = f"{weapon} {conf[pos]:.2f}"

                # Call weapon_detected_c and send_telegram_photo in separate threads using the thread pool
                executor.submit(weapon_detected_c, weapon)

                # Save the frame with the detected object
                photo_path = os.path.join(app.config['UPLOAD_FOLDER'], "detected_weapon.jpg")
                cv2.imwrite(photo_path, frame)
                executor.submit(send_telegram_photo, photo_path)

                color = (0, int(cls[pos]), 255)
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame

def gen_frames():
    yolo_model = YOLO('./runs/detect/Normal_Compressed/weights/best.pt')
    video_capture = cv2.VideoCapture(0)  # Use 0 for the default webcam
    video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    if not video_capture.isOpened():
        raise RuntimeError("Could not start webcam.")

    frame_count = 0
    process_every_n_frames = 3

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        frame_count += 1
        if frame_count % process_every_n_frames == 0:
            # Process frame in a separate thread
            frame = executor.submit(process_frame, frame, yolo_model).result()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video_capture.release()
"""
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
