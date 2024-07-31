import cv2
from ultralytics import YOLO
from gradient_ai import weapon_detected_g
from character_ai import weapon_detected_c
from telegram_mes import send_telegram_photo
import os
import time

global out_text
out_text = ""
"""
1
def detect_objects_in_photo(image_path):
    image_orig = cv2.imread(image_path)
    
    yolo_model = YOLO('./runs/detect/Normal_Compressed/weights/best.pt')
    
    results = yolo_model(image_orig)
    
    for result in results:
        classes = result.names
        cls = result.boxes.cls
        conf = result.boxes.conf
        detections = result.boxes.xyxy

        for pos, detection in enumerate(detections):
            weapon_detected_c(classes[int(cls[pos])])
            if conf[pos] >= 0.2:
                xmin, ymin, xmax, ymax = detection
                label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}" 
                color = (0, int(cls[pos]), 255)
                cv2.rectangle(image_orig, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                cv2.putText(image_orig, label+out_text, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                #cv2.putText(image_orig, "there is a weapon", (200, 644), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    name = os.path.basename(image_path).split('.')[0]
    original_path = os.path.join("uploads", f"{name}_original.jpg")
    result_path = os.path.join("uploads", f"{name}_detected.jpg")

    # Save the original image to the original_path
    cv2.imwrite(original_path, cv2.imread(image_path))
    # Save the detected image to the result_path
    cv2.imwrite(result_path, image_orig)

    send_telegram_photo(result_path)
    return original_path, result_path
"""

"""2def detect_objects_in_photo(image_path):
    image_orig = cv2.imread(image_path)
    
    yolo_model = YOLO('./runs/detect/Normal_Compressed/weights/last.pt')
    
    results = yolo_model(image_orig)
    detected_weapons = []
    detections_found = False
    ai_response = ""

    for result in results:
        classes = result.names
        cls = result.boxes.cls
        conf = result.boxes.conf
        detections = result.boxes.xyxy

        if len(detections) != 0:
            weapon_result = weapon_detected_g(classes[int(cls[0])])
            detected_weapons.append(weapon_result)
            detections_found = True
            ai_response = weapon_result  # Store AI response

        for pos, detection in enumerate(detections):
            if conf[pos] >= 0.2:
                xmin, ymin, xmax, ymax = detection
                label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}" 
                color = (0, int(cls[pos]), 255)
                cv2.rectangle(image_orig, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                cv2.putText(image_orig, label+out_text, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    if not detections_found:
        print("No weapons are detected in this image")  # or use a logging mechanism

    name = os.path.basename(image_path).split('.')[0]
    original_path = os.path.join("uploads", f"{name}_original.jpg")
    result_path = os.path.join("uploads", f"{name}_detected.jpg")

    # Save the original image to the original_path
    cv2.imwrite(original_path, cv2.imread(image_path))
    # Save the detected image to the result_path
    cv2.imwrite(result_path, image_orig)

    return original_path, result_path, detected_weapons, ai_response 2"""
def detect_objects_in_photo(image_path, conf_threshold=0.2):
    image_orig = cv2.imread(image_path)
    
    yolo_model = YOLO('./runs/detect/Normal_Compressed/weights/last.pt')
    
    results = yolo_model(image_orig)
    detected_weapons = []
    detections_found = False
    ai_response = ""

    for result in results:
        classes = result.names
        cls = result.boxes.cls
        conf = result.boxes.conf
        detections = result.boxes.xyxy

        if len(detections) != 0:
            weapon_result = weapon_detected_g(classes[int(cls[0])])
            detected_weapons.append(weapon_result)
            detections_found = True
            ai_response = weapon_result  # Store AI response

        for pos, detection in enumerate(detections):
            if conf[pos] >= conf_threshold:
                xmin, ymin, xmax, ymax = detection
                label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}" 
                color = (0, int(cls[pos]), 255)
                cv2.rectangle(image_orig, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                cv2.putText(image_orig, label+out_text, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    if not detections_found:
        print("No weapons are detected in this image")  # or use a logging mechanism

    name = os.path.basename(image_path).split('.')[0]
    original_path = os.path.join("uploads", f"{name}_original.jpg")
    result_path = os.path.join("uploads", f"{name}_detected.jpg")

    # Save the original image to the original_path
    cv2.imwrite(original_path, cv2.imread(image_path))
    # Save the detected image to the result_path
    cv2.imwrite(result_path, image_orig)

    return original_path, result_path, detected_weapons, ai_response




"""
def detect_objects_in_video(video_path):
    yolo_model = YOLO('./runs/detect/Normal_Compressed/weights/best.pt')
    video_capture = cv2.VideoCapture(video_path)
    width = int(video_capture.get(3))
    height = int(video_capture.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    result_video_path = "detected_objects_video2.avi"
    out = cv2.VideoWriter(result_video_path, fourcc, 20.0, (width, height))

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        results = yolo_model(frame)

        for result in results:
            classes = result.names
            cls = result.boxes.cls
            conf = result.boxes.conf
            detections = result.boxes.xyxy

            for pos, detection in enumerate(detections):
                if conf[pos] > 0.63:
                    xmin, ymin, xmax, ymax = detection
                    label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}" 
                    color = (0, int(cls[pos]), 255)
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                    cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                    name = f"{classes[int(cls[pos])]}_{conf[pos]:.2f}"
                    cv2.imwrite(f'from-video/{name}.jpg', frame)

        out.write(frame)
    video_capture.release()
    out.release()

    return result_video_path
"""




def detect_objects_in_video(video_path, conf_threshold=0.63):
    yolo_model = YOLO('./runs/detect/Normal_Compressed/weights/best.pt')
    video_capture = cv2.VideoCapture(video_path)
    width = int(video_capture.get(3))
    height = int(video_capture.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    result_video_path = os.path.join("uploads", "detected_objects_video.avi")
    out = cv2.VideoWriter(result_video_path, fourcc, 20.0, (width, height))
    number = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        results = yolo_model(frame)

        for result in results:
            classes = result.names
            cls = result.boxes.cls
            conf = result.boxes.conf
            detections = result.boxes.xyxy

            for pos, detection in enumerate(detections):
                if conf[pos] > conf_threshold:
                    xmin, ymin, xmax, ymax = detection
                    label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}" 

                    color = (0, int(cls[pos]), 255)
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                    cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                    name = f"{classes[int(cls[pos])]}_{conf[pos]:.2f}"
                    cv2.imwrite(f'from-video/{name}.jpg', frame)
                    number += 1
                    if number % 15 == 0:
                        weapon_response = weapon_detected_g(classes[int(cls[pos])])
                        send_telegram_photo(f'from-video/{name}.jpg')

        out.write(frame)
    video_capture.release()
    out.release()

    return result_video_path

"""


# Cooldown time in seconds
COOLDOWN_TIME = 60

# Dictionary to keep track of last notification time for each weapon
last_notification_time = {}

def detect_objects_in_video(video_path):
    yolo_model = YOLO('./runs/detect/Normal_Compressed/weights/best.pt')
    video_capture = cv2.VideoCapture(video_path)
    width = int(video_capture.get(3))
    height = int(video_capture.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    result_video_path = os.path.join("uploads", "detected_objects_video.avi")
    out = cv2.VideoWriter(result_video_path, fourcc, 20.0, (width, height))

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        results = yolo_model(frame)

        for result in results:
            classes = result.names
            cls = result.boxes.cls
            conf = result.boxes.conf
            detections = result.boxes.xyxy

            for pos, detection in enumerate(detections):
                if conf[pos] > 0.63:
                    weapon = classes[int(cls[pos])]
                    current_time = time.time()

                    # Check if the cooldown period has passed for this weapon
                    if weapon not in last_notification_time or (current_time - last_notification_time[weapon]) > COOLDOWN_TIME:
                        xmin, ymin, xmax, ymax = detection
                        label = f"{weapon} {conf[pos]:.2f}"

                        # Call weapon_detected_c when a weapon is detected
                        weapon_response = weapon_detected_c(weapon)

                        # Save the frame with the detected object
                        photo_path = os.path.join("uploads", "detected_weapon.jpg")
                        cv2.imwrite(photo_path, frame)
                        send_telegram_photo(photo_path)

                        # Update the last notification time for this weapon
                        last_notification_time[weapon] = current_time

                        color = (0, int(cls[pos]), 255)
                        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                        cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                        time.sleep(10)

        out.write(frame)
    video_capture.release()
    out.release()

    return result_video_path
"""
def detect_objects_and_plot(path_orig):
    image_orig = cv2.imread(path_orig)
    
    yolo_model = YOLO('./runs/detect/Normal_Compressed/weights/best.pt')
    
    results = yolo_model(image_orig)

    for result in results:
        classes = result.names
        cls = result.boxes.cls
        conf = result.boxes.conf
        detections = result.boxes.xyxy

        for pos, detection in enumerate(detections):
            if conf[pos] >= 0.6:
                xmin, ymin, xmax, ymax = detection
                label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}" 
                color = (0, int(cls[pos]), 255)
                cv2.rectangle(image_orig, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                cv2.putText(image_orig, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    cv2.imshow("Teste", image_orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_objects_from_webcam():
    # Load the YOLO model
    yolo_model = YOLO('./runs/detect/Normal_Compressed/weights/best.pt')

    # Start capturing video from the webcam
    video_capture = cv2.VideoCapture(0)  # Use 0 for the default webcam
    video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            break

        # Perform object detection
        results = yolo_model(frame)

        # Process results
        for result in results:
            classes = result.names
            cls = result.boxes.cls
            conf = result.boxes.conf
            detections = result.boxes.xyxy

            for pos, detection in enumerate(detections):
                if conf[pos] >= 0.7:
                    xmin, ymin, xmax, ymax = detection
                    label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}" 
                    weapon_detected_c(classes[int(cls[pos])])

                    result, image = video_capture.read()
                    cv2.imwrite("teste.jpg", image)
                    send_telegram_photo("teste.jpg")

                    color = (0, int(cls[pos]), 255)
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                    cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Webcam Object Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    video_capture.release()
    cv2.destroyAllWindows()

# Run the function to start webcam object detection
#detect_objects_in_video("cropped.mp4")
#detect_objects_in_photo("5.jpg")