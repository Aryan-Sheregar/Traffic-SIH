import cv2
import json
from ultralytics import YOLO
import torch
import tensorflow as tflite
import numpy as np
from flask import Flask, request, jsonify
import os

# Initialize Flask app
app = Flask(__name__)

# Load YOLOv8 model with lower precision on GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('yolov8n.pt').to(device)  # Using 'yolov8n.pt' for faster inference

# Define vehicle classes according to COCO dataset
VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}  # COCO class IDs for vehicles

# Load TFLite model for accident detection using TensorFlow Lite Interpreter
interpreter = tflite.lite.Interpreter(model_path='/Users/aryansheregar/Downloads/CNN Accident Detection Model.tflite')
interpreter.allocate_tensors()

# Get input and output tensors from the accident detection model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Base timings (in seconds) for each state
BASE_RED_TIME = 20
BASE_YELLOW_TIME = 5
BASE_GREEN_TIME = 15

# Thresholds for traffic density-based timing (these can be tuned)
LOW_TRAFFIC_THRESHOLD = 5
MEDIUM_TRAFFIC_THRESHOLD = 10

def detect_accident(frame):
    """
    Run accident detection on the given frame using the TFLite model.
    Returns 1 if an accident is detected, otherwise 0.
    """
    input_shape = input_details[0]['shape']
    resized_frame = cv2.resize(frame, (input_shape[2], input_shape[1]))
    input_data = np.expand_dims(resized_frame, axis=0).astype(np.float32)

    # Set tensor data and invoke model
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get prediction result
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return int(output_data[0][0])  # Assuming binary output: 0 (no accident), 1 (accident)

def calculate_timings(car_count):
    """
    Calculate the signal timings based on the number of detected cars.
    """
    if car_count > MEDIUM_TRAFFIC_THRESHOLD:
        red_time = BASE_RED_TIME
        yellow_time = BASE_YELLOW_TIME
        green_time = BASE_GREEN_TIME + 10  # Increase green time for heavy traffic
    elif car_count > LOW_TRAFFIC_THRESHOLD:
        red_time = BASE_RED_TIME
        yellow_time = BASE_YELLOW_TIME
        green_time = BASE_GREEN_TIME
    else:
        red_time = BASE_RED_TIME + 10  # Increase red time for low traffic
        yellow_time = BASE_YELLOW_TIME
        green_time = BASE_GREEN_TIME - 5  # Decrease green time for low traffic

    return red_time, yellow_time, green_time

@app.route('/process_video', methods=['GET', 'POST'])
def process_video():
    if request.method == 'GET':
        return "Use POST request to submit the video."

    # Ensure the 'uploads' directory exists
    upload_folder = './uploads/'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    # Handle POST request as usual
    video_file = request.files['video']
    video_path = os.path.join(upload_folder, video_file.filename)
    video_file.save(video_path)

    num_intersections = int(request.form.get('num_intersections', 1))

    # Process video and generate JSON response
    response_data = traffic_light_controller(video_path, num_intersections)
    return jsonify(response_data)

def traffic_light_controller(video_path, num_intersections):
    """
    Control traffic lights based on car count from a video file, detect accidents, and save output to a JSON file.
    """
    # Open video capture from the specified .mp4 file
    cap = cv2.VideoCapture(video_path)

    # Get the frame width, height, and FPS from the input video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the interval for frame processing (every 1 second)
    process_interval = int(fps)  # Number of frames to skip (1 second interval)

    # Initialize timers and states for each traffic light
    traffic_light_states = ['Red'] * num_intersections  # Start with Red light for all
    traffic_light_timers = [BASE_RED_TIME] * num_intersections

    frame_count = 0
    json_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame at 1-second intervals
        if frame_count % process_interval == 0:
            # Split the frame into regions (one per intersection)
            regions = [frame[int((height // num_intersections) * i):int((height // num_intersections) * (i + 1)), :] for i in range(num_intersections)]

            # Initialize a list to store car counts for each intersection
            car_counts = []
            accident_detected = 0  # Set accident_detected flag to 0 initially

            for region in regions:
                # Perform object detection on each region
                results = model(region, conf=0.2)
                detections = results[0].boxes

                # Filter to keep only vehicle classes
                vehicle_detections = [d for d in detections if int(d.cls) in VEHICLE_CLASSES]

                # Count the number of detected vehicles in this region
                car_count = len(vehicle_detections)
                car_counts.append(car_count)

                # Run accident detection
                if detect_accident(region) == 1:
                    accident_detected = 1  # Set flag to 1 if accident is detected

            # Update timings based on car counts for each intersection
            timings = [calculate_timings(count) for count in car_counts]

            # Update traffic light states and timers
            for i in range(num_intersections):
                if traffic_light_timers[i] > 0:
                    traffic_light_timers[i] -= 1
                else:
                    if traffic_light_states[i] == 'Red':
                        traffic_light_states[i] = 'Green'
                        traffic_light_timers[i] = timings[i][2]  # Green time
                    elif traffic_light_states[i] == 'Green':
                        traffic_light_states[i] = 'Yellow'
                        traffic_light_timers[i] = timings[i][1]  # Yellow time
                    elif traffic_light_states[i] == 'Yellow':
                        traffic_light_states[i] = 'Red'
                        traffic_light_timers[i] = timings[i][0]  # Red time

            # Collect data for JSON output
            frame_data = {
                "frame_count": frame_count,
                "intersections": [],
                "accident_detected": accident_detected  # 1 if accident detected, 0 otherwise
            }
            for i in range(num_intersections):
                intersection_data = {
                    "intersection_id": i + 1,
                    "vehicle_count": car_counts[i],
                    "traffic_light_state": traffic_light_states[i],
                    "timer": traffic_light_timers[i]
                }
                frame_data["intersections"].append(intersection_data)

            json_data.append(frame_data)

        frame_count += 1

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # Save the collected data to a file
    with open('output.json', 'w') as outfile:
        json.dump(json_data, outfile, indent=4)

    # Return the collected data
    return json_data


if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=8080)