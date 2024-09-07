import streamlit as st
import cv2
from ultralytics import YOLO
import pandas as pd
import numpy as np
import csv
from scipy.interpolate import interp1d
import Helper_util
from Helper_util import assign_car, read_license_plate, write_csv
import ast

# Function to draw borders around detected objects
def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right
    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  # top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  # bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)
    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  # top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  # bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)
    return img

# Helper function to interpolate bounding boxes
def interpolate_bounding_boxes(data):
    frame_numbers = np.array([int(row['frame_c']) for row in data])
    car_ids = np.array([int(float(row['car_id'])) for row in data])
    car_bboxes = np.array([list(map(float, row['car_bbox'][1:-1].split())) for row in data])
    license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'][1:-1].split())) for row in data])

    interpolated_data = []
    unique_car_ids = np.unique(car_ids)
    for car_id in unique_car_ids:
        car_mask = car_ids == car_id
        car_frame_numbers = frame_numbers[car_mask]
        car_bboxes_interpolated = []
        license_plate_bboxes_interpolated = []

        for i in range(len(car_bboxes[car_mask])):
            frame_number = car_frame_numbers[i]
            car_bbox = car_bboxes[car_mask][i]
            license_plate_bbox = license_plate_bboxes[car_mask][i]

            car_bboxes_interpolated.append(car_bbox)
            license_plate_bboxes_interpolated.append(license_plate_bbox)

        for i in range(len(car_bboxes_interpolated)):
            frame_number = car_frame_numbers[i]
            row = {}
            row['frame_c'] = str(frame_number)
            row['car_id'] = str(car_id)
            row['car_bbox'] = ' '.join(map(str, car_bboxes_interpolated[i]))
            row['license_plate_bbox'] = ' '.join(map(str, license_plate_bboxes_interpolated[i]))

            interpolated_data.append(row)

    return interpolated_data

# Main app
st.title('License Plate Detection and OCR')

st.write("This application detects vehicles, tracks them, and reads license plates.")

uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Load models
    model = YOLO('models/yolov8n.pt')  # Vehicle detection
    license_plate_detector = YOLO('models/best_license_plate_detector.pt')  # License plate detection

    cap = cv2.VideoCapture(uploaded_video)

    frame_c = -1
    threshold = 64
    results = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_c += 1
        results[frame_c] = {}

        # Detect vehicles
        detections = model.track(frame, persist=True, tracker="bytetrack.yaml")[0]
        vehicles_detected = [d for d in detections.boxes.data.tolist() if int(d[5]) in [2, 3, 5, 7]]

        # Detect license plates
        license_plates = license_plate_detector(frame)[0]
        for lp in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = lp
            x1_v, y1_v, x2_v, y2_v, car_id = assign_car(lp, vehicles_detected)

            if car_id != -1:
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, threshold, 255, cv2.THRESH_BINARY_INV)
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text:
                    results[frame_c][car_id] = {'car': {'bbox': [x1_v, y1_v, x2_v, y2_v]},
                                                'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                  'text': license_plate_text,
                                                                  'bbox_score': score,
                                                                  'text_score': license_plate_text_score}}

    # Writing to CSV
    write_csv(results, 'output/results.csv')
    st.success("Detection complete. Results saved.")

    # Interpolation and video annotation
    with open('output/results.csv', 'r') as file:
        reader = csv.DictReader(file)
        data = list(reader)

    interpolated_data = interpolate_bounding_boxes(data)
    pd.DataFrame(interpolated_data).to_csv('output/processed_results.csv', index=False)

    # Annotating video
    license_plate = {}
    for car_id in np.unique([d['car_id'] for d in interpolated_data]):
        frame_c = int(max([int(d['frame_c']) for d in interpolated_data if d['car_id'] == car_id]))
        st.write(f"Car ID: {car_id}, Frame: {frame_c}")

    st.video("output/test_annot.mp4")
