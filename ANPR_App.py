import streamlit as st
import cv2
import numpy as np
import pandas as pd
import ast
from ultralytics import YOLO
from io import BytesIO
import tempfile
import os

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

def assign_car(lp, vehicles_detected):
    for vehicle in vehicles_detected:
        if lp[0] > vehicle[0] and lp[1] > vehicle[1] and lp[2] < vehicle[2] and lp[3] < vehicle[3]:
            return vehicle
    return -1

def read_license_plate(crop):
    # Placeholder for actual OCR code
    # Use EasyOCR or any other OCR tool here
    return "ABC123", 0.9

def write_csv(results, file_path):
    # Write results to CSV
    with open(file_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['frame_c', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_text'])
        writer.writeheader()
        for frame, data in results.items():
            for car_id, details in data.items():
                writer.writerow({
                    'frame_c': frame,
                    'car_id': car_id,
                    'car_bbox': details['car']['bbox'],
                    'license_plate_bbox': details['license_plate']['bbox'],
                    'license_plate_text': details['license_plate']['text']
                })

# Streamlit App
st.title('License Plate Detection and OCR')
st.write("This application detects vehicles, tracks them, and reads license plates.")

uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(uploaded_video.read())
        temp_file_path = temp_file.name

    # Load models
    model = YOLO('models/yolov8n.pt')  # Vehicle detection
    license_plate_detector = YOLO('models/best_license_plate_detector.pt')  # License plate detection

    # Open video file
    cap = cv2.VideoCapture(temp_file_path)

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
    results = pd.read_csv('output/results.csv')

    # load video
    cap = cv2.VideoCapture(temp_file_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('output/test_annot.mp4', fourcc, fps, (width, height))

    license_plate = {}
    for car_id in np.unique(results['car_id']):
        max_ = np.amax(results[results['car_id'] == car_id]['license_plate_bbox'])
        license_plate[car_id] = {'license_crop': None,
                                 'license_plate_number': results[(results['car_id'] == car_id) &
                                                                 (results['license_plate_bbox'] == max_)]['license_plate_text'].iloc[0]}
        cap.set(cv2.CAP_PROP_POS_FRAMES, results[(results['car_id'] == car_id) &
                                                 (results['license_plate_bbox'] == max_)]['frame_c'].iloc[0])
        ret, frame = cap.read()

        x1, y1, x2, y2 = ast.literal_eval(results[(results['car_id'] == car_id) &
                                                  (results['license_plate_bbox'] == max_)]['license_plate_bbox'].iloc[0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))

        license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
        license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

        license_plate[car_id]['license_crop'] = license_crop

    frame_nmr = -1

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # read frames
    ret = True
    while ret:
        ret, frame = cap.read()
        frame_nmr += 1
        if ret:
            df_ = results[results['frame_c'] == frame_nmr]
            for row_indx in range(len(df_)):
                # draw car
                car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25,
                            line_length_x=200, line_length_y=200)

                # draw license plate
                x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

                # crop license plate
                license_crop = license_plate[df_.iloc[row_indx]['car_id']]['license_crop']

                H, W, _ = license_crop.shape

                try:
                    frame[int(car_y1) - H - 100:int(car_y1) - 100, int(car_x1):int(car_x1) + W] = license_crop
                    frame[int(car_y1) - H - 100:int(car_y1) - 100, int(car_x1):int(car_x1) + W, :] = (255, 255, 255)

                    (text_width, text_height), _ = cv2.getTextSize(
                        license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                        cv2.FONT_HERSHEY_SIMPLEX,
                        4.3,
                        17)

                    cv2.putText(frame,
                                license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                                (int((car_x2 + car_x1 - text_width) / 2), int(car_y1 - H - 250 + (text_height / 2))),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                4.3,
                                (0, 0, 0),
                                17)

                except:
                    pass

            out.write(frame)
            frame = cv2.resize(frame, (1280, 720))

    out.release()
    cap.release()
    os.remove(temp_file_path)

    st.video('output/test_annot.mp4')
