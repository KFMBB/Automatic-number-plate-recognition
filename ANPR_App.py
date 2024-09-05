import streamlit as st
import cv2
import numpy as np
import tempfile
import shutil
from ultralytics import YOLO
from sort.sort import Sort
from helper_util import assign_car, read_license_plate, write_csv
import easyocr
from PIL import Image

# Load models
vehicle_model = YOLO('yolov8n.pt')  # YOLOv8 for vehicle detection
license_plate_detector = YOLO('Plate_detector_Model/license_plate_detector.pt')  # License plate detection
mot_tracker = Sort()  # SORT tracker for vehicle tracking
ocr_reader = easyocr.Reader(['en'])  # EasyOCR for license plate recognition

# Function to process frames (for both image and video)
def process_frame(frame, frame_c):
    results = {}
    results[frame_c] = {}

    # Vehicle detection
    detections = vehicle_model(frame)[0]
    vehicles_detected = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in [2, 3, 5, 7]:  # Vehicle classes
            vehicles_detected.append([x1, y1, x2, y2, score])
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # Annotate vehicle

    # Track vehicles
    track_ids = mot_tracker.update(np.asarray(vehicles_detected))

    # License plate detection
    license_plates = license_plate_detector(frame)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1_lp, y1_lp, x2_lp, y2_lp, score_lp, class_id_lp = license_plate

        # Assign license plate to the vehicle
        x1_v, y1_v, x2_v, y2_v, car_id = assign_car(license_plate, track_ids)

        if car_id != -1:
            # Annotate license plate
            cv2.rectangle(frame, (int(x1_lp), int(y1_lp)), (int(x2_lp), int(y2_lp)), (0, 255, 0), 2)

            # Process license plate for OCR
            license_plate_crop = frame[int(y1_lp):int(y2_lp), int(x1_lp):int(x2_lp)]
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

            # OCR
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

            # Annotate the frame with OCR result
            cv2.putText(frame, license_plate_text, (int(x1_lp), int(y1_lp) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            # Save results
            if license_plate_text:
                results[frame_c][car_id] = {'car': {'bbox': [x1_v, y1_v, x2_v, y2_v]},
                                            'license_plate': {'bbox': [x1_lp, y1_lp, x2_lp, y2_lp],
                                                              'text': license_plate_text,
                                                              'bbox_score': score_lp,
                                                              'text_score': license_plate_text_score}}
    return frame, results

# Streamlit App
st.title("Automatic Number Plate Recognition (ANPR) System")
st.write("Upload an image or video for license plate detection and recognition.")

uploaded_file = st.file_uploader("Choose an image or video", type=["jpg", "png", "mp4"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    uploaded_file.seek(0)  # Reset file pointer to the beginning

    if uploaded_file.type == "video/mp4":
        # Save uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        # Video processing
        st.write("Processing video, please wait...")
        with st.spinner('Processing video...'):
            cap = cv2.VideoCapture(temp_file_path)
            frame_c = 0
            results = {}
            frames_annotated = []
            frames_original = []

            while True:
                frame_c += 1
                isReadingFrames, frame = cap.read()

                if not isReadingFrames:
                    break

                # Process frame
                processed_frame, frame_results = process_frame(frame, frame_c)
                results.update(frame_results)

                # Convert frames to RGB for Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

                # Append frames for final display
                frames_original.append(frame_rgb)
                frames_annotated.append(processed_frame_rgb)

            cap.release()

            write_csv(results, 'Results_video.csv')

            # Save annotated video
            annotated_video_path = 'annotated_video.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(annotated_video_path, fourcc, 20.0, (frames_annotated[0].shape[1], frames_annotated[0].shape[0]))

            for frame in frames_annotated:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            out.release()

            # Display original and annotated videos
            st.write("Original Video:")
            st.video(temp_file_path)
            st.write("Annotated Video:")
            st.video(annotated_video_path)

    else:
        # Image processing
        st.write("Processing image, please wait...")
        with st.spinner('Processing image...'):
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            frame_c = 0
            processed_frame, results = process_frame(image, frame_c)

            # Display the processed image
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            st.image(processed_frame_rgb, caption='Processed Image')

            write_csv(results, 'Results_image.csv')
