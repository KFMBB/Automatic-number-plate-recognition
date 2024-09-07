from ultralytics import YOLO
import cv2
# The steps of the project:
# 1- We'll use YOLO for license plate detection (object detection).
# 2- We'll use SORT model to keep tracking our object on the road (object tracking).
# 3- Lastly using easy OCR we'll read the license plate contents.
import helper_util
from sort.sort import *
from helper_util import assign_car, read_license_plate, write_csv

results = {}  # We'll save our results for processing later.

mot_tracker = Sort()  # For future work we can try out bytetracker and good for us it's already installed on YOLO.

# load models
# We're using two pretrained models:
# 1- The first model will focus on detecting cars while the other will focus on detgt ecting license plates:
model = YOLO('yolov8n.pt')  # Load the nano version of the yolo model.
# 2- The second model will be used to detect the license plates.
# license_plate_detector = YOLO('Plate_detector_Model/plate_dedector.pt')
license_plate_detector = YOLO('Plate_detector_Model/license_plate_detector.pt')
# load video
# cap = cv2.VideoCapture('Readings/License_Plate_detection.mp4')
cap = cv2.VideoCapture("Readings/License_Plate_detection.mp4") # try a smaller vid
# Since we're detecting vehicles we'll set a list with the YOLO class_id for vehicles.
# why? well to make our model less computationally expensive and make it focus on the task at hand
vehicles = [2, 3, 5, 7]

# read frames
frame_c = -1  # We'll assign the frame counter as a unique key for our results.
threshold = 64  # Variable we'll use when using cv2 thresholding.
isReadingFrames = True  # This is a boolean flag that keeps track of whether the frame was read or not.
while isReadingFrames:
    frame_c += 1
    isReadingFrames, frame = cap.read()
    if isReadingFrames:
        results[frame_c] = {}
        # Now apply each model to the current frame:
        # vehicle detection and tracking:
        detections = model.track(frame, persist = True, tracker="bytetrack.yaml")[0]  # the [0] will unpack the models bbox, confidence and class_id as a tuple.
        vehicles_detected = []  # Store each vehicle detected in the list.
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, track_id, score, class_id = detection
            if int(class_id) in vehicles:  # If the object detected is a vehicle:
                vehicles_detected.append([x1, y1, x2, y2, track_id, score])  # Add bbox, confidence to the list of detected vehicles.

        # track vehicles using SORT:
        # track_ids = mot_tracker.update(np.asarray(vehicles_detected))

        # detect the license plates:
        license_plates = license_plate_detector(frame)[0]  # same as the vehicles detections.
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Now that we have detected cars and license plates, we'll assign license plates to cars:
            x1_v, y1_v, x2_v, y2_v, car_id = assign_car(license_plate, vehicles_detected)

            if car_id != -1:  # If the car was assigned to a license plate successfully go to the processing pipeline

                # crop license plate:
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                # we essentially took the bbox cords and then processed the whole frame, we took slices from y1 till
                # y2 and then x1 till x2 and lastly the rgb or bgr.

                # process license plate number: why? well to make it easier for the ocr model to process, we'll grey
                # scale the license plate as colors don't really help the model with reading a plate.
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                # apply threshold: If the pixel value is less than threshold it becomes white (255),
                # else it becomes black that's why we call it binary thresholding.
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, threshold, 255, cv2.THRESH_BINARY_INV)

                # Now that we have processed our license plate image we'll read it using easy ocr:
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                # Save the results:
                if license_plate_text is not None:
                    results[frame_c][car_id] = {'car': {'bbox': [x1_v, y1_v, x2_v, y2_v]},
                                                'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                  'text': license_plate_text,
                                                                  'bbox_score': score,
                                                                  'text_score': license_plate_text_score}}

# write results
write_csv(results, 'Readings/Results.csv')
