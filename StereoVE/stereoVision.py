import sys
import cv2
import numpy as np
import time
import imutils
from matplotlib import pyplot as plt

# Function for stereo vision and depth estimation
import triangulation as tri
import calibration

# Open both cameras
cap_right = cv2.VideoCapture(2, cv2.CAP_DSHOW)
cap_left = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Stereo vision setup parameters
frame_rate = 120    # Camera frame rate (maximum at 120 fps)
B = 35              # Distance between the cameras [cm]
f = 8               # Camera lens's focal length [mm]
alpha = 49.9        # Camera field of view in the horizontal plane [degrees]

# Load YOLO
net = cv2.dnn.readNet("StereoVE/yolo_files/yolov3.weights", "StereoVE/yolo_files/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO names
with open("StereoVE/yolo_files/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Main program loop with people detection and depth estimation using stereo vision
while cap_right.isOpened() and cap_left.isOpened():

    succes_right, frame_right = cap_right.read()
    succes_left, frame_left = cap_left.read()

    ################## CALIBRATION #########################################################
    frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left)
    ########################################################################################

    # If cannot catch any frame, break
    if not succes_right or not succes_left:
        break

    else:
        start = time.time()

        # Convert the BGR image to RGB
        frame_right_rgb = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
        frame_left_rgb = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)

        # Perform people detection on the right and left frames
        def detect_people(frame):
            height, width, channels = frame.shape
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if class_id == 0 and confidence > 0.5:  # Class ID 0 is for 'person' in COCO dataset
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            return [(boxes[i], confidences[i]) for i in indexes]

        detections_right = detect_people(frame_right_rgb)
        detections_left = detect_people(frame_left_rgb)

        # Draw bounding boxes for detected people and calculate depth
        for (box_right, confidence_right) in detections_right:
            x_right, y_right, w_right, h_right = box_right
            center_right = (x_right + w_right // 2, y_right + h_right // 2)
            cv2.rectangle(frame_right, (x_right, y_right), (x_right + w_right, y_right + h_right), (0, 255, 0), 2)
            cv2.putText(frame_right, f"Person {confidence_right:.2f}", (x_right, y_right - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Find the corresponding detection in the left frame
            for (box_left, confidence_left) in detections_left:
                x_left, y_left, w_left, h_left = box_left
                center_left = (x_left + w_left // 2, y_left + h_left // 2)

                # Calculate the depth if the bounding boxes are close enough
                if abs(center_right[0] - center_left[0]) < w_right and abs(center_right[1] - center_left[1]) < h_right:
                    depth = tri.find_depth(center_right, center_left, frame_right, frame_left, B, f, alpha)
                    cv2.putText(frame_right, f"Depth: {depth:.2f} cm", (x_right, y_right + h_right + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(frame_left, f"Depth: {depth:.2f} cm", (x_left, y_left + h_left + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    break

        for (box_left, confidence_left) in detections_left:
            x_left, y_left, w_left, h_left = box_left
            cv2.rectangle(frame_left, (x_left, y_left), (x_left + w_left, y_left + h_left), (0, 255, 0), 2)
            cv2.putText(frame_left, f"Person {confidence_left:.2f}", (x_left, y_left - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frames with detected people and depth information
        cv2.imshow('Right Frame', frame_right)
        cv2.imshow('Left Frame', frame_left)

        # Wait for a key press and break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()