import cv2
import numpy as np
from PIL import Image

from utils import get_limits_hsv


# Define  color in RGB format
color = [255, 0, 0]  # RGB for orange (BGR in OpenCV)

# Get HSV limits for orange color
lowerLimit, upperLimit = get_limits_hsv(color)

# =========================read webcam================================
cam = cv2.VideoCapture(0)

# =========================visualize webcam================================
while True:
    ret, frame = cam.read()

    if not ret:
        print("Failed to grab frame")
        break

    hsvImg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask using the defined HSV limits
    mask = cv2.inRange(hsvImg, lowerLimit, upperLimit)

    mask_ = Image.fromarray(mask)

    bbox=mask_.getbbox()

    if bbox is not None:
        x1,y1,x2,y2 = bbox

        frame=cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),5)

    cv2.imshow('webcam', frame)

    # Press 'q' to stop the video feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all OpenCV windows
cam.release()
cv2.destroyAllWindows()