import cv2
import numpy as np


# Function to get HSV limits for a given RGB color
def get_limits_hsv(color):
    c = np.uint8([[color]])
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    hue = hsv[0][0][0]

    lowerLimit = (hue - 10, 100, 100)
    upperLimit = (hue + 10, 255, 255)

    lowerLimit = np.array(lowerLimit, dtype=np.uint8)
    upperLimit = np.array(upperLimit, dtype=np.uint8)

    return lowerLimit, upperLimit