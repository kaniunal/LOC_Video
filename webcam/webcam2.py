import socket
import cv2
import numpy as np
from PIL import Image


from utils import get_limits_hsv
# Server IP en poort
HOST = '0.0.0.0'
PORT = 9999

# Socket instellen
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
print("Wachten op verbinding...")

# Verbind met de client
conn, addr = server_socket.accept()
print(f"Verbonden met: {addr}")

# Zet het data buffer
data = b""
payload_size = 4  # voor frame grootte

# Define color in RGB format for red (BGR in OpenCV)
red = [0, 0, 255]  # RGB for red (BGR in OpenCV)





# Get HSV limits for red color
lowerLimit, upperLimit = get_limits_hsv(red)

while True:
    # Ontvang frame grootte
    while len(data) < payload_size:
        data += conn.recv(4096)

    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = int.from_bytes(packed_msg_size, "big")

    # Ontvang frame data
    while len(data) < msg_size:
        data += conn.recv(4096)

    frame_data = data[:msg_size]
    data = data[msg_size:]

    # Decodeer frame
    frame = np.frombuffer(frame_data, dtype=np.uint8)
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

    # Toon frame
    if frame is not None:
        hsvImg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # CreÃ«er een masker voor de rode kleur
        mask = cv2.inRange(hsvImg, lowerLimit, upperLimit)

        mask_ = Image.fromarray(mask)

        bbox = mask_.getbbox()

        if bbox is not None:
            x1, y1, x2, y2 = bbox

            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

        cv2.imshow("Video Stream", frame)

    if cv2.waitKey(1) == ord("q"):
        break

# Cleanup
conn.close()
server_socket.close()
cv2.destroyAllWindows()