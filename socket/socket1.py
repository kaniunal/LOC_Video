import socket
import cv2
import pickle
import struct


# Create socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_ip = '192.168.181.38'  # Replace with the IP of Laptop 1
port = 9999

client_socket.connect((host_ip, port))
data = b""
payload_size = struct.calcsize("Q")

while True:
    # Keep receiving the data in chunks
    while len(data) < payload_size:
        packet = client_socket.recv(4*1024)  # Adjust buffer size as needed
        if not packet:
            break
        data += packet

    # Unpack the size of the message
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("Q", packed_msg_size)[0]

    # Receive the actual frame data
    while len(data) < msg_size:
        data += client_socket.recv(4*1024)

    frame_data = data[:msg_size]
    data = data[msg_size:]

    # Deserialize the frame
    frame = pickle.loads(frame_data)

    # Display the frame
    cv2.imshow("Receiving Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

client_socket.close()