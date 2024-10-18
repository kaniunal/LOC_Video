import socket
import cv2
import pickle
import struct
#Socket creation
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_ip = '192.168.181.38'  # Server will bind to all interfaces on Laptop 1
port = 9999

socket_address = (host_ip, port)
server_socket.bind(socket_address)
server_socket.listen(5)
print("Server Listening at:", socket_address)

#The IP address of Laptop 2 (the only allowed client)
allowed_ip = '192.168.181.191'  # Replace with Laptop 2's actual IP address

while True:
    # Accept connection from client
    client_socket, client_addr = server_socket.accept()
    print('Connection request from:', client_addr)

    # Restrict access to only the allowed IP address
    if client_addr[0] == allowed_ip:
        print(f"Connection accepted from {client_addr}")
    #Initialize camera
        camera = cv2.VideoCapture(0)

        while camera.isOpened():
            ret, frame = camera.read()
            if not ret:
                break

            # Serialize the frame
            frame_data = pickle.dumps(frame)
            # Send message length first
            message = struct.pack("Q", len(frame_data)) + frame_data
            client_socket.sendall(message)

        camera.release()
        client_socket.close()
    else:
        print(f"Connection refused from {client_addr}")
        client_socket.close()  # Close the connection if the IP is not allowed