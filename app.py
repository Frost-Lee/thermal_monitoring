import socket
import os
import numpy as np
from PIL import Image
import cv2

import thermal_face_tracker as tft
            
# HOST = ''
# PORT = 50007

# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.bind((HOST, PORT))
# s.listen(1)
# connection, address = s.accept()
# print('connected by', address)
# while 1:
#     data = connection.recv(1024)
#     if not data: break
#     connection.send('got data'.encode())
#     connection.sendall(data)
# connection.close()

frames = []
for root, dirs, files in os.walk('/Users/Frost/Desktop/sample_frames'):
    for file_name in files:
        frames.append(tft.thermal_frame.ThermalFrame(np.array(Image.open(os.path.join(root, file_name)))))
        frames[-1].detect()
        cv2.imshow('frame', frames[-1].grey_frame)
for index in range(1, len(frames)):
    frames[index].link(frames[index - 1])
face = frames[-1].thermal_faces[0]
while face is not None:
    print('1')
    face = face.previous
