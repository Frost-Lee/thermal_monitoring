import numpy as np
import cv2
import time

import thermal_face_tracker as tft
import thermal_camera as tc

MAX_CACHED_FRAMES = 64
frame_cache = []

for raw_frame in tc.data_feed.file_feed('/Users/Frost/Desktop/thermal_data/mask_on_3.hdf5'):
    thermal_frame = tft.thermal_frame.ThermalFrame(raw_frame)
    cv2.imshow('frame', thermal_frame.grey_frame)
    if len(frame_cache) > 0:
        thermal_frame.link(frame_cache[-1])
    frame_cache = frame_cache[0 if len(frame_cache) < MAX_CACHED_FRAMES else 1:] + [thermal_frame]
    for index, face in enumerate(thermal_frame.thermal_faces):
        frame_count = 1
        mean_cache = [np.mean(face.breath_roi)]
        while face.previous is not None:
            face = face.previous
            frame_count += 1
            mean_cache.append(np.mean(face.breath_roi))
        if frame_count > 35:
            np.save('/Users/Frost/Desktop/a.npy', np.array(mean_cache))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
