import numpy as np
import cv2
import time

import thermal_face_tracker as tft
import thermal_camera as tc

MAX_CACHED_FRAMES = 128
frame_cache = []

for raw_frame, timestamp in tc.data_feed.file_feed('/Users/Frost/Desktop/record.hdf5'):
    thermal_frame = tft.thermal_frame.ThermalFrame(raw_frame, timestamp)
    if len(frame_cache) > 0:
        thermal_frame.link(frame_cache[-1])
    if len(frame_cache) >= MAX_CACHED_FRAMES:
        frame_cache.pop(0)
        frame_cache[0].detach()
        frame_cache.append(thermal_frame)
    frame_cache.append(thermal_frame)
    cv2.imshow('frame', thermal_frame.annotated_frame())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
