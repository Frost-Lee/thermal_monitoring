import time
import numpy as np

from . import detection
from . import recognition
from . import thermal_face


class ThermalFrame(object):

    def __init__(self, thermal_frame):
        self.timestamp = time.time()
        self.thermal_frame = thermal_frame
        self.grey_frame = ((thermal_frame - np.min(thermal_frame)) / (np.max(thermal_frame) - np.min(thermal_frame)) * 255).astype('uint8')
        self.thermal_faces = []
    
    def detect(self):
        bounding_boxes, landmarks = detection.get_face_detection(self.grey_frame)
        for bounding_box, landmark in zip(bounding_boxes, landmarks):
            self.thermal_faces.append(thermal_face.ThermalFace(bounding_box, landmark))
    
    # can be static method
    def get_face(self, image, bounding_box):
        return image[
            bounding_box[1] : bounding_box[3],
            bounding_box[0] : bounding_box[2]
        ]
    
    def link(self, previous_frame):
        matched_indices = []
        for face in self.thermal_faces:
            for index, previous_face in enumerate(previous_frame.thermal_faces):
                if index in matched_indices:
                    continue
                if recognition.is_same_person(
                    self.get_face(self.grey_frame, face.bounding_box),
                    self.get_face(previous_frame.grey_frame, previous_face.bounding_box)
                ):
                    face.previous = previous_face
                    matched_indices.append(index)
                    break
