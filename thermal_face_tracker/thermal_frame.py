import time
import numpy as np

from . import detection
from . import recognition
from . import thermal_face
from . import utils


class ThermalFrame(object):

    def __init__(self, thermal_frame):
        self.timestamp = time.time()
        self.thermal_frame = thermal_frame
        self.grey_frame = utils.rescale(thermal_frame)
        self.thermal_faces = []
        self._detect()
    
    def _detect(self):
        bounding_boxes, landmarks = detection.get_face_detection(self.grey_frame)
        self.thermal_faces = [thermal_face.ThermalFace(self, b, l) for b, l in zip(bounding_boxes, landmarks)]
    
    def link(self, previous_frame):
        matched_indices = []
        for face in self.thermal_faces:
            for index, previous_face in enumerate(previous_frame.thermal_faces):
                if index not in matched_indices and recognition.is_same_face(face,previous_face):
                    face.previous = previous_face
                    matched_indices.append(index)
                    break
    
    
