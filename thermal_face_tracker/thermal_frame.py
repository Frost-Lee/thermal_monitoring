import numpy as np
import scipy
import cv2

from . import detection
from . import recognition
from . import thermal_face
from . import utils


class ThermalFrame(object):

    def __init__(self, thermal_frame, timestamp):
        self.timestamp = timestamp
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
    
    def detach(self):
        for face in self.thermal_faces:
            face.previous = None
    
    def annotated_frame(self, annotate_temperature=True, annotate_breath_rate=True):
        annotated_frame = self.grey_frame
        for face in self.thermal_faces:
            cv2.rectangle(
                annotated_frame, 
                tuple(face.bounding_box[:2]), 
                tuple(face.bounding_box[2:]), 
                (255, 0, 0), 
                1
            )
            if annotate_temperature:
                temperature = face.temperature
                if temperature is not None:
                    cv2.putText(
                        annotated_frame,
                        str(temperature)[:5] + ' C',
                        tuple(face.bounding_box[:2]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        1
                    )
            if annotate_breath_rate:
                breath_rate = face.breath_rate
                if breath_rate is not None:
                    cv2.putText(
                        annotated_frame,
                        str(breath_rate)[:5] + 'Hz',
                        (face.bounding_box[0], face.bounding_box[3]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        1
                    )
        return annotated_frame
