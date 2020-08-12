import numpy as np
from scipy import optimize
import cv2
from deprecated import deprecated

from . import detection
from . import thermal_face
from . import utils
from . import config


class ThermalFrame(object):
    """ An object that represents a frame that the thermal camera captures.

    Attributes:
        timestamp: The timestamp when the frame is captured. In seconds.
        thermal_frame: The numpy array that represents the raw frame. Each element 
            stands for the celsius temperature of the corresponding pixel.
        grey_frame: The numpy array that is rescaled from `thermal_frame` to 0-255.
        thermal_faces: The `thermal_face.ThermalFace` objects that is detected on 
            the current frame.
    """

    def __init__(self, thermal_frame, timestamp):
        self.timestamp = timestamp
        self.thermal_frame = thermal_frame
        self.grey_frame = utils.rescale(thermal_frame)
        self.thermal_faces = []
        self._detect()

    def _detect(self):
        """ Detect all face entities in this frame.
        """
        bounding_boxes, landmarks = detection.get_face_detection(self.grey_frame)
        self.thermal_faces = [thermal_face.ThermalFace(self, b, l) for b, l in zip(bounding_boxes, landmarks)]

    def link(self, previous_frame):
        """ Link the face entities of this frame with the face entities in the 
            previous frame if they stands for the same face.

        Args:
            previous_frame: The frame to be linked with. It should be the previous 
                frame of this frame.
        """
        if len(self.thermal_faces) == 0 or len(previous_frame.thermal_faces) == 0:
            return
        similarity_matrix = np.zeros((len(self.thermal_faces), len(previous_frame.thermal_faces)))
        for i, face in enumerate(self.thermal_faces):
            for j, previous_face in enumerate(previous_frame.thermal_faces):
                similarity_matrix[i, j] = face.similarity(previous_face)
        cost_matrix = 1.0 - similarity_matrix
        for i, j in zip(*optimize.linear_sum_assignment(cost_matrix)):
            if similarity_matrix[i, j] > config.FACE_LINK_THRESHOLD:
                self.thermal_faces[i].previous = previous_frame.thermal_faces[j]
                self.thermal_faces[i].uuid = previous_frame.thermal_faces[j].uuid

    def detach(self):
        """ Detach the face entity links with the previous frame.
        """
        for face in self.thermal_faces:
            face.previous = None

    @deprecated(reason='annotated_frame is deprecated, use visualizer for visualization instead.')
    def annotated_frame(self, annotate_temperature=True, annotate_breath_rate=True):
        """ Returns a grey frame with annotation, used for visualization.

        Args:
            annotate_temperature: Whether the body temperature should be annotated 
                on the returned frame.
            annotate_breath_rate: Whether the breath rate should be annotated on 
                the returned frame.
        """
        annotated_frame = cv2.UMat(self.grey_frame)
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
