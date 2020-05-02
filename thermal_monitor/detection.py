import numpy as np
import insightface
import cv2

from . import config

detection_model = insightface.model_zoo.get_model('retinaface_r50_v1')
detection_model.prepare(ctx_id=config.GPU_ID, nms=0.4)

def get_face_detection(image):
    """ Get face detection result from given image.

    Args:
        image: An numpy array with shape `(height, width, 3)`.
    
    Returns:
        The bounding boxes and landmarks of the detected faces.
    """
    duplicated_image = np.stack([image] * 3, 2)
    bounding_boxes, landmarks = detection_model.detect(
        duplicated_image,
        threshold=config.FACE_DETECTION_THRESHOLD,
        scale=1.0
    )
    return bounding_boxes.astype(int)[:, :-1], landmarks.astype(int)
