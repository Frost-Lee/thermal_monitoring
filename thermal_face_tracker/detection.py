import numpy as np
import insightface
import cv2

from . import config

detection_model = insightface.model_zoo.get_model('retinaface_r50_v1')
detection_model.prepare(ctx_id = -1, nms=0.4)

def get_face_detection(image):
    duplicated_image = np.array([image for _ in range(3)])
    duplicated_image = np.transpose(duplicated_image, axes=(1, 2, 0))
    bounding_boxes, landmarks = detection_model.detect(
        duplicated_image,
        threshold=config.FACE_DETECTION_THRESHOLD,
        scale=1.0
    )
    bounding_boxes = bounding_boxes.astype('int')
    landmarks = landmarks.astype('int')
    return bounding_boxes, landmarks

# def get_breath_detection_roi(image, bounding_box, landmark):
#     return image[
#         landmark[2, 1] : bounding_box[0, 3],
#         (landmark[3, 0] + bounding_box[0]) // 2 : (landmark[4, 0] + bounding_box[2]) // 2
#     ]

# def get_temperature_detection_roi(image, bounding_box, landmark):
#     return image[
#         bounding_box[1] : max(landmark[0, 1], landmark[1, 1]),
#         bounding_box[0] : bounding_box[2]
#     ]
