import numpy as np
import insightface
import cv2
import scipy

from . import config

embedding_model = insightface.model_zoo.get_model('arcface_r100_v1')
embedding_model.prepare(ctx_id = -1)

def _get_embedding(face_image):
    reshaped_image = cv2.resize(face_image, (112, 112))
    return embedding_model.get_embedding(reshaped_image)[0]

def is_same_face(thermal_face_1, thermal_face_2):
    bounding_box_1 = thermal_face_1.bounding_box
    bounding_box_2 = thermal_face_2.bounding_box
    mid_x = (bounding_box_1[1] + bounding_box_1[3]) / 2
    mid_y = (bounding_box_1[0] + bounding_box_1[2]) / 2
    x_overlap = bounding_box_2[1] <= mid_x <= bounding_box_2[3]
    y_overlap = bounding_box_2[0] <= mid_y <= bounding_box_2[2]
    return x_overlap and y_overlap
