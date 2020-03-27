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

def is_same_person(face_image_1, face_image_2):
    embedding_1 = _get_embedding(face_image_1)
    embedding_2 = _get_embedding(face_image_2)
    return scipy.spatial.distance.cosine(embedding_1, embedding_2) < config.FACE_SIMILARITY_THRESHOLD
