import numpy as np


def rescale(grey_scale_image):
    numerical_range = np.max(grey_scale_image) - np.min(grey_scale_image)
    relative_values = grey_scale_image - np.min(grey_scale_image)
    return (relative_values / numerical_range * 255).astype('uint8')


def crop(image, bounding_box):
    height, width = tuple(image.shape[:2])
    bounding_box[0] = bounding_box[0] if bounding_box[0] >= 0 else 0
    bounding_box[1] = bounding_box[1] if bounding_box[1] >= 0 else 0
    bounding_box[2] = bounding_box[2] if bounding_box[2] < width else width - 1
    bounding_box[3] = bounding_box[3] if bounding_box[3] < height else height - 1
    return image[
        bounding_box[1]: bounding_box[3],
        bounding_box[0]: bounding_box[2]
    ]


def uuid_to_color(uuid, mode='rgb', ub=255):
    assert mode == 'rgb' or mode == 'bgr'
    assert ub == 255 or ub == 1
    unit = 4
    hash_list = [*map(lambda x: hash(uuid[x: x + unit]) % 255, [1, 2, 3])]
    if ub == 1:
        hash_list = [*map(lambda x: x / 255, hash_list)]
    if mode == 'bgr':
        hash_list = [*reversed(hash_list)]
    return tuple(hash_list)
