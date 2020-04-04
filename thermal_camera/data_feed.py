import h5py
import numpy as np
import time
import os

def file_feed(path):
    """ Returns an iterator of the frames and timestamps from a recording file.

    Args:
        path: The path of the recording file.
    """
    with h5py.File(path) as in_file:
        key_names = [*map(str, in_file.keys())]
        key_names.sort(key=lambda x: int(x.split('_')[-1]))
        for key_name in key_names:
            yield np.array(in_file['{}/raw_frame'.format(key_name)]), np.array(in_file['{}/timestamp'.format(key_name)])[0]

def stream_feed(gige_camera_id):
    """ Returns an iterator of the frames and timestamps from GIGE thermal camera.

    Args:
        gige_camera_id: The id of the GIGE thermal camera that streams the input.
    """
    import matlab.engine
    matlab_engine = matlab.engine.start_matlab()
    matlab_engine.addpath(os.path.dirname(os.path.realpath(__file__)))
    gigecam = matlab_engine.init_gigecam(gige_camera_id)
    while True:
        yield np.array(matlab_engine.get_temperature(gigecam)), time.time()
