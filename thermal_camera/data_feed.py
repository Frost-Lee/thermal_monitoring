import h5py
import re
import numpy as np
import time
import os

def file_feed(path):
    frame_index_re = re.compile('\d+')
    with h5py.File(path) as in_file:
        key_names = [*map(str, in_file.keys())]
        key_names.sort(key=lambda x: int(frame_index_re.findall(x)[0]))
        for key_name in key_names:
            yield np.array(in_file[key_name]), time.time()

def stream_feed(gige_cam_id):
    import matlab.engine
    matlab_engine = matlab.engine.start_matlab()
    matlab_engine.addpath(os.path.dirname(os.path.realpath(__file__)))
    gigecam = matlab_engine.init_gigecam(gige_cam_id)
    while True:
        yield np.array(matlab_engine.get_temperature(gigecam)), time.time()
