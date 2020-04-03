import h5py
import re
import numpy as np
import time

def file_feed(path):
    frame_index_re = re.compile('\d+')
    with h5py.File(path) as in_file:
        key_names = [*map(str, in_file.keys())]
        key_names.sort(key=lambda x: int(frame_index_re.findall(x)[0]))
        for key_name in key_names:
            yield np.array(in_file[key_name]), time.time()

def stream_feed():
    while True:
        yield np.zeros((240, 320)), 0
