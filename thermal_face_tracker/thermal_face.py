import numpy as np
from scipy import interpolate

from . import utils
from . import config


class ThermalFace(object):

    def __init__(self, parent, bounding_box, landmark):
        self.parent = parent
        self.bounding_box = bounding_box
        self.landmark = landmark
        self.embedding = None
        self.previous = None
    
    @property
    def timestamp(self):
        return self.parent.timestamp
    
    @property
    def thermal_image(self):
        return utils.crop(self.parent.thermal_frame, self.bounding_box)
    
    @property
    def grey_image(self):
        return utils.crop(self.parent.grey_frame, self.bounding_box)
    
    @property
    def temperature_roi(self):
        return utils.crop(self.parent.thermal_frame, self.bounding_box)
    
    @property
    def breath_roi(self):
        return self.parent.thermal_frame[
            self.landmark[2, 1] : self.bounding_box[3],
            (self.landmark[3, 0] + self.bounding_box[0]) // 2 : (self.landmark[4, 0] + self.bounding_box[2]) // 2
        ]
    
    @property
    def temperature(self):
        try:
            return np.max(self.temperature_roi)
        except ValueError:
            return None
    
    @property
    def breath_rate(self):
        root = self
        timestamps, samples = [], []
        while root is not None:
            timestamps = [root.timestamp] + timestamps
            samples = [np.mean(root.breath_roi)] + samples
            root = root.previous
        if len(timestamps) < config.BREATH_RATE_MIN_SAMPLE_THRESHOLD:
            return None
        timestamps, samples = np.array(timestamps), np.array(samples)
        timestamps -= timestamps[0]
        cubic_spline = interpolate.CubicSpline(timestamps, samples)
        sample_axises = np.arange(np.min(timestamps), np.max(timestamps), config.SPLINE_SAMPLE_INTERVAL)
        fft_max_x = (np.max(timestamps) - np.min(timestamps)) / config.SPLINE_SAMPLE_INTERVAL / 2
        max_frequency = 1 / config.SPLINE_SAMPLE_INTERVAL
        fft_roi = np.fft.fft(cubic_spline(sample_axises))[1:int(config.MAX_BREATH_FREQUENCY / max_frequency * fft_max_x)]
        frequency = (np.argmax(fft_roi) + 1) / fft_max_x * max_frequency
        return frequency
