import numpy as np
from scipy import interpolate
from scipy import signal

from . import utils
from . import config


class ThermalFace(object):
    """ An object that represents a face entity within a thermal image.

    Attributes:
        parent: The thermal_frame.ThermalFrame object that this face belongs to.
        bounding_box: The bounding box of the face in its belonging frame.
        landmark: The landmark of the face in its belonging frame.
        previous: The thermal_face.ThermalFace object that is believed to be the 
            same face entity in the previous frame.
    """

    def __init__(self, parent, bounding_box, landmark):
        self.parent = parent
        self.bounding_box = bounding_box
        self.landmark = landmark
        self.previous = None
    
    @property
    def timestamp(self):
        """ Returns the timestamp of the frame that the face entity belongs to.
        """
        return self.parent.timestamp
    
    @property
    def thermal_image(self):
        """ Returns the cropped region of the face in the thermal frame.
        """
        return utils.crop(self.parent.thermal_frame, self.bounding_box)
    
    @property
    def grey_image(self):
        """ Returns the cropped region of the face in the grey frame.
        """
        return utils.crop(self.parent.grey_frame, self.bounding_box)
    
    @property
    def temperature_roi(self):
        """ Returns the cropped region of a part of the face in the thermal frame 
            that is used for body temperature estimation.
        """
        return utils.crop(self.parent.thermal_frame, self.bounding_box)
    
    @property
    def breath_roi(self):
        """ Returns the cropped region of a part of the face in the thermal frame 
            that is used for breath rate estimation.
        """
        return self.parent.thermal_frame[
            self.landmark[2, 1] : self.bounding_box[3],
            (self.landmark[3, 0] + self.bounding_box[0]) // 2 : (self.landmark[4, 0] + self.bounding_box[2]) // 2
        ]
    
    @property
    def temperature(self):
        """ Returns the temperature estimation of the face entity. The return value 
            is `None` if the estimation is not available.
        """
        try:
            return np.max(self.temperature_roi)
        except ValueError:
            return None
    
    @property
    def breath_rate(self):
        """ Returns the breath rate (frequency) estimation of the face entity. The 
            return value is `None` if the estimation is not available.
        
        This method summarize the average temperature in the `breath_roi` across 
            all historic tracked face entities. Then it performs FFT and extract 
            the frequency with the maximum spectrum on the range 
            (0, MAX_BREATH_FREQUENCY).
        """
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
        sample_frequencies, power_spectral_density = signal.periodogram(
            cubic_spline(sample_axises), 
            fs=1/config.SPLINE_SAMPLE_INTERVAL
        )
        return sample_frequencies[np.argmax(power_spectral_density)]
