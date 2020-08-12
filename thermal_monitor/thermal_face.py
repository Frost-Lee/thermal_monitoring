import numpy as np
from scipy import interpolate
from scipy import signal
import shortuuid

from . import utils
from . import config


class ThermalFace(object):
    """ An object that represents a face entity within a thermal image.

    Attributes:
        parent: The `thermal_frame.ThermalFrame` object that this face belongs to.
        bounding_box: The bounding box of the face in its belonging frame.
        landmark: The landmark of the face in its belonging frame.
        previous: The `thermal_face.ThermalFace` object that is believed to be the 
            same face entity in the previous frame.
    """

    def __init__(self, parent, bounding_box, landmark):
        self.uuid = shortuuid.uuid()
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

    def similarity(self, another_face):
        """ Returns the similarity of the face with another face. The greater this 
            value is, the more similar this face is with another face.
        
        The similarity ranges from 0 to 1 (boundary included). This implementation 
        adopts IoU of the bounding boxes of the two faces.
        
        Args:
            another_face: Another `thermal_face.ThermalFace` object to compare with.
        """
        bb_1, bb_2 = self.bounding_box, another_face.bounding_box

        def box_area(y_1, x_1, y_2, x_2):
            if x_2 < x_1 or y_2 < y_1:
                return 0
            else:
                return (x_2 - x_1) * (y_2 - y_1)
        intersection_area = box_area(
            max(bb_1[0], bb_2[0]),
            max(bb_1[1], bb_2[1]),
            min(bb_1[2], bb_2[2]),
            min(bb_1[3], bb_2[3])
        )
        union_area = box_area(*bb_1) + box_area(*bb_2) - intersection_area
        return intersection_area / union_area

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
        return utils.crop(self.parent.thermal_frame, [
            (self.landmark[3, 0] + self.bounding_box[0]) // 2,
            self.landmark[2, 1],
            (self.landmark[4, 0] + self.bounding_box[2]) // 2,
            self.bounding_box[3]
        ])

    @property
    def breath_samples(self):
        """ Returns the timestamps and breath ROI average temperature samples for 
            breath rate analysis.
        """
        timestamps, samples = [], []
        root = self
        while root is not None:
            timestamps = [root.timestamp] + timestamps
            samples = [np.mean(root.breath_roi)] + samples
            root = root.previous
        timestamps, samples = np.array(timestamps), np.array(samples)
        timestamps -= timestamps[0]
        return timestamps, samples

    @property
    def temperature(self):
        """ Returns the temperature estimation of the face entity. The return value 
            is `None` if the estimation is not available.
        """
        return np.max(self.temperature_roi)

    @property
    def breath_rate(self):
        """ Returns the breath rate (frequency) estimation of the face entity. The 
            return value is `None` if the estimation is not available.
        
        This method summarize the average temperature in the `breath_roi` across 
            all historic tracked face entities. Then it performs FFT and extract 
            the frequency with the maximum spectrum.
        """
        timestamps, samples = self.breath_samples
        if len(timestamps) < config.BREATH_RATE_MIN_SAMPLE_THRESHOLD:
            return None
        cubic_spline = interpolate.CubicSpline(timestamps, samples)
        sample_axes = np.arange(np.min(timestamps), np.max(timestamps), config.SPLINE_SAMPLE_INTERVAL)
        sample_frequencies, power_spectral_density = signal.periodogram(
            cubic_spline(sample_axes),
            fs=1/config.SPLINE_SAMPLE_INTERVAL
        )
        return sample_frequencies[np.argmax(power_spectral_density)]
