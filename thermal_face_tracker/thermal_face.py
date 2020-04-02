from . import utils


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
    def breath_roi(self):
        return self.parent.thermal_frame[
            self.landmark[2, 1] : self.bounding_box[3],
            (self.landmark[3, 0] + self.bounding_box[0]) // 2 : (self.landmark[4, 0] + self.bounding_box[2]) // 2
        ]
