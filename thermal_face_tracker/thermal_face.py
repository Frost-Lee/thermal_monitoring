class ThermalFace(object):

    def __init__(self, bounding_box, landmark):
        self.bounding_box = bounding_box
        self.landmark = landmark
        self.previous = None
