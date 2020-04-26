# The probability threshold for face detection. If the probability for face detection
# is greater than this threshold, the entity will be taken as face.
FACE_DETECTION_THRESHOLD = 0.5

# The similarity threshold for face linking. Only 2 `thermal_face.ThermalFace` 
# objects with similarity greater than this will be considered for linking.
FACE_LINK_THRESHOLD = 0.2

# The minimum sample amount for estimating breath rate. If a face is tracked for 
# more frames than this threshold, breath rate estimation will be performed.
BREATH_RATE_MIN_SAMPLE_THRESHOLD = 128

# The cubic spline interval before performing FFT.
SPLINE_SAMPLE_INTERVAL = 0.1

# The ID of the GPU to be used. -1 for using CPU.
GPU_ID = -1

# The maximum cached frames for estimating breath rate.
MAX_CACHED_FRAMES = 512

# The temperature annotation will be updated every `TEMPERATURE_UPDATE_FRAMES` frames.
TEMPERATURE_UPDATE_FRAMES = 5

# The breath rate annotation will be updated every `BREATH_RATE_UPDATE_FRAMES` frames.
BREATH_RATE_UPDATE_FRAMES = 11

# The breath curve plot will be updated every `BREATH_CURVE_UPDATE_FRAMES` frames.
BREATH_CURVE_UPDATE_FRAMES = 7

# The resolution for visualization.
VISUALIZATION_RESOLUTION = (800, 600)
