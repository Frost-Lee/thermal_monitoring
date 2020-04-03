# The probability threshold for face detection. If the probability for face detection
# is greater than this threshold, the entity will be taken as face.
FACE_DETECTION_THRESHOLD = 0.5

# The minimum sample amount for estimating breath rate. If a face is tracked for 
# more frames than this threshold, breath rate estimation will be performed.
BREATH_RATE_MIN_SAMPLE_THRESHOLD = 16

# The cubic spline interval before performing FFT.
SPLINE_SAMPLE_INTERVAL = 0.1

# The assumed maximum frequency of breathing. Signals with higher frequency will 
# be filtered.
MAX_BREATH_FREQUENCY = 1

# The ID of the GPU to be used. -1 for using CPU.
GPU_ID = -1
