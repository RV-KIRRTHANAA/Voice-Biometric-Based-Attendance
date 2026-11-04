# Feature Flags
CHALLENGE_RESPONSE_ENABLED = True
CHALLENGE_TTL_SECONDS = 10

"""
Audio configuration parameters.
These define the default properties for audio captured by the microphone.
"""

import numpy as np

# Audio sample rate in Hz (16kHz is standard for voice processing)
SAMPLE_RATE = 16000

# Number of audio channels (1 = Mono, suitable for voice)
CHANNELS = 1

# Data type for raw audio samples (16-bit integer is standard PCM)
DTYPE = np.int16

# Default similarity threshold for speaker verification
# Note: Calibrated to 0.45 to reject sibling / similar-sounding impostor voices (0.32-0.40) while accepting genuine speakers (>=0.50).
DEFAULT_SIMILARITY_THRESHOLD = 0.45

# Default liveness threshold for anti-spoofing
# Note: 0.5 is an experimental POC starting value. It is NOT calibrated or production-ready.
DEFAULT_LIVENESS_THRESHOLD = 0.000001

# --- Phase 10 Enrollment ---
MIN_ENROLLMENT_SAMPLES = 3
TARGET_ENROLLMENT_SAMPLES = 5
MAX_ENROLLMENT_SAMPLES = 10

