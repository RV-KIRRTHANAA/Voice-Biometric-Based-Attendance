from enum import Enum
from dataclasses import dataclass
from typing import Optional

class FailureReason(Enum):
    SUCCESS = "SUCCESS"
    NO_SPEECH_DETECTED = "NO_SPEECH_DETECTED"
    AUDIO_TOO_SHORT = "AUDIO_TOO_SHORT"
    AUDIO_TOO_NOISY = "AUDIO_TOO_NOISY"
    ANTI_SPOOF_FAILED = "ANTI_SPOOF_FAILED"
    SPEAKER_MISMATCH = "SPEAKER_MISMATCH"
    MODEL_ERROR = "MODEL_ERROR"
    INVALID_REFERENCE = "INVALID_REFERENCE"
    CHALLENGE_EXPIRED = "CHALLENGE_EXPIRED"
    CHALLENGE_INVALID = "CHALLENGE_INVALID"
    CHALLENGE_USED = "CHALLENGE_USED"
    CHALLENGE_MISMATCH = "CHALLENGE_MISMATCH"
    ASR_ERROR = "ASR_ERROR"

@dataclass
class BiometricResult:
    """The raw output from the biometric ML pipeline orchestrator."""
    passed: bool
    reason: FailureReason
    liveness_score: Optional[float] = None
    similarity_score: Optional[float] = None
    liveness_threshold: Optional[float] = None
    similarity_threshold: Optional[float] = None

@dataclass
class AttendanceDecision:
    """The final business decision output from the policy engine."""
    marked: bool
    reason: FailureReason
    # Metadata for logging/auditing, without saving raw audio
    biometric_data: BiometricResult

class EnrollmentStatus(Enum):
    NOT_ENROLLED = "NOT_ENROLLED"
    IN_PROGRESS = "IN_PROGRESS"
    ENROLLED = "ENROLLED"
    FAILED = "FAILED"

