from pydantic import BaseModel, Field
from src.engine.results import FailureReason

class AttendanceResponse(BaseModel):
    """
    Public minimal API response.
    Explicitly scrubs internal ML biometric scores (similarity, liveness thresholds)
    so they are not leaked to the mobile client.
    """
    is_present: bool = Field(..., description="Whether the attendance was successfully marked")
    reason: str = Field(..., description="The structured outcome reason (e.g. SUCCESS, SPEAKER_MISMATCH)")

class EnrollmentResponse(BaseModel):
    status: str
    accepted_samples: int
    required_samples: int
    reason: str
    has_active_reference: bool = False


class EnrollmentStatusResponse(BaseModel):
    status: str
    accepted_samples: int = 0
    required_samples: int = 3
    has_active_reference: bool = False


class ChallengeResponse(BaseModel):
    challenge_id: str
    phrase: str
    expires_at: str
