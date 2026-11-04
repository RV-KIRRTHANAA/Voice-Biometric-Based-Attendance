import os
import tempfile
from datetime import datetime, timezone, timedelta
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional
from src.api.schemas import AttendanceResponse, ChallengeResponse
from sqlalchemy.orm import Session

from src.api.schemas import AttendanceResponse
from src.application.attendance_service import AttendanceService
from src.application.repositories import DatabaseReferenceEmbeddingRepository
from src.engine.results import FailureReason
from src.infrastructure.database import get_db, AttendanceSessionModel, AttendanceRecord, User
from src.api.deps import get_current_user_token, TokenData

router = APIRouter(prefix="/attendance", tags=["attendance"])

# IST offset for display formatting
IST = timezone(timedelta(hours=5, minutes=30))

# Attendance status values
class AttendanceStatus:
    PRESENT    = "PRESENT"
    NOT_MARKED = "NOT_MARKED"
    UPCOMING   = "UPCOMING"
    ENDED      = "ENDED"


class SessionInfo(BaseModel):
    """
    Client-visible session descriptor.

    session_id      : stable identifier for this session (UUID). Android uses this
                      to identify which session the user tapped.
    session_token   : credential submitted to /mark. Exposed here because the client
                      must include it in the multipart form. In production this would
                      be a short-lived signed token; for this POC it is stable.
    class_name      : display name from DB — NOT derived from session_token.
    starts_at       : ISO-8601 string in IST (with +05:30 offset) for unambiguous display.
    ends_at         : same.
    attendance_status: PRESENT | NOT_MARKED | UPCOMING | ENDED for the authenticated student.
    """
    session_id: str
    session_token: str
    class_name: str
    starts_at: str
    ends_at: str
    attendance_status: str


class StudentInfo(BaseModel):
    student_id: str
    display_name: str
    tenant_name: str


def get_attendance_service(db: Session = Depends(get_db)) -> AttendanceService:
    repo = DatabaseReferenceEmbeddingRepository(db=db)
    return AttendanceService(repository=repo, db=db)


def _fmt_ist(dt: datetime) -> str:
    """Format a naive-UTC datetime as IST ISO-8601 string."""
    utc_dt = dt.replace(tzinfo=timezone.utc)
    ist_dt = utc_dt.astimezone(IST)
    return ist_dt.strftime("%Y-%m-%dT%H:%M:%S+05:30")


def _time_ist(dt: datetime) -> str:
    """Short HH:MM display in IST."""
    utc_dt = dt.replace(tzinfo=timezone.utc)
    return utc_dt.astimezone(IST).strftime("%H:%M")


@router.get("/me", response_model=StudentInfo)
def get_student_info(
    token_data: TokenData = Depends(get_current_user_token),
    db: Session = Depends(get_db)
):
    """Returns display identity for the authenticated student."""
    user = db.query(User).filter(
        User.id == token_data.student_id,
        User.tenant_id == token_data.tenant_id
    ).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return StudentInfo(
        student_id=user.id,
        display_name=user.display_name or user.id,
        tenant_name=user.tenant.name
    )


@router.get("/sessions", response_model=List[SessionInfo])
def list_sessions(
    token_data: TokenData = Depends(get_current_user_token),
    db: Session = Depends(get_db)
):
    """
    Returns ALL of today's sessions for the authenticated tenant.
    Includes UPCOMING, ACTIVE, and ENDED sessions so the student
    can see the full day schedule.
    attendance_status is computed per-student from AttendanceRecord —
    never from client-supplied data.
    """
    now = datetime.utcnow()
    # Return sessions for today (UTC day)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_end   = today_start.replace(hour=23, minute=59, second=59)

    sessions = db.query(AttendanceSessionModel).filter(
        AttendanceSessionModel.tenant_id == token_data.tenant_id,
        AttendanceSessionModel.is_active == True,
        AttendanceSessionModel.starts_at >= today_start,
        AttendanceSessionModel.starts_at <= today_end
    ).order_by(AttendanceSessionModel.starts_at).all()

    # Fetch this student's attendance records for these session IDs
    session_ids = [s.id for s in sessions]
    marked_sessions = set()
    if session_ids:
        records = db.query(AttendanceRecord).filter(
            AttendanceRecord.user_id == token_data.student_id,
            AttendanceRecord.tenant_id == token_data.tenant_id,
            AttendanceRecord.session_id.in_(session_ids)
        ).all()
        marked_sessions = {r.session_id for r in records}

    result = []
    for s in sessions:
        if s.id in marked_sessions:
            att_status = AttendanceStatus.PRESENT
        elif now < s.starts_at:
            att_status = AttendanceStatus.UPCOMING
        elif now > s.ends_at:
            att_status = AttendanceStatus.ENDED
        else:
            att_status = AttendanceStatus.NOT_MARKED

        result.append(SessionInfo(
            session_id=s.id,
            session_token=s.session_token,
            class_name=s.class_name,
            starts_at=_fmt_ist(s.starts_at),
            ends_at=_fmt_ist(s.ends_at),
            attendance_status=att_status
        ))
    return result


@router.post("/challenge", response_model=ChallengeResponse)
async def generate_challenge(
    session_token: str = Form(..., description="The session credential from /sessions"),
    token_data: TokenData = Depends(get_current_user_token),
    service: AttendanceService = Depends(get_attendance_service)
):
    try:
        result = service.generate_challenge(
            student_id=token_data.student_id,
            tenant_id=token_data.tenant_id,
            session_token=session_token
        )
        return ChallengeResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@router.post("/mark", response_model=AttendanceResponse)
async def mark_attendance(
    audio_file: UploadFile = File(...),
    session_token: str = Form(..., description="The session credential from /sessions"),
    challenge_id: Optional[str] = Form(None, description="The challenge ID received from /challenge"),
    token_data: TokenData = Depends(get_current_user_token),
    service: AttendanceService = Depends(get_attendance_service)
):
    if not audio_file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file uploaded")
    if not audio_file.filename.endswith(".wav"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported file format, must be .wav")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp_path = tmp.name
            while chunk := await audio_file.read(8192):
                tmp.write(chunk)
        decision = service.process_attendance(
            student_id=token_data.student_id,
            tenant_id=token_data.tenant_id,
            audio_path=tmp_path,
            session_token=session_token,
            challenge_id=challenge_id
        )
        if decision.reason == FailureReason.INVALID_REFERENCE:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Reference embedding not found. Please enroll first."
            )
        return AttendanceResponse(
            is_present=decision.marked,
            reason=decision.reason.value if hasattr(decision.reason, "value") else str(decision.reason)
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
