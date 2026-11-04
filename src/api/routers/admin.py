from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from datetime import datetime
from sqlalchemy.orm import Session
import uuid

from src.infrastructure.database import get_db, AttendanceSessionModel
from src.api.deps import get_current_user_token, TokenData

router = APIRouter(prefix="/admin", tags=["admin"])

class SessionCreateRequest(BaseModel):
    starts_at: datetime
    ends_at: datetime

class SessionCreateResponse(BaseModel):
    session_token: str
    starts_at: datetime
    ends_at: datetime

@router.post("/session", response_model=SessionCreateResponse)
def create_session(
    request: SessionCreateRequest,
    db: Session = Depends(get_db),
    # Require authentication even for admin endpoints in this POC
    token_data: TokenData = Depends(get_current_user_token)
):
    """
    POC Endpoint: Creates an attendance session for the current user's tenant.
    """
    session_token = str(uuid.uuid4())
    
    db_session = AttendanceSessionModel(
        id=str(uuid.uuid4()),
        tenant_id=token_data.tenant_id,
        session_token=session_token,
        starts_at=request.starts_at,
        ends_at=request.ends_at,
        is_active=True
    )
    
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    
    return SessionCreateResponse(
        session_token=db_session.session_token,
        starts_at=db_session.starts_at,
        ends_at=db_session.ends_at
    )
