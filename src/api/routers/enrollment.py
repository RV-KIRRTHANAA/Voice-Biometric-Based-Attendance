import os
import tempfile
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.api.schemas import EnrollmentResponse, EnrollmentStatusResponse

from src.application.enrollment_service import EnrollmentService
from src.infrastructure.database import get_db
from src.api.deps import get_current_user_token, TokenData

router = APIRouter(prefix="/enrollment", tags=["enrollment"])
def get_enrollment_service(db: Session = Depends(get_db)) -> EnrollmentService:
    return EnrollmentService(db=db)

@router.get("/status", response_model=EnrollmentStatusResponse)
async def get_enrollment_status_route(
    token_data: TokenData = Depends(get_current_user_token),
    service: EnrollmentService = Depends(get_enrollment_service)
):
    result = service.get_enrollment_status(user_id=token_data.student_id, tenant_id=token_data.tenant_id)
    return EnrollmentStatusResponse(**result)

@router.post("/start", response_model=EnrollmentStatusResponse)
async def start_enrollment_route(
    token_data: TokenData = Depends(get_current_user_token),
    service: EnrollmentService = Depends(get_enrollment_service)
):
    try:
        result = service.start_enrollment(user_id=token_data.student_id, tenant_id=token_data.tenant_id)
        return EnrollmentStatusResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))




@router.post("/", response_model=EnrollmentResponse)
async def process_enrollment(
    audio_file: UploadFile = File(...),
    token_data: TokenData = Depends(get_current_user_token),
    service: EnrollmentService = Depends(get_enrollment_service)
):
    if not audio_file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file uploaded")
    if not audio_file.filename.endswith(".wav"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported file format, must be .wav")
        
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp_path = tmp.name
            while chunk := await audio_file.read(8192):
                tmp.write(chunk)
                
        # Identity comes entirely from the JWT
        try:
            result = service.process_enrollment_sample(
                user_id=token_data.student_id,
                tenant_id=token_data.tenant_id,
                audio_path=tmp_path
            )
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        
        return EnrollmentResponse(**result)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

