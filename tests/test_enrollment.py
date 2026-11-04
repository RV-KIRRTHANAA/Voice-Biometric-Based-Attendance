import os
import sys
import pytest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from datetime import datetime, timedelta
import numpy as np

import src.infrastructure.database as _db_module
from src.infrastructure.database import Base, get_db, Tenant, User, AttendanceSessionModel, ReferenceEmbeddingModel, Enrollment
from src.api.auth import get_password_hash, create_access_token
from src.engine.results import FailureReason
from src.audio.embedding import SpeakerEmbedding

with patch.object(_db_module, "init_db", lambda: None):
    from src.api.main import app

SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_db():
    app.dependency_overrides[get_db] = override_get_db
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    
    tenant = Tenant(id="test_tenant", name="Test")
    db.add(tenant)
    
    user = User(id="test_student", tenant_id="test_tenant", password_hash=get_password_hash("pass123"))
    db.add(user)
    
    # Pre-existing reference
    import io
    bio = io.BytesIO()
    np.save(bio, np.zeros(192))
    old_ref = ReferenceEmbeddingModel(
        id="old_ref_1",
        user_id="test_student",
        tenant_id="test_tenant",
        vector_blob=bio.getvalue(),
        model_id="speechbrain/spkrec-ecapa-voxceleb",
        dimension=192,
        is_active=False
    )
    db.add(old_ref)
    
    # Session for attendance
    session = AttendanceSessionModel(
        id="sess_id_1",
        tenant_id="test_tenant",
        session_token="valid_session",
        starts_at=datetime.utcnow() - timedelta(hours=1),
        ends_at=datetime.utcnow() + timedelta(hours=1),
        is_active=True
    )
    db.add(session)
    
    db.commit()
    db.close()
    
    yield
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)

def get_auth_headers():
    token = create_access_token(data={"sub": "test_student", "tenant_id": "test_tenant", "student_id": "test_student"})
    return {"Authorization": f"Bearer {token}"}

def test_missing_auth():
    files = {"audio_file": ("test.wav", b"fake audio data", "audio/wav")}
    response = client.post("/api/v1/enrollment/", files=files)
    assert response.status_code == 401

@patch("src.application.enrollment_service.load_wav")
@patch("src.application.enrollment_service.detect_speech")
@patch("src.application.enrollment_service.check_liveness")
@patch("src.application.enrollment_service.extract_embedding")
def test_successful_enrollment(mock_extract, mock_liveness, mock_vad, mock_load):
    mock_load.return_value = (16000, np.zeros(16000))
    mock_vad.return_value = (np.zeros(16000), [])
    
    liveness_mock = MagicMock()
    liveness_mock.is_live = True
    liveness_mock.score = 0.99
    mock_liveness.return_value = liveness_mock
    
    mock_extract.return_value = SpeakerEmbedding(vector=np.ones(192), model_id="speechbrain/spkrec-ecapa-voxceleb", dimension=192)
    
    start_resp = client.post("/api/v1/enrollment/start", headers=get_auth_headers())
    assert start_resp.status_code == 200, start_resp.json()
        
    # 1st sample
    files = {"audio_file": ("test.wav", b"fake audio data", "audio/wav")}
    response = client.post("/api/v1/enrollment/", files=files, headers=get_auth_headers())
    if response.status_code != 200:
        print("ERROR RESPONSE:", response.json())
    assert response.status_code == 200
    assert response.json()["status"] == "SAMPLE_ACCEPTED"
    
    # 2nd sample
    files = {"audio_file": ("test.wav", b"fake audio data", "audio/wav")}
    response = client.post("/api/v1/enrollment/", files=files, headers=get_auth_headers())
    assert response.json()["status"] == "SAMPLE_ACCEPTED"
    
    # 3rd sample -> ENROLLED (MIN_SAMPLES = 3)
    files = {"audio_file": ("test.wav", b"fake audio data", "audio/wav")}
    response = client.post("/api/v1/enrollment/", files=files, headers=get_auth_headers())
    assert response.json()["status"] == "ENROLLED"
    
    # Verify DB State
    db = TestingSessionLocal()
    enrollment = db.query(Enrollment).filter(Enrollment.user_id == "test_student").first()
    assert enrollment.status == "ENROLLED"
    
    # Old reference should be inactive, new reference active
    refs = db.query(ReferenceEmbeddingModel).filter(ReferenceEmbeddingModel.user_id == "test_student").all()
    assert len(refs) == 2
    old_ref = [r for r in refs if r.id == "old_ref_1"][0]
    new_ref = [r for r in refs if r.id != "old_ref_1"][0]
    
    assert old_ref.is_active is False
    assert new_ref.is_active is True
    assert new_ref.enrollment_id == enrollment.id
    db.close()

@patch("src.application.attendance_service.load_wav")
@patch("src.application.attendance_service.run_biometric_pipeline")
@patch("src.application.enrollment_service.load_wav")
@patch("src.application.enrollment_service.detect_speech")
@patch("src.application.enrollment_service.check_liveness")
@patch("src.application.enrollment_service.extract_embedding")
def test_enrollment_to_attendance_integration(mock_extract, mock_liveness, mock_vad, mock_load, mock_run_pipeline, mock_attendance_load):
    # Setup mocks for Enrollment
    mock_load.return_value = (16000, np.zeros(16000))
    mock_vad.return_value = (np.zeros(16000), [])
    liveness_mock = MagicMock()
    liveness_mock.is_live = True
    liveness_mock.score = 0.99
    mock_liveness.return_value = liveness_mock
    mock_extract.return_value = SpeakerEmbedding(vector=np.ones(192), model_id="speechbrain/spkrec-ecapa-voxceleb", dimension=192)
    
    start_resp = client.post("/api/v1/enrollment/start", headers=get_auth_headers())
    assert start_resp.status_code == 200, start_resp.json()
        
    # 1. Complete Enrollment (3 samples)
    files = {"audio_file": ("test.wav", b"fake audio data", "audio/wav")}
    client.post("/api/v1/enrollment/", files=files, headers=get_auth_headers())
    client.post("/api/v1/enrollment/", files=files, headers=get_auth_headers())
    response = client.post("/api/v1/enrollment/", files=files, headers=get_auth_headers())
    assert response.json()["status"] == "ENROLLED"
    
    # 2. Setup mocks for Attendance
    mock_attendance_load.return_value = (16000, np.zeros(16000))
    
    def mock_run_side_effect(audio, ref, **kwargs):
        from numpy.testing import assert_almost_equal
        # normalization of ones is ones / sqrt(192)
        norm_ones = np.ones(192) / np.linalg.norm(np.ones(192))
        assert_almost_equal(ref.vector, norm_ones)
        
        from src.engine.results import BiometricResult
        return BiometricResult(passed=True, reason=FailureReason.SUCCESS)
        
    mock_run_pipeline.side_effect = mock_run_side_effect
    
    # 3. Hit Attendance API
    ch_resp = client.post("/api/v1/attendance/challenge", data={"session_token": "valid_session"}, headers=get_auth_headers())
    challenge_id = ch_resp.json()["challenge_id"]
    files = {"audio_file": ("test.wav", b"fake audio data", "audio/wav")}
    data = {"session_token": "valid_session", "challenge_id": challenge_id}
    att_response = client.post("/api/v1/attendance/mark", files=files, data=data, headers=get_auth_headers())
    
    assert att_response.status_code == 200
    assert att_response.json().get("is_present") is True, att_response.json()

