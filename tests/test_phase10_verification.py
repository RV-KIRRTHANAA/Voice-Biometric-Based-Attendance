import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from datetime import datetime, timedelta
import numpy as np
import io

import src.infrastructure.database as _db_module
from src.infrastructure.database import Base, get_db, Tenant, User, ReferenceEmbeddingModel, Enrollment, EnrollmentSample, AttendanceSessionModel
from src.api.auth import get_password_hash, create_access_token
from src.engine.results import FailureReason
from src.audio.embedding import SpeakerEmbedding
from src.engine.results import EnrollmentStatus

# Prevent module-level init_db() in src.api.main from touching the real ./data/oneapp.db
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
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_db():
    app.dependency_overrides[get_db] = override_get_db
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    
    tenant_a = Tenant(id="tenant_a", name="Tenant A")
    tenant_b = Tenant(id="tenant_b", name="Tenant B")
    db.add_all([tenant_a, tenant_b])
    
    user_a = User(id="user_1", tenant_id="tenant_a", password_hash=get_password_hash("pass123"))
    user_b = User(id="user_2", tenant_id="tenant_b", password_hash=get_password_hash("pass123"))
    db.add_all([user_a, user_b])
    
    db.commit()
    db.close()
    
    yield
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)

def get_auth_headers(user_id="user_1", tenant_id="tenant_a"):
    token = create_access_token(data={"sub": user_id, "tenant_id": tenant_id, "student_id": user_id})
    return {"Authorization": f"Bearer {token}"}

# 1. Verify only one active reference can exist for tenant_id + user_id.
# & 6. Verify re-enrollment leaves the previous reference INACTIVE rather than deleted.
@patch("src.application.enrollment_service.load_wav")
@patch("src.application.enrollment_service.detect_speech")
@patch("src.application.enrollment_service.check_liveness")
@patch("src.application.enrollment_service.extract_embedding")
def test_one_active_reference_and_inactive_retention(mock_extract, mock_liveness, mock_vad, mock_load):
    mock_load.return_value = (16000, np.zeros(16000))
    mock_vad.return_value = (np.zeros(16000), [])
    
    liveness_mock = MagicMock()
    liveness_mock.is_live = True
    liveness_mock.score = 0.99
    mock_liveness.return_value = liveness_mock
    
    mock_extract.return_value = SpeakerEmbedding(vector=np.ones(192), model_id="test_model", dimension=192)
    
    # First Enrollment
    start_res1 = client.post("/api/v1/enrollment/start", headers=get_auth_headers())
    assert start_res1.status_code == 200
    for _ in range(3):
        client.post("/api/v1/enrollment/", files={"audio_file": ("t.wav", b"123", "audio/wav")}, headers=get_auth_headers())
    
    db = TestingSessionLocal()
    refs = db.query(ReferenceEmbeddingModel).filter_by(user_id="user_1", tenant_id="tenant_a").all()
    assert len(refs) == 1
    assert refs[0].is_active is True
    first_ref_id = refs[0].id
    db.close()
    
    # Re-enrollment with a distinct speaker vector so we can verify the new reference replaces the old one
    new_vec = np.zeros(192)
    new_vec[0] = 1.0
    mock_extract.return_value = SpeakerEmbedding(vector=new_vec, model_id="test_model", dimension=192)
    start_res2 = client.post("/api/v1/enrollment/start", headers=get_auth_headers())
    assert start_res2.status_code == 200
    for _ in range(3):
        client.post("/api/v1/enrollment/", files={"audio_file": ("t.wav", b"123", "audio/wav")}, headers=get_auth_headers())
        
    db = TestingSessionLocal()
    refs = db.query(ReferenceEmbeddingModel).filter_by(user_id="user_1", tenant_id="tenant_a").all()
    assert len(refs) == 2
    
    active_refs = [r for r in refs if r.is_active]
    inactive_refs = [r for r in refs if not r.is_active]
    
    assert len(active_refs) == 1
    assert len(inactive_refs) == 1
    
    # Verify previous reference is INACTIVE rather than deleted, and new reference is ACTIVE
    assert inactive_refs[0].id == first_ref_id
    assert active_refs[0].id != first_ref_id
    saved_new_vec = np.load(io.BytesIO(active_refs[0].vector_blob), allow_pickle=False)
    np.testing.assert_allclose(saved_new_vec, new_vec)
    db.close()

# 2. Verify failed enrollment cannot accidentally deactivate an existing active reference.
@patch("src.application.enrollment_service.load_wav")
@patch("src.application.enrollment_service.detect_speech")
@patch("src.application.enrollment_service.check_liveness")
@patch("src.application.enrollment_service.extract_embedding")
def test_failed_enrollment_does_not_deactivate(mock_extract, mock_liveness, mock_vad, mock_load):
    # Setup Active Ref
    db = TestingSessionLocal()
    bio = io.BytesIO()
    np.save(bio, np.zeros(192))
    ref = ReferenceEmbeddingModel(id="active_ref", user_id="user_1", tenant_id="tenant_a", vector_blob=bio.getvalue(), model_id="m1", dimension=192, is_active=True)
    db.add(ref)
    db.commit()
    db.close()
    
    mock_load.return_value = (16000, np.zeros(16000))
    mock_vad.return_value = (np.zeros(16000), [])
    liveness_mock = MagicMock()
    liveness_mock.is_live = False # Anti-spoof fail
    mock_liveness.return_value = liveness_mock
    
    start_res = client.post("/api/v1/enrollment/start", headers=get_auth_headers())
    assert start_res.status_code == 200
    res = client.post("/api/v1/enrollment/", files={"audio_file": ("t.wav", b"123", "audio/wav")}, headers=get_auth_headers())
    assert res.json()["status"] == "SAMPLE_REJECTED"
    
    db = TestingSessionLocal()
    refs = db.query(ReferenceEmbeddingModel).filter_by(user_id="user_1", tenant_id="tenant_a").all()
    assert len(refs) == 1
    assert refs[0].is_active is True # Still active
    db.close()

# 3. Verify EnrollmentSample rows are deleted after successful enrollment.
@patch("src.application.enrollment_service.load_wav")
@patch("src.application.enrollment_service.detect_speech")
@patch("src.application.enrollment_service.check_liveness")
@patch("src.application.enrollment_service.extract_embedding")
def test_samples_deleted_after_success(mock_extract, mock_liveness, mock_vad, mock_load):
    mock_load.return_value = (16000, np.zeros(16000))
    mock_vad.return_value = (np.zeros(16000), [])
    liveness_mock = MagicMock()
    liveness_mock.is_live = True
    liveness_mock.score = 0.99
    mock_liveness.return_value = liveness_mock
    mock_extract.return_value = SpeakerEmbedding(vector=np.ones(192), model_id="test_model", dimension=192)
    
    # Insert 1 sample
    client.post("/api/v1/enrollment/start", headers=get_auth_headers())
    client.post("/api/v1/enrollment/", files={"audio_file": ("t.wav", b"123", "audio/wav")}, headers=get_auth_headers())
    
    db = TestingSessionLocal()
    assert db.query(EnrollmentSample).count() == 1
    db.close()
    
    # Complete Enrollment
    client.post("/api/v1/enrollment/", files={"audio_file": ("t.wav", b"123", "audio/wav")}, headers=get_auth_headers())
    client.post("/api/v1/enrollment/", files={"audio_file": ("t.wav", b"123", "audio/wav")}, headers=get_auth_headers())
    
    db = TestingSessionLocal()
    assert db.query(EnrollmentSample).count() == 0
    db.close()

# 4. Verify rejected samples don"t increment accepted_samples.
@patch("src.application.enrollment_service.load_wav")
@patch("src.application.enrollment_service.detect_speech")
@patch("src.application.enrollment_service.check_liveness")
def test_rejected_samples_dont_increment(mock_liveness, mock_vad, mock_load):
    mock_load.return_value = (16000, np.zeros(16000))
    mock_vad.return_value = (np.zeros(16000), [])
    liveness_mock = MagicMock()
    liveness_mock.is_live = False
    mock_liveness.return_value = liveness_mock
    
    client.post("/api/v1/enrollment/start", headers=get_auth_headers())
    res1 = client.post("/api/v1/enrollment/", files={"audio_file": ("t.wav", b"123", "audio/wav")}, headers=get_auth_headers())
    assert res1.json()["accepted_samples"] == 0
    
    res2 = client.post("/api/v1/enrollment/", files={"audio_file": ("t.wav", b"123", "audio/wav")}, headers=get_auth_headers())
    assert res2.json()["accepted_samples"] == 0

# 5. Verify an expired/inactive enrollment cannot continue receiving samples.
# (In our architecture, if the state is ENROLLED, it just starts a new IN_PROGRESS enrollment)
@patch("src.application.enrollment_service.load_wav")
@patch("src.application.enrollment_service.detect_speech")
@patch("src.application.enrollment_service.check_liveness")
@patch("src.application.enrollment_service.extract_embedding")
def test_enrolled_state_starts_new_session(mock_extract, mock_liveness, mock_vad, mock_load):
    mock_load.return_value = (16000, np.zeros(16000))
    mock_vad.return_value = (np.zeros(16000), [])
    liveness_mock = MagicMock()
    liveness_mock.is_live = True
    liveness_mock.score = 0.99
    mock_liveness.return_value = liveness_mock
    mock_extract.return_value = SpeakerEmbedding(vector=np.ones(192), model_id="test_model", dimension=192)
    
    client.post("/api/v1/enrollment/start", headers=get_auth_headers())
    for _ in range(3):
        client.post("/api/v1/enrollment/", files={"audio_file": ("t.wav", b"123", "audio/wav")}, headers=get_auth_headers())
        
    db = TestingSessionLocal()
    enrollments = db.query(Enrollment).filter_by(user_id="user_1").all()
    assert len(enrollments) == 1
    assert enrollments[0].status == EnrollmentStatus.ENROLLED.value
    db.close()
    
    # Send another sample (need to start a new enrollment session)
    start_res = client.post("/api/v1/enrollment/start", headers=get_auth_headers())
    assert start_res.status_code == 200
    res = client.post("/api/v1/enrollment/", files={"audio_file": ("t.wav", b"123", "audio/wav")}, headers=get_auth_headers())
    assert res.json()["accepted_samples"] == 1
    
    db = TestingSessionLocal()
    enrollments = db.query(Enrollment).filter_by(user_id="user_1").order_by(Enrollment.created_at.asc()).all()
    assert len(enrollments) == 2
    assert enrollments[0].status == EnrollmentStatus.ENROLLED.value
    assert enrollments[1].status == EnrollmentStatus.IN_PROGRESS.value
    db.close()

# 7. Verify tenant A can never use tenant B"s reference embedding.
@patch("src.application.attendance_service.load_wav")
@patch("src.application.attendance_service.evaluate_attendance")
def test_tenant_isolation(mock_eval, mock_load):
    mock_load.return_value = (16000, np.zeros(16000))
    mock_eval.return_value = MagicMock() # Will not be called anyway if no ref
    
    db = TestingSessionLocal()
    bio = io.BytesIO()
    np.save(bio, np.zeros(192))
    # Tenant B has the active reference for user_1
    ref = ReferenceEmbeddingModel(id="ref_b", user_id="user_1", tenant_id="tenant_b", vector_blob=bio.getvalue(), model_id="m1", dimension=192, is_active=True)
    session = AttendanceSessionModel(id="s1", tenant_id="tenant_a", session_token="tok_a", starts_at=datetime.utcnow() - timedelta(hours=1), ends_at=datetime.utcnow() + timedelta(hours=1), is_active=True)
    db.add_all([ref, session])
    db.commit()
    db.close()
    
    # Tenant A user_1 tries to mark attendance
    ch_res = client.post("/api/v1/attendance/challenge", data={"session_token": "tok_a"}, headers=get_auth_headers(user_id="user_1", tenant_id="tenant_a"))
    ch_id = ch_res.json()["challenge_id"]
    res = client.post("/api/v1/attendance/mark", files={"audio_file": ("t.wav", b"123", "audio/wav")}, data={"session_token": "tok_a", "challenge_id": ch_id}, headers=get_auth_headers(user_id="user_1", tenant_id="tenant_a"))
    
    # Expect 404 because no active reference for user_1 in tenant_a
    assert res.status_code == 404
    assert "Reference embedding not found" in res.json()["detail"]

# 8. Verify attendance selects only the ACTIVE reference.
@patch("src.application.attendance_service.load_wav")
@patch("src.application.attendance_service.evaluate_attendance")
@patch("src.application.attendance_service.run_biometric_pipeline")
def test_attendance_selects_active(mock_run, mock_eval, mock_load):
    mock_load.return_value = (16000, np.zeros(16000))
    # mock_run returns a BiometricResult which evaluate_attendance will receive
    from src.engine.results import BiometricResult, FailureReason, AttendanceDecision
    mock_run.return_value = BiometricResult(passed=True, reason=FailureReason.SUCCESS)
    mock_eval.return_value = AttendanceDecision(marked=True, reason=FailureReason.SUCCESS, biometric_data=mock_run.return_value)
    
    db = TestingSessionLocal()
    import io
    bio_active = io.BytesIO()
    np.save(bio_active, np.ones(192))
    
    bio_inactive = io.BytesIO()
    np.save(bio_inactive, np.zeros(192))
    
    ref_inactive = ReferenceEmbeddingModel(id="ref_inact", user_id="user_1", tenant_id="tenant_a", vector_blob=bio_inactive.getvalue(), model_id="m1", dimension=192, is_active=False)
    ref_active = ReferenceEmbeddingModel(id="ref_act", user_id="user_1", tenant_id="tenant_a", vector_blob=bio_active.getvalue(), model_id="m1", dimension=192, is_active=True)
    
    session = AttendanceSessionModel(id="s1", tenant_id="tenant_a", session_token="tok_a", starts_at=datetime.utcnow() - timedelta(hours=1), ends_at=datetime.utcnow() + timedelta(hours=1), is_active=True)
    db.add_all([ref_inactive, ref_active, session])
    db.commit()
    db.close()
    
    ch_res = client.post("/api/v1/attendance/challenge", data={"session_token": "tok_a"}, headers=get_auth_headers(user_id="user_1", tenant_id="tenant_a"))
    ch_id = ch_res.json()["challenge_id"]
    res = client.post("/api/v1/attendance/mark", files={"audio_file": ("t.wav", b"123", "audio/wav")}, data={"session_token": "tok_a", "challenge_id": ch_id}, headers=get_auth_headers(user_id="user_1", tenant_id="tenant_a"))
    
    assert res.status_code == 200
    
    # Assert run_biometric_pipeline was called with the ACTIVE reference
    args, kwargs = mock_run.call_args
    passed_ref = args[1]
    
    from numpy.testing import assert_almost_equal
    assert_almost_equal(passed_ref.vector, np.ones(192))
