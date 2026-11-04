import os
import sys
import pytest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from datetime import datetime, timedelta

import src.infrastructure.database as _db_module
from src.infrastructure.database import Base, get_db, Tenant, User, AttendanceSessionModel, ReferenceEmbeddingModel, AttendanceRecord, Challenge
from src.api.auth import get_password_hash, create_access_token
from src.engine.results import AttendanceDecision, FailureReason, BiometricResult

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

    tenant = Tenant(id="test_tenant", name="Test University")
    db.add(tenant)

    user = User(
        id="test_student", tenant_id="test_tenant",
        password_hash=get_password_hash("pass123"),
        display_name="Test Student"
    )
    db.add(user)

    session = AttendanceSessionModel(
        id="sess_id_1",
        tenant_id="test_tenant",
        session_token="valid_session",
        class_name="Test Class",
        starts_at=datetime.utcnow() - timedelta(hours=1),
        ends_at=datetime.utcnow() + timedelta(hours=1),
        is_active=True
    )
    db.add(session)

    import io, numpy as np
    bio = io.BytesIO()
    np.save(bio, np.zeros(192))
    ref = ReferenceEmbeddingModel(
        id="ref_1", user_id="test_student", tenant_id="test_tenant",
        vector_blob=bio.getvalue(),
        model_id="speechbrain/spkrec-ecapa-voxceleb",
        dimension=192, is_active=True
    )
    db.add(ref)

    # Add challenge for tests
    c = Challenge(
        id="fake_id",
        tenant_id="test_tenant",
        user_id="test_student",
        session_id="sess_id_1",
        expected_phrase="present sir",
        created_at=datetime.utcnow(),
        expires_at=datetime.utcnow() + timedelta(hours=1),
        consumed=False
    )
    db.add(c)
    
    db.commit()
    db.close()

    yield

    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)


def get_auth_headers():
    token = create_access_token(data={"sub": "test_student", "tenant_id": "test_tenant", "student_id": "test_student"})
    return {"Authorization": f"Bearer {token}"}


def _get_headers(student_id, tenant_id):
    token = create_access_token(data={"sub": student_id, "tenant_id": tenant_id, "student_id": student_id})
    return {"Authorization": f"Bearer {token}"}


def _make_session(db, sid, token, class_name, tenant_id="test_tenant",
                  hours_offset_start=-1, hours_offset_end=1):
    s = AttendanceSessionModel(
        id=sid, tenant_id=tenant_id, session_token=token,
        class_name=class_name,
        starts_at=datetime.utcnow() + timedelta(hours=hours_offset_start),
        ends_at=datetime.utcnow() + timedelta(hours=hours_offset_end),
        is_active=True
    )
    db.add(s)
    db.commit()
    return s


# ─── Original tests ───────────────────────────────────────────

def get_challenge_id(session_token="valid_session"):
    resp = client.post("/api/v1/attendance/challenge", data={"session_token": session_token}, headers=get_auth_headers())
    return resp.json()["challenge_id"]

def test_missing_auth():
    files = {"audio_file": ("test.wav", b"fake audio data", "audio/wav")}
    data = {"session_token": "valid_session"}
    response = client.post("/api/v1/attendance/mark", files=files, data=data)
    assert response.status_code == 401

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200

def test_auth_token():
    response = client.post("/api/v1/auth/token", data={"username": "test_student", "password": "pass123"})
    assert response.status_code == 200
    assert "access_token" in response.json()

def test_auth_token_invalid():
    response = client.post("/api/v1/auth/token", data={"username": "test_student", "password": "wrong"})
    assert response.status_code == 401

@patch("src.application.attendance_service.load_wav")
@patch("src.application.attendance_service.evaluate_attendance")
def test_successful_match(mock_eval, mock_load):
    import numpy as np
    mock_load.return_value = (16000, np.zeros(16000))
    mock_eval.return_value = AttendanceDecision(marked=True, reason=FailureReason.SUCCESS, biometric_data=BiometricResult(passed=True, reason=FailureReason.SUCCESS))
    files = {"audio_file": ("test.wav", b"fake audio data", "audio/wav")}
    data = {"session_token": "valid_session", "challenge_id": get_challenge_id()}
    response = client.post("/api/v1/attendance/mark", files=files, data=data, headers=get_auth_headers())
    assert response.status_code == 200
    assert response.json()["is_present"] is True
    assert response.json()["reason"] == "SUCCESS"

@patch("src.application.attendance_service.load_wav")
@patch("src.application.attendance_service.evaluate_attendance")
def test_biometric_rejection(mock_eval, mock_load):
    import numpy as np
    mock_load.return_value = (16000, np.zeros(16000))
    mock_eval.return_value = AttendanceDecision(marked=False, reason=FailureReason.ANTI_SPOOF_FAILED, biometric_data=BiometricResult(passed=False, reason=FailureReason.ANTI_SPOOF_FAILED))
    files = {"audio_file": ("test.wav", b"fake audio data", "audio/wav")}
    data = {"session_token": "valid_session", "challenge_id": get_challenge_id()}
    response = client.post("/api/v1/attendance/mark", files=files, data=data, headers=get_auth_headers())
    assert response.status_code == 200
    assert response.json()["is_present"] is False
    assert response.json()["reason"] == "ANTI_SPOOF_FAILED"

@patch("src.application.attendance_service.load_wav")
def test_missing_reference(mock_load):
    import numpy as np
    mock_load.return_value = (16000, np.zeros(16000))
    ch_id = get_challenge_id()
    db = TestingSessionLocal()
    db.query(ReferenceEmbeddingModel).delete()
    db.commit()
    db.close()
    files = {"audio_file": ("test.wav", b"fake audio data", "audio/wav")}
    data = {"session_token": "valid_session", "challenge_id": ch_id}
    response = client.post("/api/v1/attendance/mark", files=files, data=data, headers=get_auth_headers())
    assert response.status_code == 404
    assert "Reference embedding not found" in response.json()["detail"]

@patch("src.application.attendance_service.load_wav")
def test_invalid_session(mock_load):
    import numpy as np
    mock_load.return_value = (16000, np.zeros(16000))
    files = {"audio_file": ("test.wav", b"fake audio data", "audio/wav")}
    data = {"session_token": "invalid_session_here"}
    response = client.post("/api/v1/attendance/mark", files=files, data=data, headers=get_auth_headers())
    assert response.status_code == 200
    assert response.json()["reason"] == "INVALID_SESSION"


# ─── Session list tests ───────────────────────────────────────

def test_sessions_requires_auth():
    resp = client.get("/api/v1/attendance/sessions")
    assert resp.status_code == 401

def test_sessions_returns_active_sessions():
    resp = client.get("/api/v1/attendance/sessions", headers=get_auth_headers())
    assert resp.status_code == 200
    active = [s for s in resp.json() if s["session_token"] == "valid_session"]
    assert len(active) == 1

def test_sessions_are_tenant_scoped():
    db = TestingSessionLocal()
    db.add(Tenant(id="other_tenant", name="Other"))
    db.add(User(id="other_student", tenant_id="other_tenant",
                password_hash=get_password_hash("pass"), display_name="Other"))
    db.commit()
    db.close()
    resp = client.get("/api/v1/attendance/sessions", headers=_get_headers("other_student", "other_tenant"))
    assert resp.status_code == 200
    assert "valid_session" not in [s["session_token"] for s in resp.json()]

def test_sessions_not_marked_status():
    resp = client.get("/api/v1/attendance/sessions", headers=get_auth_headers())
    assert resp.status_code == 200
    s = next((x for x in resp.json() if x["session_token"] == "valid_session"), None)
    assert s is not None
    assert s["attendance_status"] == "NOT_MARKED"

def test_sessions_present_status_after_marking():
    db = TestingSessionLocal()
    db.add(AttendanceRecord(tenant_id="test_tenant", user_id="test_student", session_id="sess_id_1"))
    db.commit()
    db.close()
    resp = client.get("/api/v1/attendance/sessions", headers=get_auth_headers())
    s = next((x for x in resp.json() if x["session_token"] == "valid_session"), None)
    assert s["attendance_status"] == "PRESENT"

def test_sessions_student_a_and_b_see_own_status():
    db = TestingSessionLocal()
    db.add(User(id="student_b", tenant_id="test_tenant",
                password_hash=get_password_hash("pass"), display_name="Student B"))
    db.add(AttendanceRecord(tenant_id="test_tenant", user_id="test_student", session_id="sess_id_1"))
    db.commit()
    db.close()
    resp = client.get("/api/v1/attendance/sessions", headers=_get_headers("student_b", "test_tenant"))
    s = next((x for x in resp.json() if x["session_token"] == "valid_session"), None)
    assert s["attendance_status"] == "NOT_MARKED"

def test_sessions_class_name_comes_from_db():
    resp = client.get("/api/v1/attendance/sessions", headers=get_auth_headers())
    s = next((x for x in resp.json() if x["session_token"] == "valid_session"), None)
    assert s["class_name"] == "Test Class"      # from DB, not token-derived
    assert s["class_name"] != "Valid Session"   # not from token

def test_sessions_upcoming_status():
    db = TestingSessionLocal()
    _make_session(db, "future_sess", "future_tok", "Future Class",
                  hours_offset_start=2, hours_offset_end=3)
    db.close()
    resp = client.get("/api/v1/attendance/sessions", headers=get_auth_headers())
    s = next((x for x in resp.json() if x["session_token"] == "future_tok"), None)
    assert s["attendance_status"] == "UPCOMING"

def test_sessions_ended_status():
    db = TestingSessionLocal()
    _make_session(db, "past_sess", "past_tok", "Past Class",
                  hours_offset_start=-0.05, hours_offset_end=-0.01)
    db.close()
    resp = client.get("/api/v1/attendance/sessions", headers=get_auth_headers())
    s = next((x for x in resp.json() if x["session_token"] == "past_tok"), None)
    assert s["attendance_status"] == "ENDED"

def test_mark_ended_session_is_rejected():
    db = TestingSessionLocal()
    _make_session(db, "past_sess2", "past_tok2", "Ended Class",
                  hours_offset_start=-0.05, hours_offset_end=-0.01)
    db.close()
    files = {"audio_file": ("test.wav", b"fake", "audio/wav")}
    resp = client.post("/api/v1/attendance/mark", files=files,
                       data={"session_token": "past_tok2", "challenge_id": "fake_id"}, headers=get_auth_headers())
    assert resp.json()["reason"] == "SESSION_EXPIRED_OR_NOT_STARTED"

def test_mark_upcoming_session_is_rejected():
    db = TestingSessionLocal()
    _make_session(db, "future_sess2", "future_tok2", "Future Class",
                  hours_offset_start=2, hours_offset_end=3)
    db.close()
    files = {"audio_file": ("test.wav", b"fake", "audio/wav")}
    resp = client.post("/api/v1/attendance/mark", files=files,
                       data={"session_token": "future_tok2", "challenge_id": "fake_id"}, headers=get_auth_headers())
    assert resp.json()["reason"] == "SESSION_EXPIRED_OR_NOT_STARTED"

@patch("src.application.attendance_service.load_wav")
@patch("src.application.attendance_service.evaluate_attendance")
def test_duplicate_mark_returns_already_marked(mock_eval, mock_load):
    import numpy as np
    mock_load.return_value = (16000, np.zeros(16000))
    mock_eval.return_value = AttendanceDecision(
        marked=True, reason=FailureReason.SUCCESS,
        biometric_data=BiometricResult(passed=True, reason=FailureReason.SUCCESS)
    )
    files = {"audio_file": ("test.wav", b"fake", "audio/wav")}
    headers = get_auth_headers()
    data1 = {"session_token": "valid_session", "challenge_id": get_challenge_id()}
    resp1 = client.post("/api/v1/attendance/mark", files=files, data=data1, headers=headers)
    assert resp1.json()["is_present"] is True
    data2 = {"session_token": "valid_session", "challenge_id": get_challenge_id()}
    resp2 = client.post("/api/v1/attendance/mark", files=files, data=data2, headers=headers)
    assert resp2.json()["reason"] == "ALREADY_MARKED"

def test_me_endpoint_returns_student_info():
    resp = client.get("/api/v1/attendance/me", headers=get_auth_headers())
    assert resp.status_code == 200
    data = resp.json()
    assert data["student_id"] == "test_student"
    assert data["display_name"] == "Test Student"
    assert data["tenant_name"] == "Test University"

