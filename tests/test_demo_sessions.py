import io
import os
import uuid
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

import src.infrastructure.database as _db_module
from src.api.auth import get_password_hash
from src.audio.embedding import SpeakerEmbedding
from src.engine.results import AttendanceDecision, BiometricResult, FailureReason
from src.infrastructure.database import (
    AttendanceRecord,
    AttendanceSessionModel,
    Base,
    ReferenceEmbeddingModel,
    Tenant,
    User,
    get_db,
)
from scripts.seed_dev_data import DEMO_CLASSES, build_class_sessions, seed

with patch.object(_db_module, "init_db", lambda: None):
    from src.api.main import app

client = TestClient(app)


@pytest.fixture
def isolated_db(tmp_path):
    """Provides an isolated SQLite database using tmp_path."""
    db_file = tmp_path / "test_demo_sessions.db"
    test_engine = create_engine(
        f"sqlite:///{db_file}",
        connect_args={"check_same_thread": False},
    )
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=test_engine
    )
    Base.metadata.create_all(bind=test_engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        app.dependency_overrides.clear()
        Base.metadata.drop_all(bind=test_engine)
        test_engine.dispose()


def _get_token(client, username="student_001", password="password123"):
    response = client.post("/api/v1/auth/token", data={"username": username, "password": password})
    return response.json()["access_token"]


def test_build_class_sessions_structure_and_bounds():
    """Verify build_class_sessions produces 6 classes spanning the full day."""
    now = datetime(2026, 11, 1, 14, 30, 0)
    sessions = build_class_sessions(now_dt=now)

    assert len(sessions) == 6
    expected_names = [
        "Mathematics",
        "Physics",
        "Chemistry",
        "Computer Science",
        "English",
        "Voice Biometrics Lab",
    ]
    actual_names = [s["class_name"] for s in sessions]
    assert actual_names == expected_names

    for s in sessions:
        assert s["token"].endswith("20261101")
        assert s["starts_at"] <= now <= s["ends_at"]


def test_seed_idempotency_does_not_duplicate_sessions(isolated_db: Session):
    """Running seed multiple times on the same day updates existing sessions without duplication."""
    seed(db_session=isolated_db)
    count_first = isolated_db.query(AttendanceSessionModel).filter_by(tenant_id="tenant_1").count()
    assert count_first == 6

    # Re-run seed on the same day
    seed(db_session=isolated_db)
    count_second = isolated_db.query(AttendanceSessionModel).filter_by(tenant_id="tenant_1").count()
    assert count_second == 6


def test_all_six_classes_available_and_not_marked_immediately_after_login(isolated_db: Session):
    """All 6 daily classes are immediately available (NOT_MARKED) without wait times."""
    seed(db_session=isolated_db)
    token = _get_token(client, "student_001", "password123")
    headers = {"Authorization": f"Bearer {token}"}

    res = client.get("/api/v1/attendance/sessions", headers=headers)
    assert res.status_code == 200
    sessions = res.json()
    assert len(sessions) == 6

    for s in sessions:
        assert s["attendance_status"] == "NOT_MARKED"


@patch("src.application.attendance_service.validate_and_normalize_audio")
@patch("src.application.attendance_service.load_wav")
@patch("src.application.attendance_service.evaluate_attendance")
@patch("src.application.attendance_service.run_biometric_pipeline")
def test_independent_marking_and_duplicate_prevention(
    mock_pipeline, mock_eval, mock_load, mock_norm, isolated_db: Session
):
    """Marking one class transitions only that class to PRESENT and prevents duplicate marking."""
    seed(db_session=isolated_db)

    # Seed an active reference for student_001
    bio = io.BytesIO()
    np.save(bio, np.zeros(192, dtype=np.float32))
    isolated_db.add(
        ReferenceEmbeddingModel(
            id="ref_1",
            tenant_id="tenant_1",
            user_id="student_001",
            vector_blob=bio.getvalue(),
            dimension=192,
            model_id="speechbrain/spkrec-ecapa-voxceleb",
            is_active=True,
        )
    )
    isolated_db.commit()

    token = _get_token(client, "student_001", "password123")
    headers = {"Authorization": f"Bearer {token}"}

    mock_load.return_value = (16000, np.zeros(16000, dtype=np.float32))
    mock_norm.return_value = (16000, np.zeros(16000, dtype=np.float32))
    bio_res = BiometricResult(
        passed=True,
        reason=FailureReason.SUCCESS,
        similarity_score=0.85,
        liveness_score=0.99,
    )
    mock_pipeline.return_value = bio_res
    mock_eval.return_value = AttendanceDecision(
        marked=True, reason=FailureReason.SUCCESS, biometric_data=bio_res
    )

    # 1. Fetch sessions before marking
    sessions_before = client.get("/api/v1/attendance/sessions", headers=headers).json()
    math_session = next(s for s in sessions_before if s["class_name"] == "Mathematics")
    physics_session = next(s for s in sessions_before if s["class_name"] == "Physics")

    # Request challenge for Mathematics session
    chal_res = client.post(
        "/api/v1/attendance/challenge",
        data={"session_token": math_session["session_token"]},
        headers=headers,
    )
    assert chal_res.status_code == 200
    challenge_id = chal_res.json()["challenge_id"]

    # 2. Mark Mathematics
    mark_res = client.post(
        "/api/v1/attendance/mark",
        files={"audio_file": ("test.wav", b"fake_wav", "audio/wav")},
        data={
            "session_token": math_session["session_token"],
            "challenge_id": challenge_id,
        },
        headers=headers,
    )
    assert mark_res.status_code == 200
    assert mark_res.json()["is_present"] is True
    assert mark_res.json()["reason"] == "SUCCESS"

    # 3. Check session statuses: Mathematics is PRESENT, Physics is NOT_MARKED
    sessions_after = client.get("/api/v1/attendance/sessions", headers=headers).json()
    math_after = next(s for s in sessions_after if s["class_name"] == "Mathematics")
    physics_after = next(s for s in sessions_after if s["class_name"] == "Physics")
    assert math_after["attendance_status"] == "PRESENT"
    assert physics_after["attendance_status"] == "NOT_MARKED"

    # 4. Attempt to mark Mathematics again -> duplicate prevention (ALREADY_MARKED)
    chal_res2 = client.post(
        "/api/v1/attendance/challenge",
        data={"session_token": math_session["session_token"]},
        headers=headers,
    )
    assert chal_res2.status_code == 200
    challenge_id2 = chal_res2.json()["challenge_id"]

    dup_res = client.post(
        "/api/v1/attendance/mark",
        files={"audio_file": ("test.wav", b"fake_wav", "audio/wav")},
        data={
            "session_token": math_session["session_token"],
            "challenge_id": challenge_id2,
        },
        headers=headers,
    )
    assert dup_res.status_code == 200
    assert dup_res.json()["is_present"] is False
    assert dup_res.json()["reason"] == "ALREADY_MARKED"


def test_historical_sessions_and_records_preserved(isolated_db: Session):
    """Historical sessions and records from prior dates remain intact alongside new sessions."""
    # Add a historical session and attendance record from a past date
    past_session = AttendanceSessionModel(
        id="past_sess_1",
        tenant_id="tenant_1",
        session_token="class_math_20260917",
        class_name="Mathematics",
        starts_at=datetime(2026, 9, 17, 10, 0, 0),
        ends_at=datetime(2026, 9, 17, 11, 0, 0),
        is_active=True,
    )
    past_record = AttendanceRecord(
        id=9999,
        tenant_id="tenant_1",
        user_id="student_001",
        session_id="past_sess_1",
        timestamp=datetime(2026, 9, 17, 10, 15, 0),
    )
    isolated_db.add_all([past_session, past_record])
    isolated_db.commit()

    # Seed current day
    seed(db_session=isolated_db)

    # Verify historical data is untouched
    persisted_past_sess = isolated_db.query(AttendanceSessionModel).filter_by(id="past_sess_1").first()
    assert persisted_past_sess is not None
    assert persisted_past_sess.session_token == "class_math_20260917"

    persisted_past_rec = isolated_db.query(AttendanceRecord).filter_by(id=9999).first()
    assert persisted_past_rec is not None
    assert persisted_past_rec.session_id == "past_sess_1"
