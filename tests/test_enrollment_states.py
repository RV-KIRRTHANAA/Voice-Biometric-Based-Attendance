import io
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

import src.infrastructure.database as _db_module
from src.api.auth import create_access_token, get_password_hash
from src.audio.embedding import SpeakerEmbedding
from src.engine.results import AttendanceDecision, BiometricResult, EnrollmentStatus, FailureReason
from src.infrastructure.database import (
    AttendanceSessionModel,
    Base,
    Enrollment,
    EnrollmentSample,
    ReferenceEmbeddingModel,
    Tenant,
    User,
    get_db,
)

# Prevent import-time init_db() in src.api.main from touching ./data/oneapp.db
with patch.object(_db_module, "init_db", lambda: None):
    from src.api.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def test_db_session(tmp_path):
    """Dedicated temporary SQLite database per test using tmp_path."""
    db_file = tmp_path / "test_enrollment_states.db"
    test_engine = create_engine(
        f"sqlite:///{db_file}",
        connect_args={"check_same_thread": False},
    )
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=test_engine
    )
    Base.metadata.create_all(bind=test_engine)

    # Seed isolated test tenants and students
    seed_db = TestingSessionLocal()
    seed_db.add(Tenant(id="tenant_1", name="Test University"))
    seed_db.add(Tenant(id="tenant_2", name="Other University"))
    pw_hash = get_password_hash("password123")
    for sid in ("student_001", "student_004", "student_005"):
        seed_db.add(
            User(
                id=sid,
                tenant_id="tenant_1",
                password_hash=pw_hash,
                display_name=sid,
            )
        )
    seed_db.add(
        User(
            id="student_other_tenant",
            tenant_id="tenant_2",
            password_hash=pw_hash,
            display_name="Other Tenant Student",
        )
    )
    seed_db.commit()
    seed_db.close()

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


def _clear_enrollments(db: Session, user_id: str):
    db.query(Enrollment).filter(Enrollment.user_id == user_id).delete()
    db.query(ReferenceEmbeddingModel).filter(ReferenceEmbeddingModel.user_id == user_id).delete()
    db.commit()


def test_user_with_no_reference_is_not_enrolled(test_db_session: Session):
    token = _get_token(client, "student_005", "password123")

    # ensure clean state in isolated test DB
    _clear_enrollments(test_db_session, "student_005")

    response = client.get("/api/v1/enrollment/status", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["status"] == "NOT_ENROLLED"
    assert response.json()["has_active_reference"] is False


def test_starting_enrollment_creates_one_in_progress(test_db_session: Session):
    token = _get_token(client, "student_005", "password123")
    response = client.post("/api/v1/enrollment/start", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["status"] == "IN_PROGRESS"
    assert response.json()["has_active_reference"] is False
    assert test_db_session.query(Enrollment).filter(Enrollment.user_id == "student_005").count() == 1


def test_starting_enrollment_again_does_not_create_another(test_db_session: Session):
    token = _get_token(client, "student_005", "password123")
    first_resp = client.post("/api/v1/enrollment/start", headers={"Authorization": f"Bearer {token}"})
    assert first_resp.status_code == 200
    assert first_resp.json()["status"] == "IN_PROGRESS"

    second_resp = client.post("/api/v1/enrollment/start", headers={"Authorization": f"Bearer {token}"})
    assert second_resp.status_code == 200
    assert second_resp.json()["status"] == "IN_PROGRESS"
    assert test_db_session.query(Enrollment).filter(Enrollment.user_id == "student_005").count() == 1


def test_tenant_isolation_enrollment_status(test_db_session: Session):
    token1 = _get_token(client, "student_005", "password123")
    start_resp = client.post("/api/v1/enrollment/start", headers={"Authorization": f"Bearer {token1}"})
    assert start_resp.status_code == 200

    response1 = client.get("/api/v1/enrollment/status", headers={"Authorization": f"Bearer {token1}"})
    assert response1.json()["status"] == "IN_PROGRESS"

    token2 = _get_token(client, "student_004", "password123")
    _clear_enrollments(test_db_session, "student_004")

    response2 = client.get("/api/v1/enrollment/status", headers={"Authorization": f"Bearer {token2}"})
    assert response2.json()["status"] == "NOT_ENROLLED"


@patch("src.application.enrollment_service.load_wav")
@patch("src.application.enrollment_service.detect_speech")
@patch("src.application.enrollment_service.check_liveness")
@patch("src.application.enrollment_service.extract_embedding")
def test_already_enrolled_student_can_start_reenrollment_and_keep_old_ref_active(
    mock_extract, mock_liveness, mock_vad, mock_load, test_db_session: Session
):
    """Verify an enrolled student can start re-enrollment while old active reference stays active."""
    mock_load.return_value = (16000, np.zeros(16000))
    mock_vad.return_value = (np.zeros(16000), [])
    liveness_mock = MagicMock()
    liveness_mock.is_live = True
    liveness_mock.score = 0.99
    mock_liveness.return_value = liveness_mock
    mock_extract.return_value = SpeakerEmbedding(
        vector=np.ones(192, dtype=np.float32),
        model_id="speechbrain/spkrec-ecapa-voxceleb",
        dimension=192,
    )

    token = _get_token(client, "student_005", "password123")
    headers = {"Authorization": f"Bearer {token}"}

    # 1. First-time enrollment completes
    assert client.post("/api/v1/enrollment/start", headers=headers).status_code == 200
    for _ in range(3):
        r = client.post(
            "/api/v1/enrollment/",
            files={"audio_file": ("sample.wav", b"wavbytes", "audio/wav")},
            headers=headers,
        )
        assert r.status_code == 200
    assert r.json()["status"] == "ENROLLED"
    assert r.json()["has_active_reference"] is True

    status_resp = client.get("/api/v1/enrollment/status", headers=headers)
    assert status_resp.json()["status"] == "ENROLLED"
    assert status_resp.json()["has_active_reference"] is True

    first_ref = (
        test_db_session.query(ReferenceEmbeddingModel)
        .filter_by(user_id="student_005", tenant_id="tenant_1")
        .one()
    )
    assert first_ref.is_active is True

    # 2. Initiate re-enrollment
    reenroll_start = client.post("/api/v1/enrollment/start", headers=headers)
    assert reenroll_start.status_code == 200
    assert reenroll_start.json()["status"] == "IN_PROGRESS"
    assert reenroll_start.json()["accepted_samples"] == 0
    assert reenroll_start.json()["has_active_reference"] is True

    status_during = client.get("/api/v1/enrollment/status", headers=headers)
    assert status_during.status_code == 200
    assert status_during.json()["status"] == "IN_PROGRESS"
    assert status_during.json()["has_active_reference"] is True

    # Old reference must still exist and remain active while re-enrollment is in progress
    test_db_session.expire_all()
    refs_during = (
        test_db_session.query(ReferenceEmbeddingModel)
        .filter_by(user_id="student_005", tenant_id="tenant_1")
        .all()
    )
    assert len(refs_during) == 1
    assert refs_during[0].id == first_ref.id
    assert refs_during[0].is_active is True


@patch("src.application.attendance_service.load_wav")
@patch("src.application.attendance_service.evaluate_attendance")
@patch("src.application.attendance_service.run_biometric_pipeline")
@patch("src.application.enrollment_service.load_wav")
@patch("src.application.enrollment_service.detect_speech")
@patch("src.application.enrollment_service.check_liveness")
@patch("src.application.enrollment_service.extract_embedding")
def test_abandoned_reenrollment_preserves_old_reference_for_attendance(
    mock_extract,
    mock_liveness,
    mock_vad,
    mock_enroll_load,
    mock_run_pipeline,
    mock_eval,
    mock_att_load,
    test_db_session: Session,
):
    """If re-enrollment is started and partially completed/abandoned, old reference still works for attendance."""
    mock_enroll_load.return_value = (16000, np.zeros(16000))
    mock_vad.return_value = (np.zeros(16000), [])
    liveness_mock = MagicMock()
    liveness_mock.is_live = True
    liveness_mock.score = 0.99
    mock_liveness.return_value = liveness_mock
    mock_extract.return_value = SpeakerEmbedding(
        vector=np.ones(192, dtype=np.float32),
        model_id="speechbrain/spkrec-ecapa-voxceleb",
        dimension=192,
    )

    token = _get_token(client, "student_005", "password123")
    headers = {"Authorization": f"Bearer {token}"}

    # Complete initial enrollment
    client.post("/api/v1/enrollment/start", headers=headers)
    for _ in range(3):
        client.post(
            "/api/v1/enrollment/",
            files={"audio_file": ("sample.wav", b"wavbytes", "audio/wav")},
            headers=headers,
        )

    # Start re-enrollment, submit only 1 of 3 samples, and abandon it
    start_re = client.post("/api/v1/enrollment/start", headers=headers)
    assert start_re.status_code == 200
    partial_res = client.post(
        "/api/v1/enrollment/",
        files={"audio_file": ("sample.wav", b"wavbytes", "audio/wav")},
        headers=headers,
    )
    assert partial_res.json()["status"] == "SAMPLE_ACCEPTED"
    assert partial_res.json()["accepted_samples"] == 1

    # Duplicate /start call while IN_PROGRESS must preserve accepted_samples == 1 and not create another session
    dup_start = client.post("/api/v1/enrollment/start", headers=headers)
    assert dup_start.status_code == 200
    assert dup_start.json()["status"] == "IN_PROGRESS"
    assert dup_start.json()["accepted_samples"] == 1
    in_prog_count = (
        test_db_session.query(Enrollment)
        .filter_by(
            user_id="student_005",
            tenant_id="tenant_1",
            status=EnrollmentStatus.IN_PROGRESS.value,
        )
        .count()
    )
    assert in_prog_count == 1

    # Create an active attendance session and verify student_005 can still mark attendance using the old active reference
    test_db_session.add(
        AttendanceSessionModel(
            id="att_sess_1",
            tenant_id="tenant_1",
            session_token="active_tok_1",
            class_name="Math",
            starts_at=datetime.utcnow() - timedelta(hours=1),
            ends_at=datetime.utcnow() + timedelta(hours=1),
            is_active=True,
        )
    )
    test_db_session.commit()

    mock_att_load.return_value = (16000, np.zeros(16000))
    bio_res = BiometricResult(
        passed=True,
        reason=FailureReason.SUCCESS,
        liveness_score=0.99,
        similarity_score=0.90,
    )
    mock_run_pipeline.return_value = bio_res
    mock_eval.return_value = AttendanceDecision(
        marked=True, reason=FailureReason.SUCCESS, biometric_data=bio_res
    )

    ch_resp = client.post(
        "/api/v1/attendance/challenge",
        data={"session_token": "active_tok_1"},
        headers=headers,
    )
    challenge_id = ch_resp.json()["challenge_id"]

    att_resp = client.post(
        "/api/v1/attendance/mark",
        files={"audio_file": ("att.wav", b"wavbytes", "audio/wav")},
        data={"session_token": "active_tok_1", "challenge_id": challenge_id},
        headers=headers,
    )
    assert att_resp.status_code == 200
    assert att_resp.json()["is_present"] is True


def test_cross_tenant_enrollment_isolation(test_db_session: Session):
    """A student in tenant_2 cannot access or affect an enrollment in tenant_1, even with spoofed JWT user_id."""
    # student_005 starts enrollment in tenant_1
    token_t1 = _get_token(client, "student_005", "password123")
    headers_t1 = {"Authorization": f"Bearer {token_t1}"}
    assert client.post("/api/v1/enrollment/start", headers=headers_t1).status_code == 200

    # Forge a token for the same user_id ("student_005") but scoped to tenant_2
    token_t2 = create_access_token(
        data={"sub": "student_005", "tenant_id": "tenant_2", "student_id": "student_005"}
    )
    headers_t2 = {"Authorization": f"Bearer {token_t2}"}

    # Tenant 2 must see NOT_ENROLLED and must fail sample upload because no active enrollment exists in tenant_2
    status_t2 = client.get("/api/v1/enrollment/status", headers=headers_t2)
    assert status_t2.status_code == 200
    assert status_t2.json()["status"] == "NOT_ENROLLED"

    with patch("src.application.enrollment_service.load_wav", return_value=(16000, np.zeros(16000))):
        upload_t2 = client.post(
            "/api/v1/enrollment/",
            files={"audio_file": ("sample.wav", b"wavbytes", "audio/wav")},
            headers=headers_t2,
        )
        assert upload_t2.status_code == 400
        assert "No active enrollment in progress" in upload_t2.json()["detail"]


@patch("src.application.enrollment_service.load_wav")
@patch("src.application.enrollment_service.detect_speech")
@patch("src.application.enrollment_service.check_liveness")
@patch("src.application.enrollment_service.extract_embedding")
def test_enrollment_resamples_48khz_and_rejects_unsupported_sample_rate(
    mock_extract, mock_liveness, mock_vad, mock_load, test_db_session: Session
):
    """EnrollmentService resamples supported non-16kHz WAVs to 16kHz and safely rejects unsupported sample rates."""
    token = _get_token(client, "student_005", "password123")
    headers = {"Authorization": f"Bearer {token}"}
    assert client.post("/api/v1/enrollment/start", headers=headers).status_code == 200

    # 1. Unsupported sample rate (e.g., 4000 Hz) must be safely caught and rejected without incrementing accepted_samples
    mock_load.return_value = (4000, np.zeros(4000, dtype=np.int16))
    bad_resp = client.post(
        "/api/v1/enrollment/",
        files={"audio_file": ("bad_sr.wav", b"wavbytes", "audio/wav")},
        headers=headers,
    )
    assert bad_resp.status_code == 200
    assert bad_resp.json()["status"] == "SAMPLE_REJECTED"
    assert bad_resp.json()["reason"] == FailureReason.MODEL_ERROR.value
    assert bad_resp.json()["accepted_samples"] == 0

    # 2. Valid 48kHz mono int16 WAV (1 second = 48000 samples) must be resampled to 16kHz (16000 samples)
    mock_load.return_value = (48000, np.zeros(48000, dtype=np.int16))
    mock_vad.return_value = (np.zeros(16000, dtype=np.int16), [])
    liveness_mock = MagicMock()
    liveness_mock.is_live = True
    liveness_mock.score = 0.99
    mock_liveness.return_value = liveness_mock
    mock_extract.return_value = SpeakerEmbedding(
        vector=np.ones(192, dtype=np.float32),
        model_id="speechbrain/spkrec-ecapa-voxceleb",
        dimension=192,
    )

    good_resp = client.post(
        "/api/v1/enrollment/",
        files={"audio_file": ("sr48k.wav", b"wavbytes", "audio/wav")},
        headers=headers,
    )
    assert good_resp.status_code == 200
    assert good_resp.json()["status"] == "SAMPLE_ACCEPTED"
    assert good_resp.json()["accepted_samples"] == 1
    passed_audio, passed_sr = mock_vad.call_args[0]
    assert passed_sr == 16000
    assert len(passed_audio) == 16000

