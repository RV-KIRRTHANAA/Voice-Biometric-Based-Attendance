import io
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import src.infrastructure.database as _db_module
from src.api.auth import create_access_token, get_password_hash
from src.audio import config
from src.audio.embedding import SpeakerEmbedding
from src.engine.results import FailureReason
from src.infrastructure.database import (
    AttendanceSessionModel,
    Base,
    Challenge,
    ReferenceEmbeddingModel,
    Tenant,
    User,
    get_db,
)

@pytest.fixture
def enable_challenge_response():
    """Temporarily enable CHALLENGE_RESPONSE_ENABLED for a test and always restore it."""
    original = config.CHALLENGE_RESPONSE_ENABLED
    config.CHALLENGE_RESPONSE_ENABLED = True
    try:
        yield
    finally:
        config.CHALLENGE_RESPONSE_ENABLED = original


needs_challenge = pytest.mark.usefixtures("enable_challenge_response")

# Prevent import-time init_db() in src.api.main from touching ./data/oneapp.db
with patch.object(_db_module, "init_db", lambda: None):
    from src.api.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def test_db_session(tmp_path):
    """Dedicated temporary SQLite database per test using tmp_path."""
    db_file = tmp_path / "test_challenge_response.db"
    test_engine = create_engine(
        f"sqlite:///{db_file}",
        connect_args={"check_same_thread": False},
    )
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=test_engine
    )
    Base.metadata.create_all(bind=test_engine)

    seed_db = TestingSessionLocal()
    seed_db.add(Tenant(id="tenant_1", name="Test University"))
    pw_hash = get_password_hash("password123")
    for sid in ("student_001", "student_002"):
        seed_db.add(
            User(
                id=sid,
                tenant_id="tenant_1",
                password_hash=pw_hash,
                display_name=sid,
            )
        )

    seed_db.add(
        AttendanceSessionModel(
            id="session_1",
            tenant_id="tenant_1",
            session_token="valid_session",
            class_name="Test Class",
            starts_at=datetime.utcnow() - timedelta(hours=1),
            ends_at=datetime.utcnow() + timedelta(hours=1),
            is_active=True,
        )
    )

    vec = np.ones(192, dtype=np.float32)
    bio = io.BytesIO()
    np.save(bio, vec)
    vec_bytes = bio.getvalue()
    for idx, sid in enumerate(("student_001", "student_002"), start=1):
        seed_db.add(
            ReferenceEmbeddingModel(
                id=f"ref_{idx}",
                user_id=sid,
                tenant_id="tenant_1",
                vector_blob=vec_bytes,
                model_id="speechbrain/spkrec-ecapa-voxceleb",
                dimension=192,
                is_active=True,
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


def get_auth_headers():
    response = client.post(
        "/api/v1/auth/token", data={"username": "student_001", "password": "password123"}
    )
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _create_session_and_enroll(client):
    pass


@pytest.fixture
def auth_headers():
    return get_auth_headers()


@needs_challenge
def test_generate_challenge(auth_headers):
    # 1. challenge generation
    res = client.post(
        "/api/v1/attendance/challenge",
        data={"session_token": "valid_session"},
        headers=auth_headers,
    )
    assert res.status_code == 200
    data = res.json()
    assert "challenge_id" in data
    assert "phrase" in data
    assert "present sir" in data["phrase"]


@needs_challenge
def test_tenant_isolation_wrong_session():
    # 4. tenant isolation / 5. wrong session rejection
    headers = get_auth_headers()  # student_001
    res = client.post(
        "/api/v1/attendance/challenge",
        data={"session_token": "invalid_session"},
        headers=headers,
    )
    assert res.status_code == 400


@needs_challenge
def test_already_used_challenge_rejection(auth_headers):
    # 6. already-used challenge rejection
    # Generate
    res = client.post(
        "/api/v1/attendance/challenge",
        data={"session_token": "valid_session"},
        headers=auth_headers,
    )
    c_id = res.json()["challenge_id"]
    phrase = res.json()["phrase"]

    # Mock everything to pass
    with patch("src.application.attendance_service.load_wav") as mock_load, \
         patch("src.engine.biometric_engine.detect_speech") as mock_vad, \
         patch("src.engine.biometric_engine.check_liveness") as mock_liveness, \
         patch("src.engine.biometric_engine.extract_embedding") as mock_extract, \
         patch("src.engine.biometric_engine.verify_speaker") as mock_verify, \
         patch("src.engine.biometric_engine.asr_service") as mock_asr:

        mock_load.return_value = (16000, np.zeros(16000))
        mock_vad.return_value = (np.zeros(16000), [])

        live = MagicMock()
        live.is_live = True
        live.score = 0.99
        live.threshold = 0.5
        mock_liveness.return_value = live

        mock_extract.return_value = SpeakerEmbedding(
            vector=np.ones(192, dtype=np.float32),
            model_id="speechbrain/spkrec-ecapa-voxceleb",
            dimension=192,
        )
        verify = MagicMock()
        verify.match = True
        verify.score = 0.85
        verify.threshold = 0.10
        mock_verify.return_value = verify

        mock_asr.transcribe.return_value = phrase  # Exact phrase passes

        files = {"audio_file": ("test.wav", b"fake audio", "audio/wav")}
        data = {"session_token": "valid_session", "challenge_id": c_id}

        # Request 1 (should pass)
        r1 = client.post(
            "/api/v1/attendance/mark", files=files, data=data, headers=auth_headers
        )
        assert r1.status_code == 200
        assert r1.json()["is_present"] == True

        # Request 2 (should fail because consumed)
        # Mock load wav again since files are consumed
        files2 = {"audio_file": ("test.wav", b"fake audio", "audio/wav")}
        r2 = client.post(
            "/api/v1/attendance/mark", files=files2, data=data, headers=auth_headers
        )
        assert r2.status_code == 200
        assert r2.json()["is_present"] == False
        assert r2.json()["reason"] == "CHALLENGE_USED"


@needs_challenge
def test_challenge_expiry(auth_headers, test_db_session: Session):
    # 2. challenge expiry
    res = client.post(
        "/api/v1/attendance/challenge",
        data={"session_token": "valid_session"},
        headers=auth_headers,
    )
    c_id = res.json()["challenge_id"]

    # Manually expire it in the isolated test DB
    challenge = test_db_session.query(Challenge).filter(Challenge.id == c_id).first()
    challenge.expires_at = datetime.utcnow() - timedelta(seconds=1)
    test_db_session.commit()

    files = {"audio_file": ("test.wav", b"fake audio", "audio/wav")}
    data = {"session_token": "valid_session", "challenge_id": c_id}
    r = client.post(
        "/api/v1/attendance/mark", files=files, data=data, headers=auth_headers
    )
    assert r.status_code == 200
    assert r.json()["is_present"] == False
    assert r.json()["reason"] == "CHALLENGE_EXPIRED"


@needs_challenge
def test_challenge_ownership():
    # 3. challenge ownership
    headers1 = get_auth_headers()  # student_001
    res = client.post(
        "/api/v1/attendance/challenge",
        data={"session_token": "valid_session"},
        headers=headers1,
    )
    c_id = res.json()["challenge_id"]

    # student_002 trying to use student_001's challenge
    res2 = client.post(
        "/api/v1/auth/token", data={"username": "student_002", "password": "password123"}
    )
    headers2 = {"Authorization": f"Bearer {res2.json()['access_token']}"}

    files = {"audio_file": ("test.wav", b"fake audio", "audio/wav")}
    data = {"session_token": "valid_session", "challenge_id": c_id}
    r = client.post(
        "/api/v1/attendance/mark", files=files, data=data, headers=headers2
    )
    assert r.status_code == 200
    assert r.json()["is_present"] == False
    assert r.json()["reason"] == "CHALLENGE_INVALID"


@needs_challenge
def test_normalized_phrase_passes(auth_headers):
    # 9. normalized equivalent phrase passes
    res = client.post(
        "/api/v1/attendance/challenge",
        data={"session_token": "valid_session"},
        headers=auth_headers,
    )
    c_id = res.json()["challenge_id"]
    phrase = res.json()["phrase"]  # e.g. "present sir one two three"

    # convert "one" to "1" etc for transcription simulation
    transcribed = phrase.replace("one", "1").replace("two", "2").upper() + "!"

    with patch("src.application.attendance_service.load_wav") as mock_load, \
         patch("src.engine.biometric_engine.detect_speech") as mock_vad, \
         patch("src.engine.biometric_engine.check_liveness") as mock_liveness, \
         patch("src.engine.biometric_engine.extract_embedding") as mock_extract, \
         patch("src.engine.biometric_engine.verify_speaker") as mock_verify, \
         patch("src.engine.biometric_engine.asr_service") as mock_asr:

        mock_load.return_value = (16000, np.zeros(16000))
        mock_vad.return_value = (np.zeros(16000), [])
        live = MagicMock()
        live.is_live = True
        live.score = 0.99
        live.threshold = 0.5
        mock_liveness.return_value = live
        mock_extract.return_value = SpeakerEmbedding(
            vector=np.ones(192, dtype=np.float32),
            model_id="speechbrain/spkrec-ecapa-voxceleb",
            dimension=192,
        )
        verify = MagicMock()
        verify.match = True
        verify.score = 0.85
        verify.threshold = 0.10
        mock_verify.return_value = verify

        mock_asr.transcribe.return_value = transcribed

        files = {"audio_file": ("test.wav", b"fake audio", "audio/wav")}
        data = {"session_token": "valid_session", "challenge_id": c_id}

        r = client.post(
            "/api/v1/attendance/mark", files=files, data=data, headers=auth_headers
        )
        assert r.json()["is_present"] == True


@needs_challenge
def test_phonetic_salutation_and_digit_matching(auth_headers):
    """Whisper phonetic variations of 'present sir' (e.g. 'resonator', 'resonsor') pass when numbers match."""
    from src.engine.challenge_verifier import ChallengeVerifier
    verifier = ChallengeVerifier()
    assert verifier.verify_challenge("present sir five one two", "resonator 512") is True
    assert verifier.verify_challenge("present sir nine four nine", "Resonsor 949") is True
    assert verifier.verify_challenge("present sir four zero seven", "present year 407") is True
    assert verifier.verify_challenge("present sir five one two", "5 1 2") is True


@needs_challenge
def test_wrong_phrase_fails(auth_headers):
    # 10. wrong phrase fails / 15. challenge FAIL => verification fails
    res = client.post(
        "/api/v1/attendance/challenge",
        data={"session_token": "valid_session"},
        headers=auth_headers,
    )
    c_id = res.json()["challenge_id"]

    with patch("src.application.attendance_service.load_wav") as mock_load, \
         patch("src.engine.biometric_engine.detect_speech") as mock_vad, \
         patch("src.engine.biometric_engine.check_liveness") as mock_liveness, \
         patch("src.engine.biometric_engine.extract_embedding") as mock_extract, \
         patch("src.engine.biometric_engine.verify_speaker") as mock_verify, \
         patch("src.engine.biometric_engine.asr_service") as mock_asr:

        mock_load.return_value = (16000, np.zeros(16000))
        mock_vad.return_value = (np.zeros(16000), [])
        live = MagicMock()
        live.is_live = True
        live.score = 0.99
        live.threshold = 0.5
        mock_liveness.return_value = live
        mock_extract.return_value = SpeakerEmbedding(
            vector=np.ones(192, dtype=np.float32),
            model_id="speechbrain/spkrec-ecapa-voxceleb",
            dimension=192,
        )
        verify = MagicMock()
        verify.match = True
        verify.score = 0.85
        verify.threshold = 0.10
        mock_verify.return_value = verify

        mock_asr.transcribe.return_value = "present sir totally wrong"

        files = {"audio_file": ("test.wav", b"fake audio", "audio/wav")}
        data = {"session_token": "valid_session", "challenge_id": c_id}

        r = client.post(
            "/api/v1/attendance/mark", files=files, data=data, headers=auth_headers
        )
        assert r.json()["is_present"] == False
        assert r.json()["reason"] == "CHALLENGE_MISMATCH"


@needs_challenge
def test_partial_phrase_fails(auth_headers):
    # 11. partial phrase fails
    res = client.post(
        "/api/v1/attendance/challenge",
        data={"session_token": "valid_session"},
        headers=auth_headers,
    )
    c_id = res.json()["challenge_id"]
    phrase = res.json()["phrase"]

    with patch("src.application.attendance_service.load_wav") as mock_load, \
         patch("src.engine.biometric_engine.detect_speech") as mock_vad, \
         patch("src.engine.biometric_engine.check_liveness") as mock_liveness, \
         patch("src.engine.biometric_engine.extract_embedding") as mock_extract, \
         patch("src.engine.biometric_engine.verify_speaker") as mock_verify, \
         patch("src.engine.biometric_engine.asr_service") as mock_asr:

        mock_load.return_value = (16000, np.zeros(16000))
        mock_vad.return_value = (np.zeros(16000), [])
        live = MagicMock()
        live.is_live = True
        live.score = 0.99
        live.threshold = 0.5
        mock_liveness.return_value = live
        mock_extract.return_value = SpeakerEmbedding(
            vector=np.ones(192, dtype=np.float32),
            model_id="speechbrain/spkrec-ecapa-voxceleb",
            dimension=192,
        )
        verify = MagicMock()
        verify.match = True
        verify.score = 0.85
        verify.threshold = 0.10
        mock_verify.return_value = verify

        # Only say "present sir"
        mock_asr.transcribe.return_value = "present sir"

        files = {"audio_file": ("test.wav", b"fake audio", "audio/wav")}
        data = {"session_token": "valid_session", "challenge_id": c_id}

        r = client.post(
            "/api/v1/attendance/mark", files=files, data=data, headers=auth_headers
        )
        assert r.json()["is_present"] == False
        assert r.json()["reason"] == "CHALLENGE_MISMATCH"


def test_disabled_challenge_response():
    """Verify attendance works WITHOUT challenge_id when CHALLENGE_RESPONSE_ENABLED is False.

    Uses an isolated in-memory DB. Mocks only expensive ML/model boundaries.
    Lets the real biometric pipeline and real attendance policy execute.
    Explicitly asserts ASR and ChallengeVerifier are NOT called.
    """
    # --- In-memory DB setup ---
    mem_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    MemSession = sessionmaker(autocommit=False, autoflush=False, bind=mem_engine)
    Base.metadata.create_all(bind=mem_engine)

    db = MemSession()
    tenant = Tenant(id="t1", name="TestTenant")
    db.add(tenant)
    user = User(id="u1", tenant_id="t1", password_hash=get_password_hash("pass123"))
    db.add(user)

    # Active session with valid time window
    session = AttendanceSessionModel(
        id="s1",
        tenant_id="t1",
        session_token="tok_disabled_test",
        class_name="Test Class",
        starts_at=datetime.utcnow() - timedelta(hours=1),
        ends_at=datetime.utcnow() + timedelta(hours=1),
        is_active=True,
    )
    db.add(session)

    # Active reference embedding for user u1/t1
    vec = np.ones(192, dtype=np.float32)
    bio = io.BytesIO()
    np.save(bio, vec)
    ref = ReferenceEmbeddingModel(
        id="ref1",
        user_id="u1",
        tenant_id="t1",
        vector_blob=bio.getvalue(),
        model_id="speechbrain/spkrec-ecapa-voxceleb",
        dimension=192,
        is_active=True,
    )
    db.add(ref)
    db.commit()
    db.close()

    # Override DB dependency
    from src.infrastructure.database import get_db as real_get_db

    def override_db():
        s = MemSession()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[real_get_db] = override_db

    # Auth token for user u1
    token = create_access_token(data={"sub": "u1", "tenant_id": "t1", "student_id": "u1"})
    headers = {"Authorization": f"Bearer {token}"}

    # Ensure challenge-response is disabled
    original = config.CHALLENGE_RESPONSE_ENABLED
    config.CHALLENGE_RESPONSE_ENABLED = False

    try:
        # Mock only the expensive ML/model boundaries
        with patch("src.application.attendance_service.load_wav") as mock_load, \
             patch("src.engine.biometric_engine.detect_speech") as mock_vad, \
             patch("src.engine.biometric_engine.check_liveness") as mock_liveness, \
             patch("src.engine.biometric_engine.extract_embedding") as mock_embed, \
             patch("src.engine.biometric_engine.verify_speaker") as mock_verify, \
             patch("src.engine.biometric_engine.asr_service") as mock_asr, \
             patch("src.engine.biometric_engine.challenge_verifier") as mock_chall:

            # VAD returns trimmed audio (>= 0.5s at 16kHz = 8000 samples)
            trimmed = np.zeros(16000, dtype=np.float32)
            mock_load.return_value = (16000, trimmed)
            mock_vad.return_value = (trimmed, [])

            # Anti-spoof passes
            liveness_mock = MagicMock()
            liveness_mock.is_live = True
            liveness_mock.score = 0.99
            liveness_mock.threshold = 0.5
            mock_liveness.return_value = liveness_mock

            # Embedding extraction returns a valid embedding
            mock_embed.return_value = SpeakerEmbedding(
                vector=np.ones(192), model_id="speechbrain/spkrec-ecapa-voxceleb", dimension=192
            )

            # Speaker verification passes
            verify_mock = MagicMock()
            verify_mock.match = True
            verify_mock.score = 0.85
            verify_mock.threshold = 0.10
            mock_verify.return_value = verify_mock

            # POST /mark WITHOUT challenge_id
            files = {"audio_file": ("test.wav", b"fake audio data", "audio/wav")}
            data = {"session_token": "tok_disabled_test"}  # No challenge_id

            res = client.post("/api/v1/attendance/mark", files=files, data=data, headers=headers)

            assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.json()}"
            body = res.json()
            assert body["is_present"] is True, f"Expected is_present=True, got: {body}"

            # ASR must NOT have been called
            mock_asr.transcribe.assert_not_called()

            # ChallengeVerifier must NOT have been called
            mock_chall.verify_challenge.assert_not_called()

    finally:
        config.CHALLENGE_RESPONSE_ENABLED = original
        app.dependency_overrides.clear()
        Base.metadata.drop_all(bind=mem_engine)
