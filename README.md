# OneApp: Voice Biometric Attendance System (POC)

An end-to-end voice biometric attendance platform featuring real-time speaker verification, anti-spoofing detection, dynamic challenge-response against replay/relay attacks, and an accompanying Android demo client.

---

## 1. System Architecture & Tech Stack

```
   ┌─────────────────────────────────────────────────────────┐
   │                  Android Demo Client                    │
   │  (Jetpack Compose · OkHttp / Retrofit · 16kHz PCM WAV)  │
   └────────────────────────────┬────────────────────────────┘
                                │ HTTPS / REST (JSON + Multipart)
                                ▼
   ┌─────────────────────────────────────────────────────────┐
   │                 FastAPI Backend Service                 │
   │  (JWT Auth · Session Management · Attendance Pipeline)  │
   └──────┬──────────────────────┬────────────────────┬──────┘
          │                      │                    │
          ▼                      ▼                    ▼
   ┌───────────────┐     ┌───────────────┐    ┌───────────────┐
   │ Silero VAD    │     │ SpeechBrain   │    │ OpenAI        │
   │ Speech Bounds │     │ ECAPA-TDNN    │    │ Whisper (ASR) │
   │ & Silence Cut │     │ Embeddings    │    │ Challenge Verifier
   └───────────────┘     └───────┬───────┘    └───────────────┘
                                 │
                                 ▼
                         Cosine Similarity
                        Threshold: 0.45
```

- **Backend**: Python 3.10+, FastAPI, Uvicorn, SQLAlchemy, SQLite.
- **Biometrics & Speech**:
  - **Speaker Verification**: SpeechBrain ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`, 192-dimensional embeddings).
  - **Voice Activity Detection**: Silero VAD (neural boundary detector, 16 kHz).
  - **ASR & Challenge Verification**: OpenAI Whisper (`base.en` / `base`), regex phonetic normalizer.
  - **Anti-Spoofing / Liveness**: AST (Audio Spectrogram Transformer) for synthetic speech detection.
- **Android Demo Client**: Kotlin, Jetpack Compose, Coroutines, StateFlow, Retrofit2, OkHttp3, EncryptedSharedPreferences.

---

## 2. Pretrained Models & Automatic Retrieval

Pretrained weights are **excluded from the Git repository** to prevent repository bloat. Models are automatically downloaded and cached on first execution:

| Model | Library / Source | Local Cache Destination | Automatic Fetch Mechanism |
| :--- | :--- | :--- | :--- |
| **ECAPA-TDNN** | `speechbrain/spkrec-ecapa-voxceleb` | `pretrained_models/spkrec-ecapa-voxceleb/` | Downloaded via Hugging Face Hub during `EncoderClassifier.from_hparams()` |
| **Silero VAD** | `snakers4/silero-vad` | `~/.cache/torch/hub/` | Downloaded via `torch.hub.load()` |
| **Whisper ASR** | `openai-whisper` (`base`) | `~/.cache/whisper/` | Downloaded during `whisper.load_model("base")` |
| **AST Anti-Spoof** | Hugging Face Hub | `~/.cache/huggingface/` | Downloaded via Hugging Face Transformers |

No manual model placement is required. Running the test suite or starting the server will automatically download the necessary checkpoints on first run.

---

## 3. Backend Setup & Configuration

### Prerequisites
- Python 3.10 or 3.11
- FFmpeg (required by Whisper for audio decoding)

### Setup Steps

1. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # Windows PowerShell:
   .\venv\Scripts\Activate.ps1
   # Linux/macOS:
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Copy the example environment configuration:
   ```bash
   cp .env.example .env
   ```
   Set a secure random secret key in `.env`:
   ```env
   JWT_SECRET_KEY=generate_a_random_32_character_hex_or_alphanumeric_string
   ```
   *(Alternatively, export `$env:JWT_SECRET_KEY="your_secret"` in PowerShell or `export JWT_SECRET_KEY="your_secret"` in Bash).*

4. **Initialize Database & Demo Sessions**:
   ```bash
   python scripts/seed_dev_data.py
   ```

5. **Start the API Server**:
   ```bash
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000
   ```
   The backend API will be available at `http://localhost:8000`. API docs are at `http://localhost:8000/docs`.

---

## 4. Demo Data & Class Scheduling

The development seeder (`scripts/seed_dev_data.py`) provides an idempotent demo configuration:

- **6 Daily Demo Classes**:
  1. Mathematics
  2. Physics
  3. Chemistry
  4. Computer Science
  5. English
  6. Voice Biometrics Lab
- **Calendar-Day Availability**: All six classes remain available throughout the local calendar day immediately after login, with no mandatory waiting delays between classes.
- **Independent Attendance & Duplicate Prevention**: Marking attendance for one subject does not affect others. Repeated attempts to mark the same class on the same day return `ALREADY_MARKED`.
- **Demo Credentials (Development Only)**:
  - Default user accounts: `student_001` through `student_005`.
  - Default password: `password123`.
  - *Warning*: Demo credentials must never be reused in production deployments.

---

## 5. Voice Biometrics & Verification Pipeline

1. **Enrollment**:
   - The student records 3 genuine audio samples via the Android app or `/api/v1/enrollment/`.
   - Each sample is trimmed via Silero VAD and evaluated for speech quality and liveness.
   - An average 192-dimensional reference embedding vector is stored in `reference_embeddings`.
2. **Verification & Similarity Threshold**:
   - Audio is converted to an embedding and compared against the student's stored reference using cosine distance.
   - **Current Verified Threshold**:
     $$\text{similarity} = 1.0 - \text{cosine\_distance}(\vec{v}_{\text{ref}}, \vec{v}_{\text{cur}})$$
     Configured in `src/audio/config.py`:
     ```python
     DEFAULT_SIMILARITY_THRESHOLD = 0.45
     ```
   - **Calibration Rationale**:
     - Genuine speakers consistently score **$0.50 - 0.65$**.
     - Siblings and close-sounding voices score between **$0.30 - 0.40$**.
     - Unrelated impostors score below **$0.25$**.
     - A threshold of `0.45` safely accepts genuine students while rejecting siblings and impostor voices.

---

## 6. Dynamic Challenge-Response & Replay Mitigation

To prevent pre-recorded audio attacks and remote proxy-forwarding (e.g. students sending audio recordings to friends in class over WhatsApp):

1. **Ephemeral Challenge**: The student requests a challenge token. The server generates a random salutation and 3-digit sequence (e.g., *"Present sir four two eight"*).
2. **10-Second Time-To-Live (TTL)**:
   - Configured via `CHALLENGE_TTL_SECONDS = 10` in `src/audio/config.py`.
   - The Android client displays an animated 10-second countdown bar that turns red in the final 3 seconds.
   - The recording window is constrained to 4 seconds.
   - Audio submitted after 10 seconds is rejected with `CHALLENGE_EXPIRED`.
3. **Whisper ASR Verification**: The server transcribes the audio using OpenAI Whisper, normalizes phonetics and salutations, and strictly matches the expected digit sequence.

---

## 7. Security Analysis & Spoofing Limitations

### Synthetic Speech Detection (Logical Access)
- The backend incorporates an Audio Spectrogram Transformer (AST) to detect synthetic voices, text-to-speech (TTS), and voice conversion artifacts.

### Physical Acoustic Replay Limitation
- > [!WARNING]
  > **Physical replay detection is NOT solved by spectral liveness models alone.** Playing a genuine student's pre-recorded voice through an external smartphone or loudspeaker into the microphone can match both the speaker identity and synthetic liveness checks.
- Protection against physical replay relies on the **ephemeral dynamic challenge-response**:
  - Because the random phrase expires in **10 seconds**, an absent student cannot receive the phrase, record it remotely, and send it back to a friend before the challenge expires.

---

## 8. Geofencing Implementation Status

> [!IMPORTANT]
> **Geofencing is NOT implemented in this POC.**
> - Neither the backend nor the Android app currently performs GPS coordinate validation, classroom Wi-Fi BSSID checking, or Bluetooth Low Energy (BLE) beacon detection.
> - A student who possesses valid credentials and an internet connection could theoretically mark attendance from outside the classroom.
> - **Production Requirement**: A production deployment must integrate presence verification (such as classroom Wi-Fi AP subnet / BSSID validation or BLE beacon proximity) in tandem with voice biometrics to ensure physical presence.

---

## 9. Android Demo Setup

1. Open the `android-demo/` directory in **Android Studio Hedgehog (or newer)**.
2. In `android-demo/local.properties`, configure your development machine's IP address:
   ```properties
   api.base.url=http://192.168.1.X:8000/
   ```
   *(For Android Emulator loopback to host, use `http://10.0.2.2:8000/`).*
3. Build and run the app:
   ```bash
   cd android-demo
   .\gradlew.bat assembleDebug
   ```
4. Output APK location: `android-demo/app/build/outputs/apk/debug/app-debug.apk`.

---

## 10. Running Tests

Execute the full backend automated test suite:
```bash
pytest tests/
```
All 105 automated unit and integration tests run deterministically with mocked models.

---

## 11. Known POC Limitations

1. **Absence of Geofencing**: Physical classroom presence must be confirmed via secondary mechanisms (Wi-Fi BSSID, BLE beacons, or GPS).
2. **Acoustic Noise**: In loud classroom environments, ambient chatter can degrade ASR digit transcription and speaker embedding quality.
3. **Database Engine**: Uses SQLite (`data/oneapp.db`) for POC local testing; production requires PostgreSQL with connection pooling.
4. **Single-Node Execution**: Scaled production deployments require distributing Whisper and ECAPA-TDNN inference via asynchronous task queues (e.g., Celery / Redis / Ray).
