# Phase 11: Android Demo Client

This Android application is a demonstration client for the standalone voice biometric backend. It is not the production OneApp integration.

## Networking

- **Emulator setup**: By default, the app is configured to connect to `http://10.0.2.2:8000/`. This URL loops back to the host machine running the FastAPI backend.
- **Physical Device**: If you test on a physical device, update `API_BASE_URL` in `app/build.gradle.kts` to point to the LAN IP address of your host machine running the FastAPI server.

## Architecture

- **UI**: Jetpack Compose
- **Network**: Retrofit & OkHttp
- **Security**: EncryptedSharedPreferences is used for storing the JWT.
- **Audio Capture**: The app records 16-bit, 16kHz Mono PCM audio using `AudioRecord`, which is then mathematically converted into a standard 44-byte RIFF WAV file via the internal `WavEncoder`.
- **Backend as Source of Truth**: The Android Demo app does not perform any VAD, anti-spoof checks, or ECAPA-TDNN feature extraction. It merely displays the state returned by the backend.

## Permissions Required

The app requests `RECORD_AUDIO` and `INTERNET`. Ensure you grant the microphone permission when deploying the app.
