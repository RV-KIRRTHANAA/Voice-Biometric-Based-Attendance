import torch
import whisper
import numpy as np

class ASRService:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ASRService, cls).__new__(cls)
        return cls._instance

    def load_model(self):
        if self._model is None:
            # use base or tiny model for speed in POC
            # device mapping
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = whisper.load_model("tiny", device=device)
        return self._model

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribes numpy array audio to text using Whisper.
        Whisper expects 16kHz 1D audio as float32 in range [-1.0, 1.0].
        """
        if sample_rate != 16000:
            raise ValueError(f"Whisper ASR expects a sample rate of 16000 Hz, got {sample_rate}")

        if not isinstance(audio, np.ndarray):
            raise ValueError("Audio input must be a numpy ndarray.")

        if audio.ndim == 2 and (audio.shape[0] == 1 or audio.shape[1] == 1):
            audio = audio.squeeze()
        elif audio.ndim != 1:
            raise ValueError(f"Whisper ASR expects a 1D mono audio array, got shape {audio.shape}")

        if audio.size == 0:
            raise ValueError("Audio input is empty.")

        if not np.all(np.isfinite(audio)):
            raise ValueError("Audio input contains NaN or infinite values.")

        # Convert dtype to float32 in [-1.0, 1.0] BEFORE losing int16 type metadata
        if audio.dtype == np.int16:
            audio_float32 = audio.astype(np.float32) / 32768.0
        elif np.issubdtype(audio.dtype, np.floating):
            max_abs = float(np.max(np.abs(audio)))
            if max_abs > 1.05:
                raise ValueError(
                    f"Floating-point audio must be normalized to [-1.0, 1.0], got max amplitude {max_abs}"
                )
            audio_float32 = np.clip(audio.astype(np.float32), -1.0, 1.0)
        else:
            raise ValueError(f"Unsupported audio dtype for Whisper ASR: {audio.dtype}")

        model = self.load_model()

        # whisper.transcribe takes a 1D float32 numpy array or torch tensor
        result = model.transcribe(
            audio_float32,
            language="en",
            initial_prompt="Present sir, zero, one, two, three, four, five, six, seven, eight, nine.",
            fp16=torch.cuda.is_available(),
        )
        return result["text"].strip()
