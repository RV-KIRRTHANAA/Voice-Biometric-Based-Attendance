from dataclasses import dataclass
import numpy as np
import torch
import warnings

# Suppress symlink warnings from huggingface_hub during testing
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Monkeypatch pathlib symlink to bypass Windows privilege issue
import pathlib
import shutil
if not hasattr(pathlib.Path, '_monkeypatched_for_hf'):
    _original_symlink_to = pathlib.Path.symlink_to

    def safe_symlink_to(self, target, target_is_directory=False):
        try:
            _original_symlink_to(self, target, target_is_directory)
        except OSError:
            # Fallback to copy if symlink fails
            if self.exists():
                self.unlink()
            if target_is_directory or target.is_dir():
                shutil.copytree(target, self)
            else:
                shutil.copy2(target, self)

    pathlib.Path.symlink_to = safe_symlink_to
    pathlib.Path._monkeypatched_for_hf = True

from transformers import ASTFeatureExtractor, ASTForAudioClassification
from src.audio.config import DEFAULT_LIVENESS_THRESHOLD

MODEL_ID = "MattyB95/AST-ASVspoof2019-Synthetic-Voice-Detection"

@dataclass
class LivenessResult:
    """Result of an anti-spoofing/liveness check."""
    is_live: bool
    score: float
    threshold: float
    model_id: str

# Global cache for the liveness model
_feature_extractor = None
_liveness_model = None

def _load_liveness_model():
    global _feature_extractor, _liveness_model
    if _feature_extractor is None:
        _feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_ID)
    if _liveness_model is None:
        _liveness_model = ASTForAudioClassification.from_pretrained(MODEL_ID)
        _liveness_model.eval()
    return _feature_extractor, _liveness_model

def check_liveness(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    threshold: float = DEFAULT_LIVENESS_THRESHOLD
) -> LivenessResult:
    """
    Check if the provided audio array is live (genuine) or spoofed (e.g. replay attack, synthetic).
    
    Args:
        audio_data (np.ndarray): The 1D audio array (preferably trimmed speech).
        sample_rate (int): Must be 16000 Hz.
        threshold (float): Score >= threshold means LIVE.
        
    Returns:
        LivenessResult: Contains the boolean decision, the exact LIVE score, and metadata.
    """
    if sample_rate != 16000:
        raise ValueError(f"AST model expects a sample rate of 16000 Hz, got {sample_rate}")
        
    extractor, model = _load_liveness_model()
    
    # Preprocessing: ASTFeatureExtractor expects a float array in [-1.0, 1.0]
    if audio_data.dtype == np.int16:
        audio_float32 = audio_data.astype(np.float32) / 32768.0
    else:
        audio_float32 = audio_data.astype(np.float32)
        
    # Extract Mel-spectrogram features
    inputs = extractor(audio_float32, sampling_rate=sample_rate, return_tensors="pt")
    
    # Inference
    with torch.no_grad():
        logits = model(**inputs).logits
        
    # Convert logits to probabilities using Softmax
    probs = torch.nn.functional.softmax(logits, dim=-1)
    
    # Based on the model config id2label: {'0': 'Bonafide', '1': 'Spoof'}
    # So probs[0][0] is the probability of LIVE (Bonafide)
    live_score = float(probs[0][0].item())
    
    is_live = bool(live_score >= threshold)
    
    return LivenessResult(
        is_live=is_live,
        score=live_score,
        threshold=float(threshold),
        model_id=MODEL_ID
    )
