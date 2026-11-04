from dataclasses import dataclass
import numpy as np
import torch
import shutil

# Monkeypatch pathlib symlink to bypass Windows privilege issue
import pathlib
if not hasattr(pathlib.Path, '_monkeypatched_for_speechbrain'):
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
    pathlib.Path._monkeypatched_for_speechbrain = True

from speechbrain.inference.speaker import EncoderClassifier

@dataclass
class SpeakerEmbedding:
    """
    A clean data representation of a speaker embedding.
    We avoid leaking PyTorch tensors through the architecture.
    """
    vector: np.ndarray
    model_id: str
    dimension: int
    
    def to_dict(self):
        return {
            "model_id": self.model_id,
            "dimension": self.dimension,
            "vector": self.vector.tolist()
        }

# Global cache for the classifier to avoid reloading it on every function call
_classifier = None
MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"

def _load_classifier():
    global _classifier
    if _classifier is None:
        _classifier = EncoderClassifier.from_hparams(
            source=MODEL_ID,
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device":"cpu"}
        )
    return _classifier

def extract_embedding(audio_data: np.ndarray, sample_rate: int = 16000) -> SpeakerEmbedding:
    """
    Extract a speaker embedding from a trimmed audio array.
    
    Args:
        audio_data (np.ndarray): The raw 1D audio array containing ONLY speech (VAD trimmed).
        sample_rate (int): The sample rate in Hz. Must be 16000.
        
    Returns:
        SpeakerEmbedding: A dataclass containing the numpy embedding vector and metadata.
    """
    if sample_rate != 16000:
        raise ValueError(f"ECAPA-TDNN expects a sample rate of 16000 Hz, got {sample_rate}")
        
    classifier = _load_classifier()
    
    # Preprocessing: Convert int16 to float32 if necessary
    if audio_data.dtype == np.int16:
        audio_float32 = audio_data.astype(np.float32) / 32768.0
    else:
        audio_float32 = audio_data.astype(np.float32)
        
    # Convert to PyTorch tensor and add batch dimension (1, N)
    audio_tensor = torch.from_numpy(audio_float32).unsqueeze(0)
    
    # Extract embedding
    # SpeechBrain returns a tensor of shape (batch, 1, channels) e.g., (1, 1, 192)
    with torch.no_grad():
        embeddings_tensor = classifier.encode_batch(audio_tensor)
        
    # Squeeze the batch and extra dimension to get a flat 1D vector (e.g., 192,)
    embedding_numpy = embeddings_tensor.squeeze().cpu().numpy()
    
    return SpeakerEmbedding(
        vector=embedding_numpy,
        model_id=MODEL_ID,
        dimension=len(embedding_numpy)
    )
