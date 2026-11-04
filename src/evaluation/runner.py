import time
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

from src.audio import config
from src.audio.wav_io import load_wav
from src.audio.vad import detect_speech
from src.audio.liveness import check_liveness
from src.audio.embedding import extract_embedding, SpeakerEmbedding
from src.audio.verification import verify_speaker
from src.audio.exceptions import NoSpeechDetectedError
from src.engine.aggregation import MeanEmbeddingAggregator
from src.engine.results import FailureReason

from src.evaluation.dataset import EvalSample
from src.evaluation.quality import analyze_quality, AudioQualityMetrics
from src.evaluation.latency import LatencyTracker

@dataclass
class EvaluationResultRow:
    sample_id: str
    speaker_id: str
    sample_type: str
    reference_id: Optional[str]
    condition: str
    audio_duration: float
    rms_level: float
    clipping_ratio: float
    vad_detected: bool
    antispoof_score: Optional[float]
    similarity_score: Optional[float]
    speaker_model_id: str
    antispoof_model_id: str
    # Latency tracking
    vad_ms: float
    liveness_ms: float
    embedding_ms: float
    verification_ms: float
    total_ms: float
    # These represent the raw engine output (baseline threshold)
    baseline_pass: bool
    baseline_reason: str

class EvaluationRunner:
    def __init__(self, samples: List[EvalSample]):
        self.samples = samples
        self.reference_cache: Dict[str, SpeakerEmbedding] = {}
        self._raw_ref_embeddings: Dict[str, Optional[SpeakerEmbedding]] = {}
        self.aggregator = MeanEmbeddingAggregator()
        
    def _precache_references(self):
        print("Pre-caching reference embeddings...")
        self.reference_cache.clear()
        self._raw_ref_embeddings.clear()
        
        speaker_ref_ids: Dict[str, List[str]] = defaultdict(list)
        group_ref_ids: Dict[str, List[str]] = defaultdict(list)
        
        for sample in self.samples:
            if sample.sample_type == 'reference':
                speaker_ref_ids[sample.speaker_id].append(sample.sample_id)
                if sample.reference_id:
                    group_ref_ids[sample.reference_id].append(sample.sample_id)
                try:
                    sr, raw_audio = load_wav(sample.absolute_path)
                    trimmed, _ = detect_speech(raw_audio, sr)
                    emb = extract_embedding(trimmed, sr)
                    self._raw_ref_embeddings[sample.sample_id] = emb
                    print(f"  - Extracted ref sample: {sample.sample_id}")
                except Exception as e:
                    # Do not silently substitute or fabricate failed reference samples
                    self._raw_ref_embeddings[sample.sample_id] = None
                    print(f"  - Failed ref sample {sample.sample_id}: {e}")

        # 1. Cache explicit reference groups (where reference rows share a group ID in reference_id)
        for group_id, ref_ids in group_ref_ids.items():
            target_ids = ref_ids[:config.MIN_ENROLLMENT_SAMPLES] if len(ref_ids) >= config.MIN_ENROLLMENT_SAMPLES else ref_ids
            if all(self._raw_ref_embeddings.get(rid) is not None for rid in target_ids):
                embs = [self._raw_ref_embeddings[rid] for rid in target_ids]
                self.reference_cache[group_id] = self.aggregator.aggregate(embs)

        # 2. Cache speaker-level aggregated references (3-sample production enrollment when available)
        for speaker_id, ref_ids in speaker_ref_ids.items():
            if len(ref_ids) >= config.MIN_ENROLLMENT_SAMPLES:
                target_ids = ref_ids[:config.MIN_ENROLLMENT_SAMPLES]
                if all(self._raw_ref_embeddings.get(rid) is not None for rid in target_ids):
                    embs = [self._raw_ref_embeddings[rid] for rid in target_ids]
                    agg_emb = self.aggregator.aggregate(embs)
                    self.reference_cache[speaker_id] = agg_emb
                    # Map constituent reference IDs to the speaker's 3-sample aggregate when not explicitly grouped
                    for rid in target_ids:
                        self.reference_cache[rid] = agg_emb
            elif len(ref_ids) == 1:
                rid = ref_ids[0]
                if self._raw_ref_embeddings.get(rid) is not None:
                    agg_emb = self.aggregator.aggregate([self._raw_ref_embeddings[rid]])
                    self.reference_cache[rid] = agg_emb
                    self.reference_cache[speaker_id] = agg_emb
            else:
                # 2 reference samples: cache individual normalized embeddings only; require 3 for multi-sample speaker aggregate
                for rid in ref_ids:
                    if self._raw_ref_embeddings.get(rid) is not None:
                        self.reference_cache[rid] = self.aggregator.aggregate([self._raw_ref_embeddings[rid]])

    def _resolve_reference(self, reference_id: Optional[str]) -> Optional[SpeakerEmbedding]:
        if not reference_id:
            return None
        if reference_id in self.reference_cache:
            return self.reference_cache[reference_id]
        if ';' in reference_id:
            tokens = [t.strip() for t in reference_id.split(';') if t.strip()]
            if not tokens:
                return None
            # Do not silently substitute missing/failed samples
            if not all(self._raw_ref_embeddings.get(t) is not None for t in tokens):
                return None
            embs = [self._raw_ref_embeddings[t] for t in tokens]
            agg_emb = self.aggregator.aggregate(embs)
            self.reference_cache[reference_id] = agg_emb
            return agg_emb
        return None

    def run(self) -> List[EvaluationResultRow]:
        self._precache_references()
        results = []
        
        attempts = [s for s in self.samples if s.sample_type != 'reference']
        print(f"Running evaluation on {len(attempts)} attempts...")
        
        for sample in attempts:
            tracker = LatencyTracker()
            start_total = time.perf_counter()
            
            # Load raw with correct tuple unpacking: (sample_rate, audio_data)
            sr, raw_audio = load_wav(sample.absolute_path)
            quality = analyze_quality(raw_audio, sr)
            
            row = EvaluationResultRow(
                sample_id=sample.sample_id,
                speaker_id=sample.speaker_id,
                sample_type=sample.sample_type,
                reference_id=sample.reference_id,
                condition=sample.condition,
                audio_duration=quality.duration_sec,
                rms_level=quality.rms_level,
                clipping_ratio=quality.clipping_ratio,
                vad_detected=False,
                antispoof_score=None,
                similarity_score=None,
                speaker_model_id="speechbrain/spkrec-ecapa-voxceleb",
                antispoof_model_id="MattyB95/AST-ASVspoof2019-Synthetic-Voice-Detection",
                vad_ms=0, liveness_ms=0, embedding_ms=0, verification_ms=0, total_ms=0,
                baseline_pass=False,
                baseline_reason=""
            )
            
            ref_emb = self._resolve_reference(sample.reference_id)
            
            # Offline Evaluation Pipeline Execution:
            # Liveness and Speaker Verification are evaluated independently so both scores
            # are always recorded when VAD succeeds.
            try:
                # 1. VAD
                tracker.start('vad')
                trimmed_audio, _ = detect_speech(raw_audio, sr)
                tracker.stop('vad')
                row.vad_detected = True
                
                # 2. Anti-Spoof (Liveness)
                tracker.start('liveness')
                liveness_res = check_liveness(trimmed_audio, sr, config.DEFAULT_LIVENESS_THRESHOLD)
                tracker.stop('liveness')
                row.antispoof_score = liveness_res.score
                
                # 3. Independent Speaker Embedding & Verification (computed even if liveness rejects)
                verify_res = None
                if ref_emb is not None:
                    tracker.start('embedding')
                    current_emb = extract_embedding(trimmed_audio, sr)
                    tracker.stop('embedding')
                    
                    tracker.start('verification')
                    verify_res = verify_speaker(ref_emb, current_emb, config.DEFAULT_SIMILARITY_THRESHOLD)
                    tracker.stop('verification')
                    
                    row.similarity_score = verify_res.score
                
                # 4. Baseline Decision (preserves production order: liveness -> reference -> verification)
                if not liveness_res.is_live:
                    row.baseline_pass = False
                    row.baseline_reason = FailureReason.ANTI_SPOOF_FAILED.value
                elif ref_emb is None:
                    row.baseline_pass = False
                    row.baseline_reason = FailureReason.INVALID_REFERENCE.value
                elif verify_res is not None and verify_res.match:
                    row.baseline_pass = True
                    row.baseline_reason = FailureReason.SUCCESS.value
                else:
                    row.baseline_pass = False
                    row.baseline_reason = FailureReason.SPEAKER_MISMATCH.value
                        
            except NoSpeechDetectedError:
                row.baseline_reason = FailureReason.NO_SPEECH_DETECTED.value
            except Exception:
                row.baseline_reason = FailureReason.MODEL_ERROR.value
                
            tracker.record_total(start_total)
            row.vad_ms = tracker.metrics.vad_ms
            row.liveness_ms = tracker.metrics.liveness_ms
            row.embedding_ms = tracker.metrics.embedding_ms
            row.verification_ms = tracker.metrics.verification_ms
            row.total_ms = tracker.metrics.total_ms
            
            results.append(row)
            
        return results
