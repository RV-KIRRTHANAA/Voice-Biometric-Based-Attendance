import pytest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

from src.audio.config import SAMPLE_RATE
from src.audio.embedding import SpeakerEmbedding
from src.audio.liveness import LivenessResult
from src.audio.verification import VerificationResult
from src.audio.wav_io import save_wav, load_wav
from src.engine.aggregation import MeanEmbeddingAggregator
from src.engine.results import FailureReason
from src.evaluation.dataset import DatasetValidator, EvalSample
from src.evaluation.metrics import (
    DEFAULT_SIMILARITY_SWEEP_THRESHOLDS,
    DEFAULT_LIVENESS_SWEEP_THRESHOLDS,
    sweep_thresholds,
    calculate_metrics_for_threshold,
    calculate_eer,
    calculate_roc_curve,
    calculate_liveness_metrics_for_threshold,
    sweep_liveness_thresholds,
    calculate_liveness_score_distributions,
    evaluate_replay_category,
)
from src.evaluation.runner import EvaluationRunner, EvaluationResultRow
from src.evaluation.reports import ReportGenerator


# ─── Existing Baseline Evaluation Tests (Preserved) ──────────────────────────

def test_metrics_calculation():
    # Construct synthetic raw results
    results = [
        EvaluationResultRow("1", "s1", "genuine", "ref1", "quiet", 2.0, 0.5, 0.0, True, 0.9, 0.9, "m", "m", 0,0,0,0,0, True, ""), # True Accept at 0.75
        EvaluationResultRow("2", "s1", "genuine", "ref1", "quiet", 2.0, 0.5, 0.0, True, 0.9, 0.6, "m", "m", 0,0,0,0,0, False, ""), # False Reject at 0.75
        EvaluationResultRow("3", "s2", "impostor", "ref1", "quiet", 2.0, 0.5, 0.0, True, 0.9, 0.8, "m", "m", 0,0,0,0,0, True, ""), # False Accept at 0.75
        EvaluationResultRow("4", "s2", "impostor", "ref1", "quiet", 2.0, 0.5, 0.0, True, 0.9, 0.2, "m", "m", 0,0,0,0,0, False, ""), # True Reject at 0.75
        EvaluationResultRow("5", "s1", "replay", "ref1", "phone", 2.0, 0.5, 0.0, True, 0.1, None, "m", "m", 0,0,0,0,0, False, "ANTI_SPOOF_FAILED") # Short circuit
    ]
    
    m = calculate_metrics_for_threshold(results, 0.75)
    
    assert m.true_accepts == 1
    assert m.false_rejects == 1
    assert m.false_accepts == 1
    assert m.true_rejects == 1
    
    # Genuine count = 2 -> FRR = 1/2 = 0.5
    assert m.frr == 0.5
    # Impostor count = 2 -> FAR = 1/2 = 0.5
    assert m.far == 0.5
    # Total evaluated for speaker matching = 4. Accuracy = (1+1)/4 = 0.5
    assert m.accuracy == 0.5
    assert m.tpr == 0.5
    assert m.fpr == 0.5


def test_dataset_validator_manifest(tmp_path):
    # Setup dummy manifest
    manifest_file = tmp_path / "manifest.csv"
    manifest_file.write_text("sample_id,speaker_id,sample_type,reference_id,condition,path\n"
                             "r1,s1,reference,,quiet,r1.wav\n"
                             "a1,s1,genuine,r1,noise,a1.wav\n")
    
    # Create dummy audio files so file checks pass
    (tmp_path / "r1.wav").write_text("audio")
    (tmp_path / "a1.wav").write_text("audio")
    
    validator = DatasetValidator(str(tmp_path))
    samples = validator.load_and_validate()
    
    assert len(samples) == 2
    assert samples[0].sample_id == 'r1'
    assert samples[1].reference_id == 'r1'


def test_dataset_validator_missing_ref(tmp_path):
    manifest_file = tmp_path / "manifest.csv"
    # missing reference 'r99'
    manifest_file.write_text("sample_id,speaker_id,sample_type,reference_id,condition,path\n"
                             "a1,s1,genuine,r99,noise,a1.wav\n")
    (tmp_path / "a1.wav").write_text("audio")
    
    validator = DatasetValidator(str(tmp_path))
    with pytest.raises(ValueError, match="unknown reference_id"):
        validator.load_and_validate()


def test_report_generation(tmp_path):
    reporter = ReportGenerator(str(tmp_path))
    results = [
        EvaluationResultRow("1", "s1", "genuine", "ref1", "quiet", 2.0, 0.5, 0.0, True, 0.9, 0.9, "m", "m", 0,0,0,0,0, True, "SUCCESS")
    ]
    reporter.write_evaluation_results_csv(results)
    
    assert (tmp_path / "results" / "evaluation_results.csv").exists()
    content = (tmp_path / "results" / "evaluation_results.csv").read_text()
    assert "sample_id" in content
    assert "SUCCESS" in content


# ─── Phase 1 & 2 Tests: WAV Unpacking, Valid Runner Flow, Independent Scores ─

def test_runner_wav_tuple_unpacking_and_valid_sample(tmp_path, monkeypatch):
    """
    Verifies that load_wav's (sample_rate, audio_data) return order is unpacked
    correctly in both _precache_references() and run(), passing integer sr=16000
    and ndarray audio_data to detect_speech, check_liveness, and extract_embedding.
    """
    ref_wav = tmp_path / "ref1.wav"
    gen_wav = tmp_path / "gen1.wav"
    audio_samples = np.full(SAMPLE_RATE * 2, 1000, dtype=np.int16)  # 2.0 seconds at 16kHz
    save_wav(str(ref_wav), audio_samples, SAMPLE_RATE)
    save_wav(str(gen_wav), audio_samples, SAMPLE_RATE)

    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "sample_id,speaker_id,sample_type,reference_id,condition,path\n"
        "ref1,spk1,reference,,quiet,ref1.wav\n"
        "gen1,spk1,genuine,ref1,quiet,gen1.wav\n"
    )

    samples = DatasetValidator(str(tmp_path)).load_and_validate()

    observed_calls = []

    def fake_detect_speech(audio_arr, sr):
        # Verify audio_arr is a numpy ndarray and sr is an int (not swapped!)
        assert isinstance(audio_arr, np.ndarray)
        assert isinstance(sr, int)
        assert sr == SAMPLE_RATE
        assert len(audio_arr) == SAMPLE_RATE * 2
        observed_calls.append(("vad", sr, len(audio_arr)))
        return audio_arr, [{"start": 0, "end": len(audio_arr)}]

    def fake_check_liveness(audio_arr, sr, threshold):
        assert isinstance(audio_arr, np.ndarray)
        assert sr == SAMPLE_RATE
        observed_calls.append(("liveness", sr, len(audio_arr)))
        return LivenessResult(is_live=True, score=0.92, threshold=threshold, model_id="test-ast")

    def fake_extract_embedding(audio_arr, sr):
        assert isinstance(audio_arr, np.ndarray)
        assert sr == SAMPLE_RATE
        observed_calls.append(("embedding", sr, len(audio_arr)))
        vec = np.ones(192, dtype=np.float32)
        return SpeakerEmbedding(vector=vec, model_id="speechbrain/spkrec-ecapa-voxceleb", dimension=192)

    monkeypatch.setattr("src.evaluation.runner.detect_speech", fake_detect_speech)
    monkeypatch.setattr("src.evaluation.runner.check_liveness", fake_check_liveness)
    monkeypatch.setattr("src.evaluation.runner.extract_embedding", fake_extract_embedding)

    runner = EvaluationRunner(samples)
    rows = runner.run()

    assert len(rows) == 1
    row = rows[0]
    assert row.sample_id == "gen1"
    assert pytest.approx(row.audio_duration, 0.01) == 2.0
    assert row.vad_detected is True
    assert pytest.approx(row.antispoof_score, 1e-5) == 0.92
    assert pytest.approx(row.similarity_score, 1e-5) == 1.0
    assert row.baseline_pass is True
    assert row.baseline_reason == FailureReason.SUCCESS.value
    assert len(observed_calls) == 5  # 2 for ref (vad, emb) + 3 for attempt (vad, liveness, emb)


def test_similarity_calculated_even_when_liveness_rejects(tmp_path, monkeypatch):
    """
    Verifies Phase 2 & 4 requirement:
    When liveness fails (is_live=False), EvaluationRunner still extracts the speaker
    embedding and records similarity_score, and speaker-verification FRR includes
    that genuine recording rather than silently dropping it.
    """
    ref_wav = tmp_path / "ref1.wav"
    gen_wav = tmp_path / "gen_rejected_by_liveness.wav"
    audio_samples = np.ones(SAMPLE_RATE, dtype=np.int16)
    save_wav(str(ref_wav), audio_samples, SAMPLE_RATE)
    save_wav(str(gen_wav), audio_samples, SAMPLE_RATE)

    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "sample_id,speaker_id,sample_type,reference_id,condition,path\n"
        "ref1,spk1,reference,,quiet,ref1.wav\n"
        "gen1,spk1,genuine,ref1,quiet,gen_rejected_by_liveness.wav\n"
    )

    samples = DatasetValidator(str(tmp_path)).load_and_validate()

    monkeypatch.setattr("src.evaluation.runner.detect_speech", lambda a, sr: (a, []))
    # Liveness REJECTS the sample
    monkeypatch.setattr(
        "src.evaluation.runner.check_liveness",
        lambda a, sr, th: LivenessResult(is_live=False, score=0.0, threshold=0.5, model_id="test-ast")
    )
    # Orthogonal or partial-match embedding so similarity = 0.5
    call_idx = {"n": 0}
    def fake_extract(a, sr):
        call_idx["n"] += 1
        vec = np.zeros(192, dtype=np.float64)
        if call_idx["n"] == 1:
            vec[0] = 1.0
        else:
            vec[0] = 0.5
            vec[1] = np.sqrt(3) / 2.0
        return SpeakerEmbedding(vector=vec, model_id="speechbrain/spkrec-ecapa-voxceleb", dimension=192)

    monkeypatch.setattr("src.evaluation.runner.extract_embedding", fake_extract)

    runner = EvaluationRunner(samples)
    rows = runner.run()

    assert len(rows) == 1
    row = rows[0]
    # Baseline decision reflects anti-spoof rejection
    assert row.baseline_pass is False
    assert row.baseline_reason == FailureReason.ANTI_SPOOF_FAILED.value
    # Yet similarity_score IS computed and recorded!
    assert row.similarity_score is not None
    assert pytest.approx(row.similarity_score, 1e-4) == 0.5

    # And speaker-verification FRR includes this genuine sample at threshold 0.60
    metrics_at_060 = calculate_metrics_for_threshold(rows, 0.60)
    assert metrics_at_060.false_rejects == 1
    assert metrics_at_060.frr == 1.0


# ─── Phase 3 Tests: Three-Reference Embedding Aggregation ────────────────────

def test_three_reference_embedding_aggregation_matches_production(tmp_path, monkeypatch):
    """
    Verifies that when 3 reference samples exist for a speaker (or are specified via
    r1;r2;r3), EvaluationRunner aggregates all 3 embeddings using MeanEmbeddingAggregator
    with the exact same L2 normalization as production EnrollmentService.
    """
    for name in ("r1.wav", "r2.wav", "r3.wav", "gen1.wav"):
        save_wav(str(tmp_path / name), np.ones(SAMPLE_RATE, dtype=np.int16), SAMPLE_RATE)

    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "sample_id,speaker_id,sample_type,reference_id,condition,path\n"
        "r1,spk1,reference,,quiet,r1.wav\n"
        "r2,spk1,reference,,quiet,r2.wav\n"
        "r3,spk1,reference,,quiet,r3.wav\n"
        "gen1,spk1,genuine,r1;r2;r3,quiet,gen1.wav\n"
    )

    samples = DatasetValidator(str(tmp_path)).load_and_validate()
    monkeypatch.setattr("src.evaluation.runner.detect_speech", lambda a, sr: (a, []))
    monkeypatch.setattr(
        "src.evaluation.runner.check_liveness",
        lambda a, sr, th: LivenessResult(is_live=True, score=0.99, threshold=th, model_id="m")
    )

    # Distinct unnormalized vectors for r1, r2, r3, and query gen1
    v1 = np.zeros(192); v1[0] = 2.0
    v2 = np.zeros(192); v2[1] = 4.0
    v3 = np.zeros(192); v3[2] = 6.0
    v_query = np.zeros(192); v_query[0] = 1.0; v_query[1] = 1.0; v_query[2] = 1.0

    vectors_iter = iter([v1, v2, v3, v_query])
    monkeypatch.setattr(
        "src.evaluation.runner.extract_embedding",
        lambda a, sr: SpeakerEmbedding(vector=next(vectors_iter), model_id="m", dimension=192)
    )

    runner = EvaluationRunner(samples)
    rows = runner.run()

    # Expected production reference from MeanEmbeddingAggregator:
    # Each of v1, v2, v3 normalizes to e_0, e_1, e_2; their mean normalized is (e_0 + e_1 + e_2) / sqrt(3)
    expected_ref = MeanEmbeddingAggregator().aggregate([
        SpeakerEmbedding(v1, "m", 192),
        SpeakerEmbedding(v2, "m", 192),
        SpeakerEmbedding(v3, "m", 192),
    ])
    cached_ref = runner.reference_cache["r1;r2;r3"]
    np.testing.assert_allclose(cached_ref.vector, expected_ref.vector, rtol=1e-6, atol=1e-6)
    assert pytest.approx(np.linalg.norm(cached_ref.vector), 1e-6) == 1.0

    # And v_query = [1, 1, 1, 0, ...] has cosine similarity 1.0 with expected_ref
    assert len(rows) == 1
    assert pytest.approx(rows[0].similarity_score, 1e-6) == 1.0


def test_failed_reference_sample_is_not_silently_substituted(tmp_path, monkeypatch):
    """
    Verifies that if one of the 3 reference samples fails VAD/extraction,
    EvaluationRunner does NOT silently substitute a 2-sample reference.
    """
    for name in ("r1.wav", "r2.wav", "r3.wav", "gen1.wav"):
        save_wav(str(tmp_path / name), np.ones(SAMPLE_RATE, dtype=np.int16), SAMPLE_RATE)

    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "sample_id,speaker_id,sample_type,reference_id,condition,path\n"
        "r1,spk1,reference,,quiet,r1.wav\n"
        "r2,spk1,reference,,quiet,r2.wav\n"
        "r3,spk1,reference,,quiet,r3.wav\n"
        "gen1,spk1,genuine,r1;r2;r3,quiet,gen1.wav\n"
    )

    samples = DatasetValidator(str(tmp_path)).load_and_validate()
    monkeypatch.setattr("src.evaluation.runner.detect_speech", lambda a, sr: (a, []))
    monkeypatch.setattr(
        "src.evaluation.runner.check_liveness",
        lambda a, sr, th: LivenessResult(is_live=True, score=0.99, threshold=th, model_id="m")
    )

    # Make the 2nd reference sample fail embedding extraction
    call_count = {"n": 0}
    def flaky_extract(a, sr):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("Corrupt reference audio")
        return SpeakerEmbedding(vector=np.ones(192), model_id="m", dimension=192)

    monkeypatch.setattr("src.evaluation.runner.extract_embedding", flaky_extract)

    runner = EvaluationRunner(samples)
    rows = runner.run()

    assert len(rows) == 1
    assert rows[0].similarity_score is None
    assert rows[0].baseline_pass is False
    assert rows[0].baseline_reason == FailureReason.INVALID_REFERENCE.value


# ─── Phase 4 Tests: Similarity Sweep, EER, ROC, Liveness, Replay Separation ──

def test_similarity_sweep_range_and_step():
    """
    Verifies default similarity sweep spans 0.10 to 0.85 with step 0.01 (76 thresholds).
    """
    assert len(DEFAULT_SIMILARITY_SWEEP_THRESHOLDS) == 76
    assert DEFAULT_SIMILARITY_SWEEP_THRESHOLDS[0] == 0.10
    assert DEFAULT_SIMILARITY_SWEEP_THRESHOLDS[-1] == 0.85
    diffs = [
        round(DEFAULT_SIMILARITY_SWEEP_THRESHOLDS[i + 1] - DEFAULT_SIMILARITY_SWEEP_THRESHOLDS[i], 4)
        for i in range(len(DEFAULT_SIMILARITY_SWEEP_THRESHOLDS) - 1)
    ]
    assert all(d == 0.01 for d in diffs)


def test_eer_and_roc_calculations():
    """
    Verifies EER and ROC (TPR/FPR) calculations on a deterministic dataset.
    Genuine scores: [0.40, 0.60, 0.80, 0.90]
    Impostor scores: [0.10, 0.20, 0.30, 0.50]
    At threshold = 0.45:
      - Genuine < 0.45: 1/4 -> FRR = 0.25, TPR = 0.75
      - Impostor >= 0.45: 1/4 -> FAR = 0.25, FPR = 0.25
    So FAR == FRR == 0.25 -> EER = 0.25!
    """
    rows = [
        EvaluationResultRow("g1", "s1", "genuine", "r1", "q", 1.0, 0.5, 0.0, True, 0.9, 0.40, "m", "m", 0,0,0,0,0, True, ""),
        EvaluationResultRow("g2", "s1", "genuine", "r1", "q", 1.0, 0.5, 0.0, True, 0.9, 0.60, "m", "m", 0,0,0,0,0, True, ""),
        EvaluationResultRow("g3", "s1", "genuine", "r1", "q", 1.0, 0.5, 0.0, True, 0.9, 0.80, "m", "m", 0,0,0,0,0, True, ""),
        EvaluationResultRow("g4", "s1", "genuine", "r1", "q", 1.0, 0.5, 0.0, True, 0.9, 0.90, "m", "m", 0,0,0,0,0, True, ""),
        EvaluationResultRow("i1", "s2", "impostor", "r1", "q", 1.0, 0.5, 0.0, True, 0.9, 0.10, "m", "m", 0,0,0,0,0, False, ""),
        EvaluationResultRow("i2", "s2", "impostor", "r1", "q", 1.0, 0.5, 0.0, True, 0.9, 0.20, "m", "m", 0,0,0,0,0, False, ""),
        EvaluationResultRow("i3", "s2", "impostor", "r1", "q", 1.0, 0.5, 0.0, True, 0.9, 0.30, "m", "m", 0,0,0,0,0, False, ""),
        EvaluationResultRow("i4", "s2", "impostor", "r1", "q", 1.0, 0.5, 0.0, True, 0.9, 0.50, "m", "m", 0,0,0,0,0, False, ""),
    ]

    eer_res = calculate_eer(rows)
    assert eer_res.eer is not None
    assert pytest.approx(eer_res.eer, 1e-4) == 0.25
    assert 0.40 < eer_res.threshold <= 0.50

    roc_points = calculate_roc_curve(rows)
    assert len(roc_points) == 76
    point_045 = next(p for p in roc_points if p.threshold == 0.45)
    assert pytest.approx(point_045.tpr, 1e-4) == 0.75
    assert pytest.approx(point_045.fpr, 1e-4) == 0.25
    assert pytest.approx(point_045.far, 1e-4) == 0.25
    assert pytest.approx(point_045.frr, 1e-4) == 0.25


def test_liveness_sweep_and_replay_not_mixed_with_synthetic():
    """
    Verifies:
    1. Bonafide FRR and Synthetic Spoof FAR are calculated accurately.
    2. Replay samples ('replay') are NOT counted in synthetic spoof metrics or distributions.
    3. evaluate_replay_category() reports replay separately and reports non-evaluability when 0 replay samples exist.
    """
    rows = [
        EvaluationResultRow("g1", "s1", "genuine", "r1", "q", 1.0, 0.5, 0.0, True, 0.80, 0.9, "m", "m", 0,0,0,0,0, True, ""),
        EvaluationResultRow("g2", "s1", "genuine", "r1", "q", 1.0, 0.5, 0.0, True, 0.40, 0.8, "m", "m", 0,0,0,0,0, False, ""),
        EvaluationResultRow("syn1", "s1", "synthetic", "r1", "tts", 1.0, 0.5, 0.0, True, 0.60, 0.7, "m", "m", 0,0,0,0,0, True, ""),
        EvaluationResultRow("syn2", "s1", "synthetic", "r1", "vc", 1.0, 0.5, 0.0, True, 0.10, 0.7, "m", "m", 0,0,0,0,0, False, ""),
        # Replay sample with high antispoof_score (0.95) - MUST NOT be mixed into synthetic_count or synthetic_far!
        EvaluationResultRow("rep1", "s1", "replay", "r1", "phone", 1.0, 0.5, 0.0, True, 0.95, 0.9, "m", "m", 0,0,0,0,0, True, ""),
    ]

    liv = calculate_liveness_metrics_for_threshold(rows, threshold=0.50)
    assert liv.bonafide_count == 2
    assert liv.synthetic_count == 2  # 'rep1' was NOT mixed into synthetic_count
    assert liv.bonafide_true_accepts == 1
    assert liv.bonafide_false_rejects == 1
    assert liv.bonafide_frr == 0.5
    assert liv.synthetic_false_accepts == 1
    assert liv.synthetic_true_rejects == 1
    assert liv.synthetic_far == 0.5

    swept_liv = sweep_liveness_thresholds(rows, thresholds=[0.20, 0.50, 0.70])
    assert len(swept_liv) == 3
    assert swept_liv[0].bonafide_frr == 0.0
    assert swept_liv[2].synthetic_far == 0.0

    dists = calculate_liveness_score_distributions(rows)
    assert set(dists.keys()) == {"genuine", "synthetic"}
    assert dists["genuine"].count == 2
    assert dists["synthetic"].count == 2
    assert 0.95 not in dists["synthetic"].scores  # replay score excluded

    # When no replay samples exist, evaluate_replay_category reports non-evaluability
    rows_without_replay = [r for r in rows if r.sample_type != "replay"]
    rep_empty = evaluate_replay_category(rows_without_replay)
    assert rep_empty.evaluable is False
    assert rep_empty.replay_count == 0
    assert "cannot be evaluated" in rep_empty.message


def test_empty_dataset_and_missing_file_handling(tmp_path):
    """
    Verifies graceful handling of empty datasets in metrics and FileNotFoundError on missing audio files.
    """
    empty_metrics = calculate_metrics_for_threshold([], 0.5)
    assert empty_metrics.far == 0.0
    assert empty_metrics.frr == 0.0
    assert empty_metrics.accuracy == 0.0

    empty_eer = calculate_eer([])
    assert empty_eer.eer is None
    assert empty_eer.threshold is None

    empty_liv = calculate_liveness_metrics_for_threshold([], 0.5)
    assert empty_liv.bonafide_frr == 0.0
    assert empty_liv.synthetic_far == 0.0

    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "sample_id,speaker_id,sample_type,reference_id,condition,path\n"
        "r1,spk1,reference,,quiet,does_not_exist.wav\n"
    )
    validator = DatasetValidator(str(tmp_path))
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        validator.load_and_validate()
