from dataclasses import dataclass, field
import statistics
from typing import List, Dict, Optional
from src.evaluation.runner import EvaluationResultRow

# Default similarity sweep: 0.10 to 0.85 inclusive, step 0.01 (76 thresholds)
DEFAULT_SIMILARITY_SWEEP_THRESHOLDS: List[float] = [
    round(0.10 + 0.01 * i, 2) for i in range(76)
]

# Default liveness sweep covering low log-scale probabilities and [0.01..0.99]
DEFAULT_LIVENESS_SWEEP_THRESHOLDS: List[float] = (
    [1e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
    + [round(0.01 + 0.01 * i, 2) for i in range(99)]
)


@dataclass
class ThresholdMetrics:
    threshold: float
    true_accepts: int
    false_accepts: int
    true_rejects: int
    false_rejects: int
    far: float
    frr: float
    accuracy: float
    tpr: float = 0.0
    fpr: float = 0.0


@dataclass
class EERResult:
    eer: Optional[float]
    threshold: Optional[float]
    far_at_eer: Optional[float]
    frr_at_eer: Optional[float]


@dataclass
class ROCPoint:
    threshold: float
    tpr: float
    fpr: float
    far: float
    frr: float


@dataclass
class LivenessThresholdMetrics:
    threshold: float
    bonafide_count: int
    synthetic_count: int
    bonafide_true_accepts: int
    bonafide_false_rejects: int
    synthetic_false_accepts: int
    synthetic_true_rejects: int
    bonafide_frr: float
    synthetic_far: float


@dataclass
class ScoreDistributionSummary:
    category: str
    count: int
    min_score: Optional[float]
    max_score: Optional[float]
    mean_score: Optional[float]
    median_score: Optional[float]
    scores: List[float] = field(default_factory=list)


@dataclass
class ReplayCategoryReport:
    replay_count: int
    evaluable: bool
    message: str
    false_accepts: Optional[int] = None
    true_rejects: Optional[int] = None
    replay_far: Optional[float] = None


def calculate_metrics_for_threshold(results: List[EvaluationResultRow], threshold: float) -> ThresholdMetrics:
    """
    Calculates speaker-verification FAR, FRR, Accuracy, TPR, and FPR for a specific threshold.
    Evaluates all 'genuine' and 'impostor' rows with a similarity_score, independent of liveness decisions.
    """
    ta, fa, tr, fr = 0, 0, 0, 0
    genuine_count = 0
    impostor_count = 0
    
    for r in results:
        if r.similarity_score is None:
            continue
            
        is_genuine = (r.sample_type == 'genuine')
        is_impostor = (r.sample_type == 'impostor')
        
        if not is_genuine and not is_impostor:
            continue
            
        if is_genuine:
            genuine_count += 1
        if is_impostor:
            impostor_count += 1
            
        accepted = r.similarity_score >= threshold
        
        if is_genuine and accepted:
            ta += 1
        elif is_genuine and not accepted:
            fr += 1
        elif is_impostor and accepted:
            fa += 1
        elif is_impostor and not accepted:
            tr += 1
            
    far = (fa / impostor_count) if impostor_count > 0 else 0.0
    frr = (fr / genuine_count) if genuine_count > 0 else 0.0
    total = genuine_count + impostor_count
    acc = ((ta + tr) / total) if total > 0 else 0.0
    tpr = (ta / genuine_count) if genuine_count > 0 else 0.0
    fpr = far
    
    return ThresholdMetrics(
        threshold=float(threshold),
        true_accepts=ta,
        false_accepts=fa,
        true_rejects=tr,
        false_rejects=fr,
        far=far,
        frr=frr,
        accuracy=acc,
        tpr=tpr,
        fpr=fpr,
    )


def sweep_thresholds(
    results: List[EvaluationResultRow],
    thresholds: Optional[List[float]] = None
) -> List[ThresholdMetrics]:
    """
    Sweeps speaker similarity thresholds.
    Defaults to 0.10 through 0.85 with step 0.01 when thresholds is None.
    """
    if thresholds is None:
        thresholds = DEFAULT_SIMILARITY_SWEEP_THRESHOLDS
    return [calculate_metrics_for_threshold(results, t) for t in thresholds]


def calculate_roc_curve(
    results: List[EvaluationResultRow],
    thresholds: Optional[List[float]] = None
) -> List[ROCPoint]:
    """
    Generates ROC curve points (TPR, FPR, FAR, FRR) across the similarity threshold sweep.
    """
    metrics_list = sweep_thresholds(results, thresholds)
    return [
        ROCPoint(
            threshold=m.threshold,
            tpr=m.tpr,
            fpr=m.fpr,
            far=m.far,
            frr=m.frr,
        )
        for m in metrics_list
    ]


def calculate_eer(
    results: List[EvaluationResultRow],
    thresholds: Optional[List[float]] = None
) -> EERResult:
    """
    Computes the Equal Error Rate (EER) where FAR == FRR for speaker verification.
    Returns EERResult with None fields if genuine or impostor samples are missing.
    """
    genuine_scores = [r.similarity_score for r in results if r.sample_type == 'genuine' and r.similarity_score is not None]
    impostor_scores = [r.similarity_score for r in results if r.sample_type == 'impostor' and r.similarity_score is not None]
    
    if not genuine_scores or not impostor_scores:
        return EERResult(eer=None, threshold=None, far_at_eer=None, frr_at_eer=None)
        
    if thresholds is None:
        # Combine standard sweep grid with observed scores and midpoints for exact crossing detection
        observed = sorted(set(genuine_scores + impostor_scores))
        midpoints = [(observed[i] + observed[i + 1]) / 2.0 for i in range(len(observed) - 1)]
        candidate_thresholds = sorted(
            set(DEFAULT_SIMILARITY_SWEEP_THRESHOLDS)
            | set(observed)
            | set(midpoints)
            | {round(-1.0 + 0.01 * i, 2) for i in range(201)}
        )
    else:
        candidate_thresholds = sorted(thresholds)
        
    swept = [calculate_metrics_for_threshold(results, t) for t in candidate_thresholds]
    
    # Find the threshold with the smallest |FAR - FRR|
    best = min(swept, key=lambda m: abs(m.far - m.frr))
    
    # Also check if there is a crossing interval where (far - frr) changes sign
    for i in range(len(swept) - 1):
        m1, m2 = swept[i], swept[i + 1]
        diff1 = m1.far - m1.frr
        diff2 = m2.far - m2.frr
        if diff1 == 0.0:
            return EERResult(eer=m1.far, threshold=m1.threshold, far_at_eer=m1.far, frr_at_eer=m1.frr)
        if diff1 > 0.0 and diff2 < 0.0:
            # Linear interpolation of the crossing point between m1 and m2
            denom = diff1 - diff2
            alpha = diff1 / denom if denom != 0 else 0.5
            interp_eer = m1.far + alpha * (m2.far - m1.far)
            interp_thresh = m1.threshold + alpha * (m2.threshold - m1.threshold)
            # If best discrete point already has diff == 0, prefer exact; otherwise return interpolated EER
            if abs(best.far - best.frr) == 0.0:
                return EERResult(eer=best.far, threshold=best.threshold, far_at_eer=best.far, frr_at_eer=best.frr)
            return EERResult(
                eer=float(interp_eer),
                threshold=float(interp_thresh),
                far_at_eer=float(m2.far if abs(diff2) < abs(diff1) else m1.far),
                frr_at_eer=float(m2.frr if abs(diff2) < abs(diff1) else m1.frr),
            )
            
    eer_val = (best.far + best.frr) / 2.0
    return EERResult(
        eer=float(eer_val),
        threshold=float(best.threshold),
        far_at_eer=float(best.far),
        frr_at_eer=float(best.frr),
    )


def calculate_liveness_metrics_for_threshold(
    results: List[EvaluationResultRow],
    threshold: float
) -> LivenessThresholdMetrics:
    """
    Calculates synthetic-speech liveness metrics for a given threshold:
    - Bonafide False Rejection Rate (bonafide_frr) over 'genuine' samples
    - Synthetic Spoof False Acceptance Rate (synthetic_far) over 'synthetic' samples
    Physical 'replay' samples are strictly excluded from synthetic speech metrics.
    """
    bonafide_ta, bonafide_fr = 0, 0
    synthetic_fa, synthetic_tr = 0, 0
    bonafide_count = 0
    synthetic_count = 0
    
    for r in results:
        if r.antispoof_score is None:
            continue
            
        if r.sample_type == 'genuine':
            bonafide_count += 1
            if r.antispoof_score >= threshold:
                bonafide_ta += 1
            else:
                bonafide_fr += 1
        elif r.sample_type == 'synthetic':
            synthetic_count += 1
            if r.antispoof_score >= threshold:
                synthetic_fa += 1
            else:
                synthetic_tr += 1
                
    bonafide_frr = (bonafide_fr / bonafide_count) if bonafide_count > 0 else 0.0
    synthetic_far = (synthetic_fa / synthetic_count) if synthetic_count > 0 else 0.0
    
    return LivenessThresholdMetrics(
        threshold=float(threshold),
        bonafide_count=bonafide_count,
        synthetic_count=synthetic_count,
        bonafide_true_accepts=bonafide_ta,
        bonafide_false_rejects=bonafide_fr,
        synthetic_false_accepts=synthetic_fa,
        synthetic_true_rejects=synthetic_tr,
        bonafide_frr=bonafide_frr,
        synthetic_far=synthetic_far,
    )


def sweep_liveness_thresholds(
    results: List[EvaluationResultRow],
    thresholds: Optional[List[float]] = None
) -> List[LivenessThresholdMetrics]:
    """
    Sweeps liveness thresholds independently of speaker verification.
    """
    if thresholds is None:
        thresholds = DEFAULT_LIVENESS_SWEEP_THRESHOLDS
    return [calculate_liveness_metrics_for_threshold(results, t) for t in thresholds]


def calculate_liveness_score_distributions(
    results: List[EvaluationResultRow]
) -> Dict[str, ScoreDistributionSummary]:
    """
    Computes liveness score distributions separately for 'genuine' and 'synthetic' recordings.
    Physical 'replay' recordings are never mixed into 'synthetic'.
    """
    distributions: Dict[str, ScoreDistributionSummary] = {}
    for category in ('genuine', 'synthetic'):
        scores = [
            float(r.antispoof_score)
            for r in results
            if r.sample_type == category and r.antispoof_score is not None
        ]
        if scores:
            distributions[category] = ScoreDistributionSummary(
                category=category,
                count=len(scores),
                min_score=min(scores),
                max_score=max(scores),
                mean_score=statistics.mean(scores),
                median_score=statistics.median(scores),
                scores=scores,
            )
        else:
            distributions[category] = ScoreDistributionSummary(
                category=category,
                count=0,
                min_score=None,
                max_score=None,
                mean_score=None,
                median_score=None,
                scores=[],
            )
    return distributions


def evaluate_replay_category(
    results: List[EvaluationResultRow],
    threshold: Optional[float] = None
) -> ReplayCategoryReport:
    """
    Evaluates physical replay samples as a strictly separate category from synthetic speech.
    If no replay samples exist, reports that replay performance cannot be evaluated.
    """
    replay_rows = [r for r in results if r.sample_type == 'replay' and r.antispoof_score is not None]
    if not replay_rows:
        return ReplayCategoryReport(
            replay_count=0,
            evaluable=False,
            message="No physical replay dataset available; physical replay performance cannot be evaluated.",
        )
        
    if threshold is None:
        return ReplayCategoryReport(
            replay_count=len(replay_rows),
            evaluable=True,
            message=f"Found {len(replay_rows)} physical replay sample(s); evaluated separately from synthetic speech.",
        )
        
    fa = sum(1 for r in replay_rows if r.antispoof_score >= threshold)
    tr = len(replay_rows) - fa
    far = fa / len(replay_rows)
    return ReplayCategoryReport(
        replay_count=len(replay_rows),
        evaluable=True,
        message=f"Evaluated {len(replay_rows)} physical replay sample(s) separately from synthetic speech.",
        false_accepts=fa,
        true_rejects=tr,
        replay_far=far,
    )
