"""
Main entry point for Phase 7 Biometric Evaluation.
"""
import os
import sys

from src.evaluation.dataset import DatasetValidator
from src.evaluation.runner import EvaluationRunner
from src.evaluation.metrics import (
    DEFAULT_SIMILARITY_SWEEP_THRESHOLDS,
    DEFAULT_LIVENESS_SWEEP_THRESHOLDS,
    sweep_thresholds,
    calculate_eer,
    calculate_roc_curve,
    sweep_liveness_thresholds,
    calculate_liveness_score_distributions,
    evaluate_replay_category,
)
from src.evaluation.reports import ReportGenerator

def main():
    print("=== OneApp Voice Biometric POC - Phase 7 Evaluation ===")
    
    data_dir = "data/evaluation"
    output_dir = "evaluation"
    
    print(f"Loading manifest from {data_dir}...")
    try:
        validator = DatasetValidator(data_dir)
        samples = validator.load_and_validate()
        print(f"Successfully validated {len(samples)} samples from manifest.")
    except Exception as e:
        print(f"Dataset Validation Error: {e}")
        sys.exit(1)
        
    if not samples:
        print("Manifest is empty. Nothing to evaluate.")
        sys.exit(0)
        
    print("\n--- Starting Evaluation Runner ---")
    runner = EvaluationRunner(samples)
    results = runner.run()
    
    print("\n--- Calculating Speaker Verification Metrics (0.10 to 0.85, step 0.01) ---")
    metrics = sweep_thresholds(results, DEFAULT_SIMILARITY_SWEEP_THRESHOLDS)
    eer_res = calculate_eer(results)
    roc_points = calculate_roc_curve(results, DEFAULT_SIMILARITY_SWEEP_THRESHOLDS)
    if eer_res.eer is not None:
        print(f"Speaker Verification EER: {eer_res.eer:.4f} at threshold {eer_res.threshold:.4f}")
    else:
        print("Speaker Verification EER: Insufficient genuine/impostor samples to compute EER.")

    print("\n--- Calculating Synthetic Speech Liveness Metrics ---")
    liveness_metrics = sweep_liveness_thresholds(results, DEFAULT_LIVENESS_SWEEP_THRESHOLDS)
    liveness_dists = calculate_liveness_score_distributions(results)
    for cat, dist in liveness_dists.items():
        print(f"Liveness distribution [{cat}]: count={dist.count}, min={dist.min_score}, max={dist.max_score}, mean={dist.mean_score}")

    print("\n--- Checking Physical Replay Category (Separate from Synthetic Speech) ---")
    replay_report = evaluate_replay_category(results)
    print(replay_report.message)
    
    print("\n--- Generating Reports & Plots ---")
    reporter = ReportGenerator(output_dir)
    reporter.write_evaluation_results_csv(results)
    reporter.write_threshold_analysis_csv(metrics)
    reporter.write_liveness_threshold_analysis_csv(liveness_metrics)
    reporter.write_roc_csv(roc_points)
    reporter.write_latency_csv(results)
    reporter.write_antispoof_results_csv(results)
    
    reporter.plot_similarity_distribution(results)
    reporter.plot_far_frr_curve(metrics)
    reporter.plot_roc_curve(metrics)
    reporter.plot_antispoof_distribution(results)
    
    print(f"Evaluation complete! Check the '{output_dir}/' directory for CSVs and Plots.")

if __name__ == "__main__":
    main()
