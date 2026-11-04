import csv
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import statistics

from src.evaluation.runner import EvaluationResultRow
from src.evaluation.metrics import (
    ThresholdMetrics,
    LivenessThresholdMetrics,
    ROCPoint,
)

class ReportGenerator:
    def __init__(self, output_dir: str):
        self.out = Path(output_dir)
        self.results_dir = self.out / "results"
        self.plots_dir = self.out / "plots"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
    def write_evaluation_results_csv(self, rows: List[EvaluationResultRow]):
        path = self.results_dir / "evaluation_results.csv"
        with open(path, 'w', newline='', encoding='utf-8') as f:
            if not rows:
                return
            writer = csv.DictWriter(f, fieldnames=rows[0].__dict__.keys())
            writer.writeheader()
            for r in rows:
                writer.writerow(r.__dict__)
                
    def write_threshold_analysis_csv(self, metrics: List[ThresholdMetrics]):
        path = self.results_dir / "threshold_analysis.csv"
        with open(path, 'w', newline='', encoding='utf-8') as f:
            if not metrics:
                return
            writer = csv.DictWriter(f, fieldnames=metrics[0].__dict__.keys())
            writer.writeheader()
            for m in metrics:
                writer.writerow(m.__dict__)

    def write_liveness_threshold_analysis_csv(self, metrics: List[LivenessThresholdMetrics]):
        path = self.results_dir / "liveness_threshold_analysis.csv"
        with open(path, 'w', newline='', encoding='utf-8') as f:
            if not metrics:
                return
            writer = csv.DictWriter(f, fieldnames=metrics[0].__dict__.keys())
            writer.writeheader()
            for m in metrics:
                writer.writerow(m.__dict__)

    def write_roc_csv(self, roc_points: List[ROCPoint]):
        path = self.results_dir / "roc_curve.csv"
        with open(path, 'w', newline='', encoding='utf-8') as f:
            if not roc_points:
                return
            writer = csv.DictWriter(f, fieldnames=roc_points[0].__dict__.keys())
            writer.writeheader()
            for p in roc_points:
                writer.writerow(p.__dict__)
                
    def write_latency_csv(self, rows: List[EvaluationResultRow]):
        path = self.results_dir / "latency_results.csv"
        fields = ['sample_id', 'vad_ms', 'liveness_ms', 'embedding_ms', 'verification_ms', 'total_ms']
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: getattr(r, k) for k in fields})
                
    def write_antispoof_results_csv(self, rows: List[EvaluationResultRow]):
        path = self.results_dir / "antispoof_results.csv"
        fields = ['sample_id', 'sample_type', 'condition', 'antispoof_score', 'baseline_pass', 'baseline_reason']
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: getattr(r, k) for k in fields})

    def plot_similarity_distribution(self, rows: List[EvaluationResultRow]):
        genuine = [r.similarity_score for r in rows if r.sample_type == 'genuine' and r.similarity_score is not None]
        impostor = [r.similarity_score for r in rows if r.sample_type == 'impostor' and r.similarity_score is not None]
        
        plt.figure(figsize=(10, 6))
        plt.hist(genuine, bins=20, alpha=0.5, label='Genuine', color='blue')
        plt.hist(impostor, bins=20, alpha=0.5, label='Impostor', color='red')
        plt.title("Speaker Similarity Score Distribution")
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Frequency")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.savefig(self.plots_dir / "speaker_similarity_distribution.png")
        plt.close()
        
    def plot_far_frr_curve(self, metrics: List[ThresholdMetrics]):
        thresholds = [m.threshold for m in metrics]
        fars = [m.far for m in metrics]
        frrs = [m.frr for m in metrics]
        
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, fars, label='FAR (False Accept)', marker='o', color='red')
        plt.plot(thresholds, frrs, label='FRR (False Reject)', marker='o', color='blue')
        plt.title("FAR / FRR vs Threshold")
        plt.xlabel("Threshold")
        plt.ylabel("Error Rate")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.plots_dir / "far_frr_curve.png")
        plt.close()

    def plot_roc_curve(self, metrics: List[ThresholdMetrics]):
        fprs = [m.fpr for m in metrics]
        tprs = [m.tpr for m in metrics]
        
        plt.figure(figsize=(10, 6))
        plt.plot(fprs, tprs, label='ROC (TPR vs FPR)', marker='o', color='purple')
        plt.plot([0.0, 1.0], [0.0, 1.0], linestyle='--', color='gray', label='Random Chance')
        plt.title("Speaker Verification ROC Curve")
        plt.xlabel("False Positive Rate (FPR / FAR)")
        plt.ylabel("True Positive Rate (TPR / 1 - FRR)")
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.savefig(self.plots_dir / "roc_curve.png")
        plt.close()

    def plot_antispoof_distribution(self, rows: List[EvaluationResultRow]):
        # Strictly separate synthetic speech attacks from physical replay
        live = [r.antispoof_score for r in rows if r.sample_type == 'genuine' and r.antispoof_score is not None]
        synthetic = [r.antispoof_score for r in rows if r.sample_type == 'synthetic' and r.antispoof_score is not None]
        
        plt.figure(figsize=(10, 6))
        plt.hist(live, bins=20, alpha=0.5, label='Live (Genuine)', color='green')
        plt.hist(synthetic, bins=20, alpha=0.5, label='Synthetic Spoof', color='orange')
        plt.title("Synthetic Speech Liveness Score Distribution")
        plt.xlabel("Liveness Probability")
        plt.ylabel("Frequency")
        plt.legend(loc='upper center')
        plt.grid(True, alpha=0.3)
        plt.savefig(self.plots_dir / "antispoof_score_distribution.png")
        plt.close()
