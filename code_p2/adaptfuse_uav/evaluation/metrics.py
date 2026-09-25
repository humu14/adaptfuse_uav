"""
AdapFuse-UAV: Metrics computation.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    precision_score, recall_score,
)
from typing import List, Dict, Optional


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_proba: Optional[List] = None,
    num_classes: int = 4,
    prefix: str = "",
) -> Dict[str, float]:
    """Compute all classification metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{prefix}precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

    if y_proba is not None:
        y_proba = np.array(y_proba)
        try:
            if num_classes == 2:
                auroc = roc_auc_score(y_true, y_proba[:, 1])
            else:
                auroc = roc_auc_score(
                    y_true, y_proba,
                    multi_class="ovr", average="macro",
                    labels=list(range(num_classes))
                )
            metrics[f"{prefix}auroc"] = auroc
        except Exception:
            metrics[f"{prefix}auroc"] = 0.0

    return metrics


def format_metrics_table(metrics: Dict[str, float], title: str = "Metrics") -> str:
    """Format metrics dict as a readable table string."""
    lines = [f"\n{'='*50}", f"  {title}", f"{'='*50}"]
    for k, v in sorted(metrics.items()):
        lines.append(f"  {k:<30} {v:.4f}")
    lines.append(f"{'='*50}")
    return "\n".join(lines)


def print_classification_report(y_true, y_pred, class_names=None):
    """Print sklearn classification report."""
    if class_names is None:
        class_names = [str(i) for i in range(max(y_pred) + 1)]
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    EXP-9: Expected Calibration Error (ECE). Lower is better; perfect = 0.
    probs: (N, C) softmax probabilities
    labels: (N,) integer class labels
    """
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 4) -> float:
    """
    EXP-5: False Positive Rate — fraction of clean (class-0) samples misclassified as disaster.
    Row 0 in confusion matrix = true-negative class.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    if cm[0].sum() == 0:
        return 0.0
    return float(cm[0, 1:].sum() / cm[0].sum())
