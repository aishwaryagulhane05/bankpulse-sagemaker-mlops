"""Evaluation: parse Batch Transform output, choose the threshold, compute metrics, compare with the baseline.

    scores = parse_batch_output('validation_features.csv.out')
    threshold = choose_threshold(y_validation, validation_scores)     # validation only
    metrics = evaluate(y_test, test_scores, threshold)                 # test, once
    compare(metrics, baseline_metrics)

About 11% of customers subscribe, so accuracy is misleading (predicting "no" for everyone
scores 89%). The headline metric is PR-AUC; precision, recall and F1 describe the chosen
threshold, and lift says how much better than random the top-scored customers are.
"""

from typing import Dict, Mapping, Sequence, Union

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score

from bankpulse_core.io import PathLike, read_bytes

# Metrics where a higher value is better, compared by compare().
COMPARED = ('pr_auc', 'roc_auc', 'precision', 'recall', 'f1', 'lift_top_decile')

ArrayLike = Union[Sequence[float], np.ndarray]


class EvaluationError(ValueError):
    """Scores or labels are malformed, or don't line up."""


def parse_batch_output(path: PathLike) -> np.ndarray:
    """Probabilities from a Batch Transform .out file (local or S3), one per input row.

    Built-in XGBoost with binary:logistic writes one probability per line.
    """
    lines = [line.strip() for line in read_bytes(path).decode('utf-8').splitlines() if line.strip()]
    try:
        # Step 1: a line may be "0.42" or, with some content types, "0.42," - take the first value.
        scores = np.array([float(line.split(',')[0]) for line in lines])
    except ValueError as exc:
        raise EvaluationError(f'{path}: every line must be a probability: {exc}') from exc
    # Step 2: probabilities outside [0, 1] mean the wrong objective or output was read.
    if len(scores) == 0 or ((scores < 0) | (scores > 1)).any():
        raise EvaluationError(f'{path}: expected probabilities in [0, 1], got {len(scores)} values '
                              f'in [{scores.min() if len(scores) else "-"}, {scores.max() if len(scores) else "-"}]')
    return scores


def _check(y_true: ArrayLike, scores: ArrayLike):
    y_true, scores = np.asarray(y_true), np.asarray(scores, dtype=float)
    if len(y_true) != len(scores):
        raise EvaluationError(f'{len(y_true)} labels but {len(scores)} scores: rows do not line up')
    if not set(np.unique(y_true)) <= {0, 1} or len(np.unique(y_true)) < 2:
        raise EvaluationError('Labels must be 0/1 and contain both classes')
    return y_true, scores


def choose_threshold(y_true: ArrayLike, scores: ArrayLike) -> float:
    """The score threshold with the highest F1 on these (validation) labels.

    Choose it on validation and reuse it on test: choosing on test would tune the reported
    result to the test set.
    """
    y_true, scores = _check(y_true, scores)
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # The last precision/recall pair has no threshold, so drop it before matching.
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


def evaluate(y_true: ArrayLike, scores: ArrayLike, threshold: float) -> Dict:
    """Ranking metrics (threshold-free) and classification metrics at threshold, JSON-safe."""
    y_true, scores = _check(y_true, scores)
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_true, predicted, labels=[0, 1]).ravel())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    # Lift: subscription rate among the 10% highest-scored customers vs the overall rate.
    top = np.argsort(-scores, kind='stable')[:max(1, len(scores) // 10)]
    lift = float(y_true[top].mean() / y_true.mean())
    return {
        'rows': int(len(y_true)),
        'positive_rate': round(float(y_true.mean()), 4),
        'pr_auc': round(float(average_precision_score(y_true, scores)), 4),
        'roc_auc': round(float(roc_auc_score(y_true, scores)), 4),
        'threshold': round(float(threshold), 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
        'lift_top_decile': round(lift, 3),
        'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
    }


def compare(candidate: Mapping, baseline: Mapping) -> Dict:
    """Per-metric difference (candidate - baseline) and which model is better on each."""
    result = {}
    for metric in COMPARED:
        difference = round(candidate[metric] - baseline[metric], 4)
        result[metric] = {'candidate': candidate[metric], 'baseline': baseline[metric], 'difference': difference,
                          'better': 'candidate' if difference > 0 else 'baseline' if difference < 0 else 'tie'}
    # PR-AUC decides overall: it measures ranking quality on the rare positive class.
    result['candidate_beats_baseline'] = result['pr_auc']['difference'] > 0
    return result
