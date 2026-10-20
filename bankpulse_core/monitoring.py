"""Monitoring: data drift, prediction drift and subgroup checks.

    features, scores = parse_capture(capture_files, schema['columns'])
    report = build_report(reference_features, features, reference_scores, scores,
                          groups=group_from_one_hot(features, 'age_band'), threshold=threshold)

Drift is measured with the Population Stability Index (PSI) between a reference sample
(the training features and validation scores) and live data (SageMaker Data Capture):

    PSI < 0.1    stable
    0.1 - 0.2    moderate shift, watch
    > 0.2        significant shift, investigate / consider retraining

Everything returned is JSON-safe, so reports can be saved to S3 and the headline numbers
published as CloudWatch metrics.
"""

import base64
import json
import re
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from bankpulse_core.io import PathLike, read_bytes

PSI_WARN, PSI_ALERT = 0.1, 0.2
# Subgroup positive-rate ratio below this flags possible disparate impact (the "80% rule").
SUBGROUP_RATIO_ALERT = 0.8
# Floor for empty bins, so log(0) never happens and a single empty bin can't dominate.
_EPSILON = 1e-4


class MonitoringError(ValueError):
    """Capture data or monitoring inputs are malformed."""


# ---- Data Capture --------------------------------------------------------------------------

def _decode(part: Mapping) -> str:
    data = part.get('data', '')
    return base64.b64decode(data).decode('utf-8') if part.get('encoding') == 'BASE64' else data


def _csv_rows(text: str) -> List[List[float]]:
    return [[float(value) for value in line.split(',')] for line in text.strip().splitlines() if line.strip()]


def parse_capture(paths: Iterable[PathLike], columns: Sequence[str]) -> Tuple[pd.DataFrame, pd.Series]:
    """Features sent to the endpoint and the scores it returned, from Data Capture .jsonl files.

    Each line of a capture file is one request: endpointInput holds the CSV rows sent,
    endpointOutput the probabilities returned (plain CSV, or BASE64-encoded).
    """
    features, scores = [], []
    for path in paths:
        for number, line in enumerate(read_bytes(path).decode('utf-8').splitlines(), start=1):
            if not line.strip():
                continue
            # Step 1: decode one captured request and its response.
            record = json.loads(line)
            capture = record['captureData']
            rows = _csv_rows(_decode(capture['endpointInput']))
            # XGBoost returns one score per row, separated by newlines or commas.
            outputs = [float(v) for v in re.split(r'[,\n]', _decode(capture['endpointOutput'])) if v.strip()]
            # Step 2: one score per row, and every row has the schema's columns.
            if len(rows) != len(outputs):
                raise MonitoringError(f'{path} line {number}: {len(rows)} input rows but {len(outputs)} scores')
            if any(len(row) != len(columns) for row in rows):
                raise MonitoringError(f'{path} line {number}: expected {len(columns)} values per row')
            features += rows
            scores += outputs
    if not features:
        raise MonitoringError('No captured records found')
    return pd.DataFrame(features, columns=list(columns)), pd.Series(scores, name='score')


# ---- drift -------------------------------------------------------------------------------

def psi(expected: Sequence[float], actual: Sequence[float], bins: int = 10) -> float:
    """Population Stability Index of actual vs expected (the reference distribution).

    Columns with few distinct values (0/1 flags, small counts) are compared value by value;
    continuous columns use quantile bins of the reference, so each bin holds ~10% of it.
    """
    expected, actual = np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)
    if len(expected) == 0 or len(actual) == 0:
        raise MonitoringError('psi() needs non-empty reference and current data')

    values = np.unique(expected)
    if len(values) <= bins:
        # Step 1a: categorical-like - one "bin" per reference value, plus one for values never
        # seen in the reference (e.g. a month that appears only in production).
        expected_share = np.array([np.mean(expected == v) for v in values] + [0.0])
        actual_share = np.array([np.mean(actual == v) for v in values] + [np.mean(~np.isin(actual, values))])
    else:
        # Step 1b: continuous - inner edges from reference quantiles; open-ended outer bins
        # catch live values beyond the reference range.
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)[1:-1]))
        expected_share = np.bincount(np.searchsorted(edges, expected, side='right'), minlength=len(edges) + 1)
        actual_share = np.bincount(np.searchsorted(edges, actual, side='right'), minlength=len(edges) + 1)
        expected_share, actual_share = expected_share / len(expected), actual_share / len(actual)

    # Step 2: sum over bins of (actual - expected) * ln(actual / expected).
    expected_share = np.clip(expected_share, _EPSILON, None)
    actual_share = np.clip(actual_share, _EPSILON, None)
    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


def _status(value: float) -> str:
    return 'alert' if value > PSI_ALERT else 'warn' if value > PSI_WARN else 'stable'


def data_drift(reference: pd.DataFrame, current: pd.DataFrame,
               columns: Optional[Sequence[str]] = None) -> Dict:
    """PSI per feature column, sorted by drift, with the columns above the alert level."""
    columns = list(columns or reference.columns)
    missing = [c for c in columns if c not in current.columns]
    if missing:
        raise MonitoringError(f'Current data is missing columns {missing[:5]}')
    scores = {c: round(psi(reference[c], current[c]), 4) for c in columns}
    ranked = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    return {
        'max_psi': max(ranked.values()),
        'drifted_columns': [c for c, value in ranked.items() if value > PSI_ALERT],
        'psi': ranked,
    }


def prediction_drift(reference_scores: Sequence[float], current_scores: Sequence[float]) -> Dict:
    """Shift in the model's output: PSI of the scores and the change in mean score."""
    reference, current = np.asarray(reference_scores, dtype=float), np.asarray(current_scores, dtype=float)
    value = round(psi(reference, current), 4)
    return {'psi': value, 'status': _status(value), 'reference_mean': round(float(reference.mean()), 4),
            'current_mean': round(float(current.mean()), 4)}


# ---- subgroups ------------------------------------------------------------------------------

def group_from_one_hot(matrix: pd.DataFrame, feature: str) -> pd.Series:
    """Group label per row from a one-hot feature, e.g. 'age_band' -> '30_44'."""
    columns = [c for c in matrix.columns if c.startswith(f'{feature}__')]
    if not columns:
        raise MonitoringError(f'No one-hot columns for {feature!r}')
    labels = matrix[columns].idxmax(axis=1).str.slice(len(feature) + 2)
    return labels.where(matrix[columns].sum(axis=1) > 0, 'none')


def subgroup_rates(scores: Sequence[float], groups: Sequence[str], threshold: float) -> Dict:
    """Share of customers the model would call (score >= threshold) in each group.

    ratio = lowest group rate / highest group rate. Below 0.8, one group is selected far
    less often than another, which should be reviewed for fairness.
    """
    frame = pd.DataFrame({'selected': np.asarray(scores, dtype=float) >= threshold, 'group': list(groups)})
    by_group = frame.groupby('group')['selected'].agg(['size', 'mean'])
    rates = {str(group): {'rows': int(row['size']), 'selected_rate': round(float(row['mean']), 4)}
             for group, row in by_group.iterrows()}
    highest = max(item['selected_rate'] for item in rates.values())
    lowest = min(item['selected_rate'] for item in rates.values())
    ratio = round(lowest / highest, 4) if highest else 1.0
    return {'threshold': threshold, 'groups': rates, 'min_max_ratio': ratio,
            'status': 'alert' if ratio < SUBGROUP_RATIO_ALERT else 'ok'}


# ---- report ---------------------------------------------------------------------------------

def build_report(reference_features: pd.DataFrame, current_features: pd.DataFrame,
                 reference_scores: Sequence[float], current_scores: Sequence[float],
                 groups: Sequence[str], threshold: float) -> Dict:
    """One JSON-safe monitoring report with the headline metrics for CloudWatch."""
    drift = data_drift(reference_features, current_features)
    predictions = prediction_drift(reference_scores, current_scores)
    subgroups = subgroup_rates(current_scores, groups, threshold)

    alerts = []
    if drift['drifted_columns']:
        alerts.append(f'data drift in {len(drift["drifted_columns"])} columns (max PSI {drift["max_psi"]})')
    if predictions['status'] == 'alert':
        alerts.append(f'prediction drift (PSI {predictions["psi"]})')
    if subgroups['status'] == 'alert':
        alerts.append(f'subgroup selection ratio {subgroups["min_max_ratio"]} below {SUBGROUP_RATIO_ALERT}')

    current = np.asarray(current_scores, dtype=float)
    return {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'records': int(len(current_features)),
        # Headline numbers, one CloudWatch metric each.
        'metrics': {
            'max_feature_psi': drift['max_psi'],
            'prediction_psi': predictions['psi'],
            'positive_rate': round(float((current >= threshold).mean()), 4),
            'subgroup_min_max_ratio': subgroups['min_max_ratio'],
        },
        'data_drift': drift,
        'prediction_drift': predictions,
        'subgroups': subgroups,
        'alerts': alerts,
    }
