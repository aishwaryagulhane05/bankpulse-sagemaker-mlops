"""Tests for bankpulse_core.monitoring."""

import base64
import json

import numpy as np
import pandas as pd
import pytest

from bankpulse_core.monitoring import (MonitoringError, build_report, data_drift, group_from_one_hot,
                                       parse_capture, prediction_drift, psi, subgroup_rates)

RNG = np.random.default_rng(0)
REFERENCE = RNG.normal(0, 1, 5000)


# ---- psi ----------------------------------------------------------------------------------

def test_identical_data_has_psi_near_zero():
    assert psi(REFERENCE, RNG.normal(0, 1, 5000)) < 0.02


def test_shifted_data_has_psi_above_alert_level():
    assert psi(REFERENCE, RNG.normal(1, 1, 5000)) > 0.2


def test_values_beyond_reference_range_are_counted():
    # Every live value is above the training maximum: a large shift, not "no data".
    assert psi(REFERENCE, np.full(1000, 10.0)) > 1


def test_binary_column_uses_value_shares():
    reference = np.array([0] * 90 + [1] * 10)
    assert psi(reference, reference) == pytest.approx(0, abs=1e-9)
    assert psi(reference, np.array([0] * 50 + [1] * 50)) > 0.2


def test_unseen_category_value_counts_as_drift():
    reference = np.array([1, 2, 3] * 100)
    assert psi(reference, np.array([1, 2, 3, 4] * 75)) > 0.2


def test_empty_input_is_rejected():
    with pytest.raises(MonitoringError, match='non-empty'):
        psi([], [1.0])


# ---- data and prediction drift --------------------------------------------------------------

def test_data_drift_ranks_columns_and_flags_drifted_ones():
    reference = pd.DataFrame({'stable': REFERENCE, 'moved': REFERENCE})
    current = pd.DataFrame({'stable': RNG.normal(0, 1, 5000), 'moved': RNG.normal(2, 1, 5000)})
    drift = data_drift(reference, current)
    assert list(drift['psi']) == ['moved', 'stable']
    assert drift['drifted_columns'] == ['moved'] and drift['max_psi'] == drift['psi']['moved']


def test_data_drift_requires_the_same_columns():
    with pytest.raises(MonitoringError, match='missing columns'):
        data_drift(pd.DataFrame({'a': [1.0]}), pd.DataFrame({'b': [1.0]}))


def test_prediction_drift_status():
    scores = RNG.beta(2, 8, 5000)
    assert prediction_drift(scores, RNG.beta(2, 8, 5000))['status'] == 'stable'
    shifted = prediction_drift(scores, RNG.beta(5, 5, 5000))
    assert shifted['status'] == 'alert' and shifted['current_mean'] > shifted['reference_mean']


# ---- subgroups -------------------------------------------------------------------------------

def test_group_from_one_hot():
    matrix = pd.DataFrame({'age_band__lt30': [1, 0, 0], 'age_band__60plus': [0, 1, 0], 'other': [5, 5, 5]})
    assert group_from_one_hot(matrix, 'age_band').tolist() == ['lt30', '60plus', 'none']


def test_subgroup_rates_and_ratio():
    result = subgroup_rates([0.9, 0.1, 0.8, 0.7, 0.2, 0.1], ['young', 'young', 'old', 'old', 'old', 'old'], 0.5)
    assert result['groups']['young'] == {'rows': 2, 'selected_rate': 0.5}
    assert result['groups']['old'] == {'rows': 4, 'selected_rate': 0.5}
    assert result['min_max_ratio'] == 1.0 and result['status'] == 'ok'
    unfair = subgroup_rates([0.9, 0.8, 0.1, 0.2], ['a', 'a', 'b', 'b'], 0.5)
    assert unfair['min_max_ratio'] == 0.0 and unfair['status'] == 'alert'


# ---- Data Capture ---------------------------------------------------------------------------

def capture_line(rows, scores, base64_output=False):
    output = '\n'.join(str(s) for s in scores)
    return json.dumps({
        'captureData': {
            'endpointInput': {'observedContentType': 'text/csv', 'mode': 'INPUT', 'encoding': 'CSV',
                              'data': '\n'.join(','.join(str(v) for v in row) for row in rows)},
            'endpointOutput': {'observedContentType': 'text/csv', 'mode': 'OUTPUT',
                               'encoding': 'BASE64' if base64_output else 'CSV',
                               'data': base64.b64encode(output.encode()).decode() if base64_output else output},
        },
        'eventMetadata': {'eventId': 'e1', 'inferenceTime': '2026-10-21T00:00:00Z'},
        'eventVersion': '0',
    })


def test_parse_capture_reads_csv_and_base64_records(tmp_path):
    path = tmp_path / 'capture.jsonl'
    path.write_text(capture_line([[1, 0.5], [0, 2.5]], [0.1, 0.9]) + '\n'
                    + capture_line([[1, 1.5]], [0.4], base64_output=True) + '\n')
    features, scores = parse_capture([path], ['a', 'b'])
    assert features.values.tolist() == [[1, 0.5], [0, 2.5], [1, 1.5]]
    assert scores.tolist() == [0.1, 0.9, 0.4]


def test_comma_separated_scores_are_accepted(tmp_path):
    line = json.loads(capture_line([[1, 0.5], [0, 2.5]], [0.1, 0.9]))
    line['captureData']['endpointOutput']['data'] = '0.1,0.9'
    path = tmp_path / 'capture.jsonl'
    path.write_text(json.dumps(line))
    assert parse_capture([path], ['a', 'b'])[1].tolist() == [0.1, 0.9]


@pytest.mark.parametrize('rows, scores, columns, message', [
    ([[1, 2]], [0.1, 0.2], ['a', 'b'], '1 input rows but 2 scores'),
    ([[1, 2, 3]], [0.1], ['a', 'b'], 'expected 2 values per row'),
])
def test_malformed_capture_is_rejected(tmp_path, rows, scores, columns, message):
    path = tmp_path / 'capture.jsonl'
    path.write_text(capture_line(rows, scores))
    with pytest.raises(MonitoringError, match=message):
        parse_capture([path], columns)


# ---- report ----------------------------------------------------------------------------------

def test_report_is_json_safe_and_lists_alerts():
    reference = pd.DataFrame({'x': REFERENCE, 'age_band__lt30': (REFERENCE > 0).astype(int),
                              'age_band__60plus': (REFERENCE <= 0).astype(int)})
    current = reference.assign(x=reference['x'] + 2)
    scores = RNG.beta(2, 8, 5000)
    report = build_report(reference, current, scores, scores + 0.3,
                          groups=group_from_one_hot(current, 'age_band'), threshold=0.3)
    json.dumps(report)
    assert set(report['metrics']) == {'max_feature_psi', 'prediction_psi', 'positive_rate', 'subgroup_min_max_ratio'}
    assert report['records'] == 5000
    assert any(alert.startswith('data drift in 1 columns') for alert in report['alerts'])
    assert any(alert.startswith('prediction drift') for alert in report['alerts'])
