"""Tests for bankpulse_core.evaluation."""

import json

import numpy as np
import pytest

from bankpulse_core.evaluation import EvaluationError, choose_threshold, compare, evaluate, parse_batch_output

# Ten customers, three subscribers; scores rank two of them on top.
Y = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0])
SCORES = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])


def test_parse_batch_output(tmp_path):
    path = tmp_path / 'test_features.csv.out'
    path.write_text('0.12\n0.98\n\n0.5,\n')
    assert parse_batch_output(path).tolist() == [0.12, 0.98, 0.5]


@pytest.mark.parametrize('text, message', [('0.1\nabc\n', 'must be a probability'), ('0.1\n3.2\n', r'\[0, 1\]'),
                                           ('', r'\[0, 1\]')])
def test_bad_batch_output_is_rejected(tmp_path, text, message):
    path = tmp_path / 'bad.out'
    path.write_text(text)
    with pytest.raises(EvaluationError, match=message):
        parse_batch_output(path)


def test_threshold_maximises_f1():
    # Threshold 0.8 keeps the top two (both positive): precision 1, recall 2/3, F1 0.8 - the best.
    assert choose_threshold(Y, SCORES) == 0.8


def test_evaluate_known_scores():
    metrics = evaluate(Y, SCORES, threshold=0.5)
    assert metrics['confusion_matrix'] == {'tn': 5, 'fp': 2, 'fn': 0, 'tp': 3}
    assert metrics['precision'] == 0.6 and metrics['recall'] == 1.0 and metrics['f1'] == 0.75
    # Average precision: (1/1 + 2/2 + 3/5) / 3
    assert metrics['pr_auc'] == round((1 + 1 + 0.6) / 3, 4)
    # Top decile = 1 customer, a subscriber: 100% vs 30% overall.
    assert metrics['lift_top_decile'] == round(1 / 0.3, 3)
    json.dumps(metrics)


def test_perfect_and_random_rankings():
    perfect = evaluate(Y, Y * 0.9 + 0.05, threshold=0.5)
    assert perfect['pr_auc'] == 1.0 and perfect['roc_auc'] == 1.0 and perfect['f1'] == 1.0


def test_threshold_above_every_score_predicts_no_one():
    metrics = evaluate(Y, SCORES, threshold=0.99)
    assert metrics['precision'] == 0.0 and metrics['recall'] == 0.0 and metrics['f1'] == 0.0


@pytest.mark.parametrize('labels, scores, message', [
    (Y[:5], SCORES, 'do not line up'),
    (np.zeros(10), SCORES, 'both classes'),
    (Y * 2, SCORES, 'must be 0/1'),
])
def test_bad_inputs_are_rejected(labels, scores, message):
    with pytest.raises(EvaluationError, match=message):
        evaluate(labels, scores, 0.5)


def test_compare_reports_differences_and_winner():
    better = evaluate(Y, Y * 0.9 + 0.05, threshold=0.5)
    baseline = evaluate(Y, SCORES, threshold=0.5)
    result = compare(better, baseline)
    assert result['pr_auc']['better'] == 'candidate' and result['pr_auc']['difference'] > 0
    assert result['recall']['better'] == 'tie'
    assert result['candidate_beats_baseline'] is True
    assert compare(baseline, better)['candidate_beats_baseline'] is False
