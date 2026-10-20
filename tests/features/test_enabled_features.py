"""Tests for the features enabled in config/data_prep.yaml, checked against config/contract.yaml."""

from pathlib import Path

import json

import pandas as pd
import pytest
import yaml

import features  # noqa: F401  (registers every feature)
from bankpulse_core.feature_base import Feature, FeatureError, check_inputs, resolve

CONFIG_DIR = Path(__file__).resolve().parents[2] / 'config'


def load_yaml(name):
    with open(CONFIG_DIR / name, encoding='utf-8') as handle:
        return yaml.safe_load(handle)


ENABLED = load_yaml('data_prep.yaml')['features']
CONTRACT = load_yaml('contract.yaml')
# Columns features may read: every contract column except the target and forbidden ones.
FORBIDDEN = CONTRACT['forbidden_columns']
ALLOWED = [column for column in CONTRACT['columns'] if column != CONTRACT['target'] and column not in FORBIDDEN]


def test_config_resolves_to_twelve_unfitted_features_in_order():
    resolved = resolve(ENABLED)
    assert len(resolved) == 12
    assert all(isinstance(feature, Feature) and not feature.fitted for feature in resolved)
    assert [feature.name for feature in resolved] == list(ENABLED)


@pytest.mark.parametrize('feature', resolve(ENABLED), ids=lambda feature: feature.key())
def test_enabled_feature_passes_contract(feature):
    check_inputs(feature, allowed=ALLOWED, forbidden=FORBIDDEN)


def test_contract_lists_every_dataset_column():
    header = (CONFIG_DIR.parent / 'data/raw/bank-additional-full.csv').open(encoding='utf-8').readline()
    assert list(CONTRACT['columns']) == [name.strip('"') for name in header.strip().split(';')]
    assert len(ALLOWED) == 19


def test_invalid_version_raises_clear_error():
    with pytest.raises(FeatureError, match=r"Unknown feature age v9 \(registered versions: \['v1'\]\)"):
        resolve({**ENABLED, 'age': 'v9'})


# ---- Phase 2 review: learned values come from training data only ----------------------

# Two hand-made training rows with every contract column.
TRAIN = pd.DataFrame({
    'age': [25, 50], 'job': ['admin.', 'technician'], 'marital': ['married', 'single'],
    'education': ['basic.4y', 'unknown'], 'default': ['no', 'no'], 'housing': ['yes', 'no'],
    'loan': ['no', 'yes'], 'contact': ['cellular', 'telephone'], 'month': ['may', 'jun'],
    'day_of_week': ['mon', 'fri'], 'campaign': [1, 3], 'pdays': [999, 6], 'previous': [0, 1],
    'poutcome': ['nonexistent', 'success'], 'emp.var.rate': [1.1, -1.8], 'cons.price.idx': [93.9, 92.9],
    'cons.conf.idx': [-36.4, -46.2], 'euribor3m': [4.86, 1.31], 'nr.employed': [5191.0, 5099.1],
})
# Later rows look very different: extreme counts and categories never seen in training.
LATER = TRAIN.assign(campaign=[40, 56], job=['student', 'student'], month=['sep', 'sep'])


def fit_all():
    return [feature.fit(TRAIN) for feature in resolve(ENABLED)]


def test_transforming_later_data_never_changes_learned_state():
    fitted = fit_all()
    before = [feature.state() for feature in fitted]
    for feature in fitted:
        feature.transform(LATER)
    assert [feature.state() for feature in fitted] == before


def test_learned_values_match_training_data():
    states = {feature.name: feature.state() for feature in fit_all()}
    # The cap comes from TRAIN (1 and 3), not from LATER's 40 and 56.
    assert states['capped_campaign']['cap'] <= TRAIN['campaign'].max()
    # Category lists hold only training values; "student" and "sep" are not in them.
    assert states['categorical_profile']['job']['categories'] == sorted(TRAIN['job'])
    assert 'sep' not in states['contact_timing']['month']['categories']


def test_all_states_are_json_safe_and_reload_to_identical_output():
    for trained in fit_all():
        restored = type(trained)().load_state(json.loads(json.dumps(trained.state())))
        pd.testing.assert_frame_equal(trained.transform(LATER), restored.transform(LATER))
