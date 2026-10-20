"""Tests for the features enabled in config/data_prep.yaml, checked against config/contract.yaml."""

from pathlib import Path

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


def test_config_resolves_to_twelve_unfitted_features_in_order():
    resolved = resolve(ENABLED)
    assert len(resolved) == 12
    assert all(isinstance(feature, Feature) and not feature.fitted for feature in resolved)
    assert [feature.name for feature in resolved] == list(ENABLED)


@pytest.mark.parametrize('feature', resolve(ENABLED), ids=lambda feature: feature.key())
def test_enabled_feature_passes_contract(feature):
    check_inputs(feature, allowed=CONTRACT['allowed_columns'], forbidden=CONTRACT['forbidden_columns'])


def test_contract_keeps_target_and_leakage_out_of_allowed_columns():
    allowed = set(CONTRACT['allowed_columns'])
    assert CONTRACT['target'] not in allowed
    assert not allowed & set(CONTRACT['forbidden_columns'])


def test_invalid_version_raises_clear_error():
    with pytest.raises(FeatureError, match=r"Unknown feature age v9 \(registered versions: \['v1'\]\)"):
        resolve({**ENABLED, 'age': 'v9'})
