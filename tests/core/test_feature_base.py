"""Tests for bankpulse_core.feature_base."""

import numpy as np
import pandas as pd
import pytest

from bankpulse_core import feature_base
from bankpulse_core.feature_base import Feature, FeatureError, check_inputs, get_feature, register, resolve


@pytest.fixture(autouse=True)
def empty_registry(monkeypatch):
    """Each test starts with an empty registry, so test features don't leak between tests."""
    monkeypatch.setattr(feature_base, '_REGISTRY', {})


def make_feature(name='doubled', version='v1', inputs=('x',)):
    """A minimal stateless feature: twice the input column."""
    class Doubled(Feature):
        description = 'Twice x'
        owner = 'tests'

        def _transform(self, df):
            return df[inputs[0]] * 2

    Doubled.name, Doubled.version, Doubled.inputs = name, version, inputs
    return Doubled


class CappedX(Feature):
    """A feature that learns a value (the training maximum) and saves it as state."""
    name, version, inputs, owner, description = 'capped_x', 'v1', ('x',), 'tests', 'x capped at training max'

    def _fit(self, train):
        self.cap = float(train['x'].max())

    def _transform(self, df):
        return df['x'].clip(upper=self.cap)

    def _state(self):
        return {'cap': self.cap}

    def _load_state(self, state):
        self.cap = state['cap']


DATA = pd.DataFrame({'x': [1.0, 2.0, 3.0], 'other': [0, 0, 0]})


def test_feature_without_transform_cannot_be_created():
    class Incomplete(Feature):
        name, version, inputs = 'incomplete', 'v1', ('x',)

    with pytest.raises(TypeError):
        Incomplete()


def test_transform_before_fit_raises():
    with pytest.raises(FeatureError, match='must be fitted'):
        make_feature()().transform(DATA)


def test_fit_then_transform_returns_named_frame_with_same_index():
    result = make_feature()().fit(DATA).transform(DATA)
    assert list(result.columns) == ['doubled']
    assert result['doubled'].tolist() == [2.0, 4.0, 6.0]
    assert result.index.equals(DATA.index)


def test_missing_input_column_is_named_in_the_error():
    with pytest.raises(FeatureError, match=r"\['x'\]"):
        make_feature()().fit(DATA.drop(columns='x'))


def test_learned_values_come_from_training_data_only():
    feature = CappedX().fit(DATA)
    later = pd.DataFrame({'x': [10.0]})
    assert feature.transform(later)['capped_x'].tolist() == [3.0]


def test_state_round_trip_gives_identical_output():
    trained = CappedX().fit(DATA)
    restored = CappedX().load_state(trained.state())
    later = pd.DataFrame({'x': [0.5, 10.0]})
    pd.testing.assert_frame_equal(trained.transform(later), restored.transform(later))


def test_state_must_be_json_safe():
    class NumpyState(CappedX):
        name = 'numpy_state'

        def _state(self):
            return {'cap': np.float32(self.cap)}

    with pytest.raises(FeatureError, match='not JSON-safe'):
        NumpyState().fit(DATA).state()


def test_output_must_align_with_input_rows():
    class Misaligned(CappedX):
        name = 'misaligned'

        def _transform(self, df):
            return df['x'].reset_index(drop=True).iloc[:1]

    with pytest.raises(FeatureError, match='same index'):
        Misaligned().fit(DATA).transform(DATA)


def test_register_and_get_feature():
    cls = register(make_feature())
    assert get_feature('doubled', 'v1') is cls


def test_duplicate_registration_raises():
    register(make_feature())
    with pytest.raises(FeatureError, match='already registered'):
        register(make_feature())


def test_two_versions_can_coexist():
    v1, v2 = register(make_feature(version='v1')), register(make_feature(version='v2'))
    assert get_feature('doubled', 'v1') is v1 and get_feature('doubled', 'v2') is v2


@pytest.mark.parametrize('kwargs, message', [
    ({'name': ''}, 'name'),
    ({'version': '1'}, 'version'),
    ({'inputs': ()}, 'inputs'),
])
def test_incomplete_definitions_are_rejected(kwargs, message):
    with pytest.raises(FeatureError, match=message):
        register(make_feature(**kwargs))


def test_resolve_builds_unfitted_features_in_config_order():
    register(make_feature(name='a'))
    register(make_feature(name='b'))
    features = resolve({'b': 'v1', 'a': 'v1'})
    assert [f.name for f in features] == ['b', 'a']
    assert not any(f.fitted for f in features)


def test_resolve_unknown_version_lists_known_versions():
    register(make_feature())
    with pytest.raises(FeatureError, match=r"doubled v9 \(registered versions: \['v1'\]\)"):
        resolve({'doubled': 'v9'})


def test_check_inputs_rejects_leakage_column():
    leaky = make_feature(name='call_length', inputs=('duration',))()
    with pytest.raises(FeatureError, match='forbidden'):
        check_inputs(leaky, allowed=['x', 'duration'], forbidden=['duration'])


def test_check_inputs_rejects_column_outside_contract():
    with pytest.raises(FeatureError, match='not in the data contract'):
        check_inputs(make_feature()(), allowed=['age'])


def test_check_inputs_passes_for_allowed_columns():
    check_inputs(make_feature()(), allowed=['x', 'age'], forbidden=['duration'])
