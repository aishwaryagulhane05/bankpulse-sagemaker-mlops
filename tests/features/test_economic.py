"""Tests for features.economic."""

import pandas as pd
import pytest

from bankpulse_core.feature_base import FeatureError, check_inputs, get_feature
from features.economic import EconomicIndicators, EmploymentLevel, EuriborRate

# Two hand-made rows from different periods; a non-default index checks rows stay aligned.
RAW = pd.DataFrame(
    {
        'euribor3m': [4.857, 0.635],
        'nr.employed': [5191.0, 4991.6],
        'emp.var.rate': [1.1, -3.4],
        'cons.price.idx': [93.994, 92.649],
        'cons.conf.idx': [-36.4, -30.1],
    },
    index=[20, 21],
)

ALLOWED = list(RAW.columns)


@pytest.mark.parametrize('cls, column', [(EuriborRate, 'euribor3m'), (EmploymentLevel, 'nr.employed')])
def test_single_indicator_passes_value_through(cls, column):
    result = cls().fit(RAW).transform(RAW)
    assert list(result.columns) == [cls.name]
    assert result[cls.name].tolist() == RAW[column].tolist()
    assert result.index.equals(RAW.index)


def test_economic_indicators_columns_and_values():
    result = EconomicIndicators().fit(RAW).transform(RAW)
    assert list(result.columns) == [
        'economic_indicators__emp_var_rate',
        'economic_indicators__cons_price_idx',
        'economic_indicators__cons_conf_idx',
    ]
    assert result.loc[21].tolist() == [-3.4, 92.649, -30.1]
    assert (result.dtypes == float).all()
    assert result.index.equals(RAW.index)


def test_missing_indicator_column_is_reported():
    with pytest.raises(FeatureError, match='cons.conf.idx'):
        EconomicIndicators().fit(RAW.drop(columns='cons.conf.idx'))


@pytest.mark.parametrize('name, cls', [
    ('euribor_rate', EuriborRate),
    ('employment_level', EmploymentLevel),
    ('economic_indicators', EconomicIndicators),
])
def test_registered_and_inputs_allowed(name, cls):
    assert get_feature(name, 'v1') is cls
    check_inputs(cls, allowed=ALLOWED, forbidden=['duration'])


@pytest.mark.parametrize('cls', [EuriborRate, EmploymentLevel, EconomicIndicators])
def test_docs_flag_drift(cls):
    assert 'drift' in cls.description and 'drift' in cls.__doc__
