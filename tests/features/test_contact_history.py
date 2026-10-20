"""Tests for features.contact_history."""

import pandas as pd
import pytest

from bankpulse_core.feature_base import check_inputs, get_feature
from features.contact_history import PreviousContacts, PreviouslyContacted, PriorSuccess

# Hand-made rows covering each case; a non-default index checks rows stay aligned.
RAW = pd.DataFrame(
    {
        'pdays': [999, 3, 0, 999],
        'poutcome': ['nonexistent', 'success', 'failure', 'success'],
        'previous': [0, 1, 2, 1],
        'duration': [100, 200, 300, 400],
    },
    index=[10, 11, 12, 13],
)

# Columns the data contract allows (subset that matters here), and the leakage column.
ALLOWED = ['pdays', 'poutcome', 'previous', 'campaign', 'age']
FORBIDDEN = ['duration']

FEATURES = [PreviouslyContacted, PriorSuccess, PreviousContacts]


@pytest.mark.parametrize('cls, expected', [
    (PreviouslyContacted, [0, 1, 1, 0]),   # pdays 0 is a real value (contacted the same day)
    (PriorSuccess, [0, 1, 0, 1]),
    (PreviousContacts, [0, 1, 2, 1]),
])
def test_values_on_hand_made_rows(cls, expected):
    result = cls().fit(RAW).transform(RAW)
    assert list(result.columns) == [cls.name]
    assert result[cls.name].tolist() == expected
    assert result.index.equals(RAW.index)


@pytest.mark.parametrize('name, cls', [
    ('previously_contacted', PreviouslyContacted),
    ('prior_success', PriorSuccess),
    ('previous_contacts', PreviousContacts),
])
def test_registered_under_name_and_v1(name, cls):
    assert get_feature(name, 'v1') is cls


@pytest.mark.parametrize('cls', FEATURES)
def test_inputs_are_in_contract_and_not_leaky(cls):
    check_inputs(cls, allowed=ALLOWED, forbidden=FORBIDDEN)


@pytest.mark.parametrize('cls', FEATURES)
def test_stateless_features_round_trip_with_empty_state(cls):
    trained = cls().fit(RAW)
    assert trained.state() == {}
    restored = cls().load_state(trained.state())
    pd.testing.assert_frame_equal(trained.transform(RAW), restored.transform(RAW))
