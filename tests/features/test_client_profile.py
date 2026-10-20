"""Tests for features.client_profile."""

import json

import pandas as pd
import pytest

from bankpulse_core.feature_base import check_inputs, get_feature
from features.client_profile import Age, AgeBand, CategoricalProfile, ContactTiming

PROFILE = ['job', 'marital', 'education', 'default', 'housing', 'loan']
TIMING = ['contact', 'month', 'day_of_week']

# Hand-made training rows. "unknown" is a real category in this dataset.
TRAIN = pd.DataFrame({
    'age': [25, 40],
    'job': ['admin.', 'technician'],
    'marital': ['married', 'single'],
    'education': ['university.degree', 'unknown'],
    'default': ['no', 'unknown'],
    'housing': ['yes', 'no'],
    'loan': ['no', 'no'],
    'contact': ['cellular', 'telephone'],
    'month': ['may', 'jun'],
    'day_of_week': ['mon', 'tue'],
})

# A later row with values never seen in training: job "student" and month "sep".
LATER = TRAIN.iloc[[0]].assign(job='student', month='sep').set_index(pd.Index([99]))


# ---- age and age_band ------------------------------------------------------------

def test_age_is_raw():
    assert Age().fit(TRAIN).transform(TRAIN)['age'].tolist() == [25, 40]


@pytest.mark.parametrize('age, band', [
    (17, 'lt30'), (29, 'lt30'),
    (30, '30_44'), (44, '30_44'),
    (45, '45_59'), (59, '45_59'),
    (60, '60plus'), (98, '60plus'),
])
def test_age_band_boundaries(age, band):
    result = AgeBand().fit(TRAIN).transform(pd.DataFrame({'age': [age]}))
    assert result.columns.tolist() == ['age_band__lt30', 'age_band__30_44', 'age_band__45_59', 'age_band__60plus']
    # Exactly one band is set, and it's the right one.
    assert result.iloc[0].sum() == 1 and result.loc[0, f'age_band__{band}'] == 1


# ---- one-hot profile and timing ---------------------------------------------------

def test_categories_are_learned_from_training_only():
    feature = CategoricalProfile().fit(TRAIN)
    feature.transform(LATER)                      # must not add "student" to the list
    assert feature.state()['job']['categories'] == ['admin.', 'technician']


def test_unseen_category_goes_to_other():
    profile = CategoricalProfile().fit(TRAIN).transform(LATER)
    timing = ContactTiming().fit(TRAIN).transform(LATER)
    assert profile.loc[99, 'categorical_profile__job__other'] == 1
    assert timing.loc[99, 'contact_timing__month__other'] == 1
    assert timing.loc[99, 'contact_timing__month__may'] == 0


def test_unknown_is_its_own_category_not_other():
    result = CategoricalProfile().fit(TRAIN).transform(TRAIN.iloc[[1]])
    assert result.iloc[0]['categorical_profile__education__unknown'] == 1
    assert result.iloc[0]['categorical_profile__education__other'] == 0


@pytest.mark.parametrize('cls, columns', [(CategoricalProfile, PROFILE), (ContactTiming, TIMING)])
def test_column_names_follow_feature_column_category(cls, columns):
    result = cls().fit(TRAIN).transform(TRAIN)
    assert all(name.startswith(f'{cls.name}__') for name in result.columns)
    # Raw columns appear in the declared input order.
    prefixes = [name.split('__')[1] for name in result.columns]
    assert list(dict.fromkeys(prefixes)) == columns
    # Exactly one hot column per raw column in every row.
    assert (result.sum(axis=1) == len(columns)).all()
    assert result.index.equals(TRAIN.index)


@pytest.mark.parametrize('cls', [CategoricalProfile, ContactTiming])
def test_state_round_trips_through_json(cls):
    trained = cls().fit(TRAIN)
    restored = cls().load_state(json.loads(json.dumps(trained.state())))
    pd.testing.assert_frame_equal(trained.transform(LATER), restored.transform(LATER))


# ---- registration -------------------------------------------------------------------

@pytest.mark.parametrize('name, cls', [
    ('age', Age), ('age_band', AgeBand),
    ('categorical_profile', CategoricalProfile), ('contact_timing', ContactTiming),
])
def test_registered_and_inputs_allowed(name, cls):
    assert get_feature(name, 'v1') is cls
    check_inputs(cls, allowed=list(TRAIN.columns), forbidden=['duration'])
