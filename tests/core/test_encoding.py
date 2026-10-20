"""Tests for bankpulse_core.encoding."""

import json

import numpy as np
import pandas as pd
import pytest

from bankpulse_core.encoding import OTHER, CategoryEncoder, Capper, log1p, sentinel_flag

TRAIN_JOBS = pd.Series(['admin.', 'technician', 'unknown', 'admin.'], name='job')


# ---- CategoryEncoder ---------------------------------------------------------------

def test_encoder_learns_sorted_categories_and_fixed_columns():
    encoder = CategoryEncoder('job').fit(TRAIN_JOBS)
    assert encoder.categories == ['admin.', 'technician', 'unknown']
    assert encoder.columns == ['job__admin.', 'job__technician', 'job__unknown', f'job__{OTHER}']


def test_unseen_category_goes_to_other_and_unknown_stays_its_own_category():
    encoder = CategoryEncoder('job').fit(TRAIN_JOBS)
    result = encoder.transform(pd.Series(['unknown', 'astronaut'], index=[10, 11]))
    assert result.loc[10, 'job__unknown'] == 1 and result.loc[10, f'job__{OTHER}'] == 0
    assert result.loc[11, f'job__{OTHER}'] == 1 and result.loc[11].sum() == 1
    assert result.index.tolist() == [10, 11]


def test_every_row_has_exactly_one_hot_column():
    encoder = CategoryEncoder('job').fit(TRAIN_JOBS)
    result = encoder.transform(pd.Series(['admin.', 'new', 'technician']))
    assert result.sum(axis=1).tolist() == [1, 1, 1]


def test_column_order_is_stable_whatever_the_training_order():
    shuffled = TRAIN_JOBS.sample(frac=1, random_state=3)
    assert CategoryEncoder('job').fit(shuffled).columns == CategoryEncoder('job').fit(TRAIN_JOBS).columns


def test_encoder_state_round_trips_through_json():
    trained = CategoryEncoder('job').fit(TRAIN_JOBS)
    restored = CategoryEncoder.from_dict(json.loads(json.dumps(trained.to_dict())))
    later = pd.Series(['technician', 'astronaut'])
    pd.testing.assert_frame_equal(trained.transform(later), restored.transform(later))


def test_encoder_must_be_fitted():
    with pytest.raises(ValueError, match='must be fitted'):
        CategoryEncoder('job').transform(TRAIN_JOBS)


# ---- Capper --------------------------------------------------------------------------

def test_cap_is_learned_from_training_data_only():
    train = pd.Series(range(1, 101))            # 1..100, 50th percentile = 50.5
    capper = Capper(quantile=0.5).fit(train)
    later = pd.Series([10, 60, 1000])
    assert capper.cap == pytest.approx(50.5)
    assert capper.transform(later).tolist() == [10, 50.5, 50.5]


def test_capper_state_round_trips_and_is_json_safe():
    trained = Capper(0.99).fit(pd.Series(np.arange(1000, dtype='int64')))
    state = json.loads(json.dumps(trained.to_dict()))
    restored = Capper.from_dict(state)
    later = pd.Series([5, 5000])
    pd.testing.assert_series_equal(trained.transform(later), restored.transform(later))


@pytest.mark.parametrize('quantile', [0, 99, -0.5])
def test_capper_rejects_invalid_quantile(quantile):
    with pytest.raises(ValueError, match='quantile'):
        Capper(quantile)


def test_capper_rejects_all_missing_training_data():
    with pytest.raises(ValueError, match='non-missing'):
        Capper().fit(pd.Series([np.nan, np.nan]))


def test_capper_must_be_fitted():
    with pytest.raises(ValueError, match='must be fitted'):
        Capper().transform(pd.Series([1]))


# ---- stateless helpers -------------------------------------------------------------

def test_log1p_values():
    assert log1p(pd.Series([0, np.e - 1])).tolist() == pytest.approx([0.0, 1.0])


def test_log1p_rejects_negative_values():
    with pytest.raises(ValueError, match='non-negative'):
        log1p(pd.Series([1, -2], name='campaign'))


def test_sentinel_flag_marks_real_values():
    pdays = pd.Series([999, 3, 999, 0], index=[5, 6, 7, 8])
    flag = sentinel_flag(pdays, 999)
    assert flag.tolist() == [0, 1, 0, 1] and flag.index.tolist() == [5, 6, 7, 8]


def test_sentinel_flag_rejects_nan_sentinel():
    with pytest.raises(ValueError, match='isna'):
        sentinel_flag(pd.Series([1.0]), float('nan'))
