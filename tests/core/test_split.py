"""Tests for bankpulse_core.split."""

import pandas as pd
import pytest

from bankpulse_core.split import check_no_overlap, split_summary, time_split, window_split

FRACTIONS = {'train': 0.7, 'validation': 0.15, 'test': 0.15}


def make_frame(n: int) -> pd.DataFrame:
    """Small stand-in for the raw file: rows in time order, every 4th row a subscriber."""
    return pd.DataFrame({'row_id': range(n), 'y': ['yes' if i % 4 == 0 else 'no' for i in range(n)]})


def test_sizes_match_fractions():
    splits = time_split(make_frame(100), FRACTIONS)
    assert {name: len(part) for name, part in splits.items()} == {'train': 70, 'validation': 15, 'test': 15}


def test_last_split_takes_remainder():
    # 101 * 0.7 = 70.7 and 101 * 0.15 = 15.15 round down, so the extra row goes to test
    splits = time_split(make_frame(101), FRACTIONS)
    assert sum(len(part) for part in splits.values()) == 101
    assert len(splits['test']) == 16


def test_order_is_kept():
    splits = time_split(make_frame(100), FRACTIONS)
    assert splits['train']['row_id'].tolist() == list(range(70))
    assert splits['test']['row_id'].tolist() == list(range(85, 100))


def test_works_with_a_non_default_index():
    # Positional slicing must not depend on the index being 0..n-1
    frame = make_frame(20).set_index(pd.Index(range(1000, 1020)))
    splits = time_split(frame, {'train': 0.5, 'test': 0.5})
    assert splits['train'].index.tolist() == list(range(1000, 1010))


@pytest.mark.parametrize('fractions, message', [
    ({'train': 0.7, 'validation': 0.15, 'test': 0.05}, 'add up to 1'),
    ({'train': 1.1, 'test': -0.1}, 'between 0 and 1'),
    ({'train': 1.0}, 'at least two splits'),
])
def test_bad_fractions_raise(fractions, message):
    with pytest.raises(ValueError, match=message):
        time_split(make_frame(100), fractions)


def test_too_few_rows_raise():
    with pytest.raises(ValueError, match='too few'):
        time_split(make_frame(3), FRACTIONS)


def test_optional_production_split():
    splits = time_split(make_frame(100), {'train': 0.4, 'validation': 0.1, 'test': 0.1, 'production': 0.4})
    assert list(splits) == ['train', 'validation', 'test', 'production']
    assert [len(part) for part in splits.values()] == [40, 10, 10, 40]


def test_no_overlap_passes_for_time_split():
    check_no_overlap(time_split(make_frame(100), FRACTIONS))


def test_no_overlap_detects_shared_rows():
    frame = make_frame(10)
    with pytest.raises(ValueError, match='shares'):
        check_no_overlap({'train': frame.iloc[:6], 'test': frame.iloc[5:]})


def test_no_overlap_detects_wrong_order():
    frame = make_frame(10)
    with pytest.raises(ValueError, match='out of time order'):
        check_no_overlap({'train': frame.iloc[5:], 'test': frame.iloc[:5]})


def test_summary_positive_rate():
    summary = split_summary(time_split(make_frame(100), FRACTIONS), target='y')
    assert summary['train'] == {'rows': 70, 'positive_rate': round(18 / 70, 4)}
    assert summary['test']['rows'] == 15


def test_summary_with_encoded_labels():
    frame = pd.DataFrame({'y': [1, 0, 0, 1]})
    assert split_summary({'all': frame}, target='y', positive=1) == {'all': {'rows': 4, 'positive_rate': 0.5}}


# ---- window_split ---------------------------------------------------------------------------

def test_window_takes_newest_batches_and_drops_old_ones():
    frame = make_frame(10)
    splits = window_split(frame, [2, 2, 2, 2, 2], train_batches=2, validation_batches=1, test_batches=1)
    assert splits['train']['row_id'].tolist() == [2, 3, 4, 5]          # batch 1 (rows 0-1) aged out
    assert splits['validation']['row_id'].tolist() == [6, 7]
    assert splits['test']['row_id'].tolist() == [8, 9]
    check_no_overlap(splits)


def test_window_needs_enough_batches():
    with pytest.raises(ValueError, match='needs 4 batches, only 3'):
        window_split(make_frame(6), [2, 2, 2], train_batches=2, validation_batches=1, test_batches=1)


def test_window_sizes_must_match_the_data():
    with pytest.raises(ValueError, match='add up to 5 rows, but the data has 6'):
        window_split(make_frame(6), [2, 3], train_batches=1, validation_batches=1, test_batches=1)
