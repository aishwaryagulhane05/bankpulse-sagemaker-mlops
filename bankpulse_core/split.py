"""Splitting: time-based train / validation / test / production splits.

The bank marketing file is ordered by contact date (May 2008 -> Nov 2010), and the
subscription rate drifts over that period. A random split would let the model learn
from "future" rows and report scores it can't achieve in production, so every split
here keeps the original row order: oldest rows train the model, newest rows test it.
"""

from typing import Dict, Mapping, Sequence, Union

import pandas as pd

# Floating-point fractions such as 0.7 + 0.15 + 0.15 don't add up to exactly 1.0,
# so the "sums to 1" check allows for this tiny rounding error.
_TOLERANCE = 1e-9


def _check_fractions(fractions: Mapping[str, float]) -> None:
    """Raise ValueError unless every fraction is in (0, 1) and they add up to 1."""
    # Step 1: there must be at least two splits, otherwise nothing is held out.
    if len(fractions) < 2:
        raise ValueError(f"Need at least two splits, got {list(fractions)}")

    # Step 2: each fraction must be a share of the data, strictly between 0 and 1.
    bad = {name: value for name, value in fractions.items() if not 0 < value < 1}
    if bad:
        raise ValueError(f"Each fraction must be between 0 and 1 (exclusive), got {bad}")

    # Step 3: together the fractions must cover the whole dataset, no more, no less.
    total = sum(fractions.values())
    if abs(total - 1) > _TOLERANCE:
        raise ValueError(f"Fractions must add up to 1, got {total:.6f} from {dict(fractions)}")


def time_split(df: pd.DataFrame, fractions: Mapping[str, float]) -> Dict[str, pd.DataFrame]:
    """
    Split rows in their existing (time) order, without shuffling.

    Args:
        df: Data already sorted oldest -> newest.
        fractions: Split name -> share of rows, in the order the splits should be taken,
            e.g. {"train": 0.7, "validation": 0.15, "test": 0.15}. An optional fourth
            entry such as "production" is held back for monitoring later.

    Returns:
        Split name -> DataFrame. Each keeps its original index, so rows can be traced back
        to the raw file and checked for overlap.

    Raises:
        ValueError: if the fractions are invalid or the data is too small to fill every split.
    """
    # Step 1: validate the configuration before touching the data.
    _check_fractions(fractions)

    n_rows = len(df)
    names = list(fractions)

    # Step 2: turn each fraction into a whole number of rows. int() rounds down, so the
    # counts can fall short of n_rows; the last split takes whatever is left.
    counts = [int(n_rows * fractions[name]) for name in names[:-1]]
    counts.append(n_rows - sum(counts))

    # Step 3: refuse to return an empty split, e.g. when a tiny file is split three ways.
    empty = [name for name, count in zip(names, counts) if count == 0]
    if empty:
        raise ValueError(f"{n_rows} rows are too few: split(s) {empty} would be empty")

    # Step 4: slice consecutive blocks by position. iloc works on row positions, so this
    # is correct even if the index isn't 0..n-1.
    splits, start = {}, 0
    for name, count in zip(names, counts):
        splits[name] = df.iloc[start:start + count]
        start += count
    return splits


def check_no_overlap(splits: Mapping[str, pd.DataFrame]) -> None:
    """
    Raise ValueError if any row appears in two splits or the splits are out of time order.

    Splits are expected in the order they were created (train before validation before test).
    """
    seen = set()
    previous_name, previous_last = None, None

    for name, part in splits.items():
        index = set(part.index)

        # Step 1: a row shared with an earlier split would leak data between them.
        shared = seen & index
        if shared:
            raise ValueError(f"Split '{name}' shares {len(shared)} row(s) with an earlier split, "
                             f"e.g. index {sorted(shared)[:5]}")
        seen |= index

        # Step 2: every split must start after the previous one ends (time order).
        if len(part) and previous_last is not None and part.index.min() <= previous_last:
            raise ValueError(f"Split '{name}' starts at index {part.index.min()}, before "
                             f"'{previous_name}' ends at {previous_last}: splits are out of time order")
        if len(part):
            previous_name, previous_last = name, part.index.max()


def split_summary(splits: Mapping[str, pd.DataFrame], target: str,
                  positive: Union[str, int] = 'yes') -> Dict[str, Dict[str, float]]:
    """
    Rows and positive rate of each split, for the data-quality report.

    Args:
        splits: Split name -> DataFrame.
        target: Name of the label column.
        positive: Value that counts as a positive label; "yes" for the raw file, 1 once encoded.

    Returns:
        Split name -> {"rows": int, "positive_rate": float}.
    """
    summary = {}
    for name, part in splits.items():
        # Step 1: share of rows whose label equals the positive value. An empty split has
        # no rate, so report 0.0 instead of dividing by zero.
        rate = float((part[target] == positive).mean()) if len(part) else 0.0

        # Step 2: round for readability; the full precision isn't meaningful in a report.
        summary[name] = {'rows': len(part), 'positive_rate': round(rate, 4)}
    return summary


def window_split(df: pd.DataFrame, batch_sizes: Sequence[int], train_batches: int, validation_batches: int,
                 test_batches: int) -> Dict[str, pd.DataFrame]:
    """
    Rolling-window split over consecutive batches (e.g. months), newest last.

    The newest test_batches go to test, the validation_batches before them to validation,
    and the train_batches before those to train. Older batches are left out, so the model
    learns from recent conditions instead of the whole history.

    Args:
        df: Rows of every batch, concatenated in arrival order.
        batch_sizes: Row count of each batch, in the same order (sum must equal len(df)).
    """
    # Step 1: the batch sizes must describe df exactly, or the boundaries would be wrong.
    if sum(batch_sizes) != len(df):
        raise ValueError(f'batch sizes add up to {sum(batch_sizes)} rows, but the data has {len(df)}')
    needed = train_batches + validation_batches + test_batches
    if len(batch_sizes) < needed:
        raise ValueError(f'the window needs {needed} batches, only {len(batch_sizes)} are available')

    # Step 2: row offsets of the window boundaries, counted back from the newest batch.
    ends = [0]
    for size in batch_sizes:
        ends.append(ends[-1] + size)
    test_start = ends[len(batch_sizes) - test_batches]
    validation_start = ends[len(batch_sizes) - test_batches - validation_batches]
    train_start = ends[len(batch_sizes) - needed]

    # Step 3: consecutive positional slices, like time_split.
    return {'train': df.iloc[train_start:validation_start],
            'validation': df.iloc[validation_start:test_start],
            'test': df.iloc[test_start:]}
