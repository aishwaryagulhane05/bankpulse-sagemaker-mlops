"""Encoding helpers: one-hot, unseen-category handling, capping, log transforms.

These are generic building blocks for features/; they know nothing about the bank data.
Helpers that learn something (CategoryEncoder, Capper) follow one pattern:

    fit(train_series)        learn from TRAINING data only
    transform(series)        apply what was learned to any split or live data
    to_dict() / from_dict()  save and restore what was learned, as JSON-safe values

so the real-time endpoint encodes a customer exactly as the training data was encoded.
"""

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Column suffix for values never seen in training. It is deliberately different from the
# dataset's own "unknown" category: "unknown" is a known, meaningful value in training,
# while OTHER means "this value didn't exist when the model was trained".
OTHER = 'other'


def _require_fitted(helper: object, learned: object) -> None:
    if learned is None:
        raise ValueError(f'{type(helper).__name__} must be fitted (or loaded with from_dict) before transform()')


class CategoryEncoder:
    """One-hot encode a categorical column, with a fixed column for unseen values."""

    def __init__(self, name: str) -> None:
        # Prefix for output columns, e.g. "job" -> "job__admin.", "job__other"
        self.name = name
        self.categories: Optional[List[str]] = None

    def fit(self, train: pd.Series) -> 'CategoryEncoder':
        """Learn the categories present in training data."""
        # Step 1: compare as strings, so 1 and "1" can't become two different categories.
        values = train.astype(str)
        # Step 2: sort, so the column order is the same on every run and machine.
        self.categories = sorted(values.unique())
        return self

    @property
    def columns(self) -> List[str]:
        """Output column names, in a fixed order: learned categories, then OTHER."""
        _require_fitted(self, self.categories)
        return [f'{self.name}__{category}' for category in self.categories] + [f'{self.name}__{OTHER}']

    def transform(self, series: pd.Series) -> pd.DataFrame:
        """One 0/1 column per learned category plus the OTHER column; same index as series."""
        _require_fitted(self, self.categories)
        values = series.astype(str)
        known = values.isin(self.categories)

        # Step 1: one column per training category (1 where the row has that value).
        encoded = {f'{self.name}__{category}': (values == category).astype(int) for category in self.categories}
        # Step 2: rows with a value never seen in training go to OTHER, so a new job title
        # at inference time is scored instead of crashing the endpoint.
        encoded[f'{self.name}__{OTHER}'] = (~known).astype(int)

        # Step 3: build the frame in the fixed column order and keep the input index.
        return pd.DataFrame(encoded, index=series.index)[self.columns]

    def to_dict(self) -> Dict:
        _require_fitted(self, self.categories)
        return {'name': self.name, 'categories': list(self.categories)}

    @classmethod
    def from_dict(cls, state: Dict) -> 'CategoryEncoder':
        encoder = cls(state['name'])
        encoder.categories = [str(category) for category in state['categories']]
        return encoder


class Capper:
    """Clip a numeric column at an upper limit learned from training data (a quantile)."""

    def __init__(self, quantile: float = 0.99) -> None:
        # Step 0: a quantile outside (0, 1] has no meaning; catch typos such as 99 instead of 0.99.
        if not 0 < quantile <= 1:
            raise ValueError(f'quantile must be in (0, 1], got {quantile}')
        self.quantile = quantile
        self.cap: Optional[float] = None

    def fit(self, train: pd.Series) -> 'Capper':
        """Learn the cap as the training data's quantile (e.g. the 99th percentile)."""
        values = pd.to_numeric(train)
        # Step 1: an all-missing column has no quantile; fail here with a clear message.
        if values.dropna().empty:
            raise ValueError('Capper.fit() needs at least one non-missing value')
        # Step 2: store a plain float so the state is JSON-safe.
        self.cap = float(values.quantile(self.quantile))
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        """Values above the learned cap are replaced by the cap; others are unchanged."""
        _require_fitted(self, self.cap)
        return pd.to_numeric(series).clip(upper=self.cap)

    def to_dict(self) -> Dict:
        _require_fitted(self, self.cap)
        return {'quantile': self.quantile, 'cap': self.cap}

    @classmethod
    def from_dict(cls, state: Dict) -> 'Capper':
        capper = cls(state['quantile'])
        capper.cap = float(state['cap'])
        return capper


def log1p(series: pd.Series) -> pd.Series:
    """log(1 + x) for skewed non-negative counts (stateless, so no fitting is needed)."""
    values = pd.to_numeric(series)
    # log1p is undefined below -1 and misleading for negatives, so reject them explicitly.
    if (values < 0).any():
        raise ValueError(f'log1p expects non-negative values; {series.name} has a minimum of {values.min()}')
    return np.log1p(values)


def sentinel_flag(series: pd.Series, sentinel: object) -> pd.Series:
    """1 where the value is real, 0 where it equals the sentinel (e.g. pdays == 999 = never contacted)."""
    # A sentinel such as 999 isn't a quantity, so a model would otherwise read it as
    # "a very long time ago". The flag lets the model treat "no value" separately.
    if isinstance(sentinel, float) and math.isnan(sentinel):
        raise ValueError('Use isna() for missing values; sentinel_flag compares against a real value')
    return (series != sentinel).astype(int)
