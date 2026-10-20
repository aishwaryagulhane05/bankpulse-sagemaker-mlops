"""Feature interface and registry: name, version, inputs, owner, fit(), transform().

Every feature in features/ subclasses Feature and is added to the registry with
@register. The pipeline never imports feature classes directly: it looks them up by
(name, version) from config, so a new feature or a v2 upgrade needs no pipeline change.

Feature classes implement two hooks:
    _transform(df)  required: compute the feature from raw columns
    _fit(train)     optional: learn values (caps, category lists) from TRAINING data

The public fit() / transform() wrap these hooks with the checks every feature needs
(inputs present, fitted before use, output aligned with the input rows), so no feature
can skip them.
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Mapping, Tuple, Type, Union

import pandas as pd

# Versions look like "v1", "v2", ...: simple to compare and to write in config.
_VERSION_PATTERN = re.compile(r'^v\d+$')


class FeatureError(Exception):
    """A feature was defined, configured or used incorrectly."""


class Feature(ABC):
    """
    Base class for every feature.

    Subclasses set the class attributes below and implement _transform(). Features that
    learn from data also implement _fit(), _state() and _load_state().
    """

    # Identity and documentation; checked when the class is registered.
    name: str = ''
    version: str = ''
    inputs: Tuple[str, ...] = ()
    owner: str = ''
    description: str = ''

    def __init__(self) -> None:
        # A new feature object hasn't learned anything yet.
        self._fitted = False

    @property
    def fitted(self) -> bool:
        return self._fitted

    # ---- public API used by the pipeline -------------------------------------------

    def fit(self, train: pd.DataFrame) -> 'Feature':
        """Learn from training rows only; returns self so calls can be chained."""
        # Step 1: fail early with a clear message if a required column is missing.
        self._check_columns(train)
        # Step 2: let the subclass learn whatever it needs (no-op for stateless features).
        self._fit(train)
        # Step 3: only now is transform() allowed.
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the feature for df using what fit() learned; returns one or more columns."""
        # Step 1: using an unfitted feature would silently skip learned values (e.g. caps).
        if not self._fitted:
            raise FeatureError(f'{self.key()} must be fitted (or have its state loaded) before transform()')
        # Step 2: the same input check as fit(), for validation, test and live data.
        self._check_columns(df)
        # Step 3: let the subclass compute the values.
        result = self._transform(df[list(self.inputs)])
        # Step 4: one-column features may return a Series; name it after the feature.
        if isinstance(result, pd.Series):
            result = result.to_frame(self.name)
        # Step 5: every output row must line up with an input row, or columns from
        # different features would be joined to the wrong customers.
        if not isinstance(result, pd.DataFrame) or not result.index.equals(df.index):
            raise FeatureError(f'{self.key()} must return a Series or DataFrame with the same index as its input')
        return result

    def state(self) -> Dict:
        """What fit() learned, as a JSON-safe dict (saved in feature_schema.json)."""
        if not self._fitted:
            raise FeatureError(f'{self.key()} has no state before fit()')
        state = self._state()
        # The endpoint reloads this from JSON, so catch NumPy types, sets, etc. here,
        # where the cause is obvious, rather than when the schema file is written.
        try:
            json.dumps(state)
        except TypeError as exc:
            raise FeatureError(f'{self.key()} state is not JSON-safe: {exc}') from exc
        return state

    def load_state(self, state: Mapping) -> 'Feature':
        """Restore learned values (e.g. at inference time) without refitting."""
        self._load_state(dict(state))
        self._fitted = True
        return self

    def key(self) -> str:
        """Readable identifier used in messages and logs, e.g. 'campaign_intensity v2'."""
        return f'{self.name} {self.version}'

    # ---- hooks implemented by subclasses ---------------------------------------------

    def _fit(self, train: pd.DataFrame) -> None:
        """Learn from training data; default: nothing to learn."""

    @abstractmethod
    def _transform(self, df: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        """Compute the feature; df contains only the declared input columns."""

    def _state(self) -> Dict:
        """Learned values; default: none."""
        return {}

    def _load_state(self, state: Dict) -> None:
        """Restore learned values; default: none to restore."""

    # ---- internal helpers ------------------------------------------------------------

    def _check_columns(self, df: pd.DataFrame) -> None:
        missing = [column for column in self.inputs if column not in df.columns]
        if missing:
            raise FeatureError(f'{self.key()} needs column(s) {missing}, which are missing from the data')


# (name, version) -> Feature class. Keyed by version so v1 and v2 can live side by side:
# models trained with v1 must keep receiving v1 values after v2 is released.
_REGISTRY: Dict[Tuple[str, str], Type[Feature]] = {}


def _check_definition(cls: Type[Feature]) -> None:
    """Raise FeatureError if a feature class is missing required metadata."""
    if not (isinstance(cls, type) and issubclass(cls, Feature)):
        raise FeatureError(f'{cls!r} is not a Feature subclass')
    if not cls.name or not cls.name.isidentifier():
        raise FeatureError(f'{cls.__name__}.name must be a non-empty snake_case identifier, got {cls.name!r}')
    if not _VERSION_PATTERN.match(cls.version or ''):
        raise FeatureError(f'{cls.__name__}.version must look like "v1", got {cls.version!r}')
    if not cls.inputs or not all(isinstance(column, str) for column in cls.inputs):
        raise FeatureError(f'{cls.__name__}.inputs must be a non-empty list of column names')
    if not cls.owner or not cls.description:
        raise FeatureError(f'{cls.__name__} needs an owner and a description')


def register(cls: Type[Feature]) -> Type[Feature]:
    """Class decorator: validate the feature and add it to the registry."""
    # Step 1: reject incomplete definitions when the module is imported, not mid-pipeline.
    _check_definition(cls)
    # Step 2: two classes with the same name and version would make results ambiguous.
    key = (cls.name, cls.version)
    if key in _REGISTRY:
        raise FeatureError(f'Feature {cls.name} {cls.version} is already registered by '
                           f'{_REGISTRY[key].__module__}.{_REGISTRY[key].__name__}')
    _REGISTRY[key] = cls
    # Step 3: decorators must return the class, or the module name would be bound to None.
    return cls


def get_feature(name: str, version: str) -> Type[Feature]:
    """The registered class for (name, version); raises FeatureError if unknown."""
    try:
        return _REGISTRY[(name, version)]
    except KeyError:
        known = sorted(v for n, v in _REGISTRY if n == name)
        hint = f'registered versions: {known}' if known else 'no feature with that name is registered'
        raise FeatureError(f'Unknown feature {name} {version} ({hint})') from None


def resolve(enabled: Mapping[str, str]) -> List[Feature]:
    """
    New, unfitted feature objects for the config's enabled features, in config order.

    Args:
        enabled: Feature name -> version, e.g. {"previously_contacted": "v1"}.
    """
    return [get_feature(name, version)() for name, version in enabled.items()]


def check_inputs(feature: Feature, allowed: Iterable[str], forbidden: Iterable[str] = ()) -> None:
    """
    Raise FeatureError if a feature reads a forbidden column or one outside the contract.

    Args:
        feature: Feature object (or class) to check.
        allowed: Columns the data contract allows.
        forbidden: Leakage columns that must never be used, e.g. ["duration"].
    """
    # Step 1: leakage first, with its own message, because it's the most serious error.
    leaking = sorted(set(feature.inputs) & set(forbidden))
    if leaking:
        raise FeatureError(f'{feature.name} {feature.version} reads forbidden column(s) {leaking}')
    # Step 2: every other input must be a column the contract knows about.
    unknown = sorted(set(feature.inputs) - set(allowed))
    if unknown:
        raise FeatureError(f'{feature.name} {feature.version} reads column(s) {unknown} not in the data contract')
