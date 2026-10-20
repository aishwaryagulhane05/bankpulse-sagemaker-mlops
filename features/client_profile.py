"""Client profile and contact timing features: who the customer is and when they were called.

    age v1                   age in years (raw)
    age_band v1              age in four bands (<30, 30-44, 45-59, 60+), one-hot
    categorical_profile v1   job, marital, education, default, housing, loan; one-hot
    contact_timing v1        contact, month, day_of_week; one-hot

Output columns are named "<feature>__<category>". Features that one-hot several raw
columns add the raw column too: "<feature>__<column>__<category>", e.g.
"categorical_profile__job__admin.", so two columns with the same value never clash.

Category lists are learned from TRAINING data only. A value never seen in training goes
to the column's "__other" column (in this dataset, month "sep" appears only after the
training period), while the dataset's own "unknown" value stays a normal category.
"""

from typing import Dict, Tuple

import pandas as pd

from bankpulse_core.encoding import CategoryEncoder
from bankpulse_core.feature_base import Feature, register

OWNER = 'bankpulse-ds'


@register
class Age(Feature):
    name = 'age'
    version = 'v1'
    inputs = ('age',)
    owner = OWNER
    description = 'Customer age in years (raw)'

    def _transform(self, df: pd.DataFrame) -> pd.Series:
        return df['age'].astype(int)


@register
class AgeBand(Feature):
    name = 'age_band'
    version = 'v1'
    inputs = ('age',)
    owner = OWNER
    description = 'Customer age in bands <30, 30-44, 45-59, 60+ (one-hot)'

    # Lower bound of each band, in order. A band includes its lower bound, so 30 is in
    # "30-44", 45 in "45-59" and 60 in "60+". The bands are fixed business rules, not
    # learned, so every band always has a column even if training has no one in it.
    BANDS: Tuple[Tuple[int, str], ...] = ((0, 'lt30'), (30, '30_44'), (45, '45_59'), (60, '60plus'))

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        age = df['age']
        lower = [bound for bound, _ in self.BANDS]
        upper = lower[1:] + [float('inf')]
        # Step 1: one 0/1 column per band: lower bound <= age < next band's lower bound.
        columns = {
            f'{self.name}__{label}': ((age >= low) & (age < high)).astype(int)
            for (low, label), high in zip(self.BANDS, upper)
        }
        # Step 2: keep the input index so rows line up with the other features.
        return pd.DataFrame(columns, index=df.index)


class _OneHotColumns(Feature):
    """Shared logic for one-hot features over several raw columns (not registered itself).

    Keeps one CategoryEncoder per input column, fitted on training data.
    """

    def __init__(self) -> None:
        super().__init__()
        self._encoders: Dict[str, CategoryEncoder] = {}

    def _encoder_name(self, column: str) -> str:
        # Prefix "<feature>__<column>", so outputs read "<feature>__<column>__<category>".
        return f'{self.name}__{column}'

    def _fit(self, train: pd.DataFrame) -> None:
        # Step 1: learn each column's categories from TRAINING rows only.
        self._encoders = {
            column: CategoryEncoder(self._encoder_name(column)).fit(train[column])
            for column in self.inputs
        }

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Step 2: encode each column with what training learned (unseen -> "__other"),
        # then join them side by side in the declared input order.
        return pd.concat([self._encoders[column].transform(df[column]) for column in self.inputs], axis=1)

    def _state(self) -> Dict:
        # Step 3: save every column's category list so the endpoint builds the same columns.
        return {column: encoder.to_dict() for column, encoder in self._encoders.items()}

    def _load_state(self, state: Dict) -> None:
        self._encoders = {column: CategoryEncoder.from_dict(saved) for column, saved in state.items()}


@register
class CategoricalProfile(_OneHotColumns):
    name = 'categorical_profile'
    version = 'v1'
    inputs = ('job', 'marital', 'education', 'default', 'housing', 'loan')
    owner = OWNER
    description = 'Job, marital status, education, credit default, housing and personal loan (one-hot)'


@register
class ContactTiming(_OneHotColumns):
    name = 'contact_timing'
    version = 'v1'
    inputs = ('contact', 'month', 'day_of_week')
    owner = OWNER
    description = 'Contact channel, month and day of week of the last contact (one-hot)'
