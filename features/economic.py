"""Economic indicator features: the economy at the time of the call, not the customer.

    euribor_rate v1           3-month Euribor rate (euribor3m)
    employment_level v1       number of employees in the economy (nr.employed)
    economic_indicators v1    employment variation rate, consumer price index and
                              consumer confidence index (one column each)

DRIFT WARNING: these values describe the period of the call, so they change over time
and are the same for every customer called in the same period. A model trained on one
period sees different values later; in this dataset, with a time split, the training
mean of euribor3m is about 4.7 but the test mean is about 1.0. Watch these columns
first in drift monitoring, and expect them to be the main cause of performance changes.

All three are stateless: values are passed through as floats with no fitting.
"""

import pandas as pd

from bankpulse_core.feature_base import Feature, register

OWNER = 'bankpulse-ds'


@register
class EuriborRate(Feature):
    """3-month Euribor rate. Time-dependent: monitor for drift."""

    name = 'euribor_rate'
    version = 'v1'
    inputs = ('euribor3m',)
    owner = OWNER
    description = '3-month Euribor interest rate at the time of the call (time-dependent; monitor for drift)'

    def _transform(self, df: pd.DataFrame) -> pd.Series:
        # Pass through as float so every split has the same dtype.
        return df['euribor3m'].astype(float)


@register
class EmploymentLevel(Feature):
    """Number of employees in the economy (quarterly). Time-dependent: monitor for drift."""

    name = 'employment_level'
    version = 'v1'
    inputs = ('nr.employed',)
    owner = OWNER
    description = 'Number of employees in the economy, quarterly (time-dependent; monitor for drift)'

    def _transform(self, df: pd.DataFrame) -> pd.Series:
        return df['nr.employed'].astype(float)


@register
class EconomicIndicators(Feature):
    """Three macro indicators as separate columns. Time-dependent: monitor for drift."""

    name = 'economic_indicators'
    version = 'v1'
    inputs = ('emp.var.rate', 'cons.price.idx', 'cons.conf.idx')
    owner = OWNER
    description = ('Employment variation rate, consumer price index and consumer confidence index '
                   '(time-dependent; monitor for drift)')

    # Raw column -> output column. The raw names contain dots, which break some tools
    # (SQL, Feature Store), so outputs use "<feature>__<snake_case>" like the encoders do.
    OUTPUTS = {
        'emp.var.rate': 'economic_indicators__emp_var_rate',
        'cons.price.idx': 'economic_indicators__cons_price_idx',
        'cons.conf.idx': 'economic_indicators__cons_conf_idx',
    }

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Step 1: rename to the safe output names; Step 2: fix the dtype and column order.
        return df.rename(columns=self.OUTPUTS)[list(self.OUTPUTS.values())].astype(float)
