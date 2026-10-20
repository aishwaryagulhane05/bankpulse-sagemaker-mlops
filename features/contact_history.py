"""Contact history features: what happened in earlier campaigns with this customer.

    previously_contacted v1   1 if the customer was contacted in an earlier campaign
    prior_success v1          1 if the earlier campaign ended in a subscription
    previous_contacts v1      number of contacts before this campaign (raw count)

All three are stateless: they learn nothing from training data, so fit() only checks
the input columns and state() is empty.
"""

import pandas as pd

from bankpulse_core.encoding import sentinel_flag
from bankpulse_core.feature_base import Feature, register

OWNER = 'bankpulse-ds'

# In the raw file pdays == 999 means "never contacted before", not "999 days ago".
NEVER_CONTACTED = 999


@register
class PreviouslyContacted(Feature):
    name = 'previously_contacted'
    version = 'v1'
    inputs = ('pdays',)
    owner = OWNER
    description = '1 if the customer was contacted in an earlier campaign (pdays is not the 999 sentinel)'

    def _transform(self, df: pd.DataFrame) -> pd.Series:
        # The sentinel isn't a quantity, so turn it into a yes/no flag instead of a number.
        return sentinel_flag(df['pdays'], NEVER_CONTACTED)


@register
class PriorSuccess(Feature):
    name = 'prior_success'
    version = 'v1'
    inputs = ('poutcome',)
    owner = OWNER
    description = '1 if the previous campaign ended in a subscription (poutcome == "success")'

    def _transform(self, df: pd.DataFrame) -> pd.Series:
        # "failure" and "nonexistent" both mean no earlier subscription, so both map to 0.
        return (df['poutcome'] == 'success').astype(int)


@register
class PreviousContacts(Feature):
    name = 'previous_contacts'
    version = 'v1'
    inputs = ('previous',)
    owner = OWNER
    description = 'Number of contacts with the customer before this campaign (raw count)'

    def _transform(self, df: pd.DataFrame) -> pd.Series:
        # Kept raw in v1: the counts are small (0-7 in the data), so no capping is needed.
        return df['previous'].astype(int)
