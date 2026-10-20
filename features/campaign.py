"""Campaign features: how hard the bank pushed this customer in the current campaign.

    campaign_intensity v1   number of contacts in this campaign (raw count)
    capped_campaign v1      the same count, clipped at the training data's 99th percentile

The raw count has a long tail (most customers get 1-3 calls, a few get 40+). v1 keeps
the raw count and a capped copy; a log1p version is planned as a v2 upgrade, so the two
versions can be compared without changing v1.
"""

from typing import Dict

import pandas as pd

from bankpulse_core.encoding import Capper
from bankpulse_core.feature_base import Feature, register

OWNER = 'bankpulse-ds'

# Clip at the 99th percentile: keeps 99% of training values unchanged and stops a few
# extreme customers from dominating linear models.
CAP_QUANTILE = 0.99


@register
class CampaignIntensity(Feature):
    name = 'campaign_intensity'
    version = 'v1'
    inputs = ('campaign',)
    owner = OWNER
    description = 'Number of contacts with the customer during this campaign (raw count)'

    def _transform(self, df: pd.DataFrame) -> pd.Series:
        # Stateless: the raw count, as an int so every split has the same dtype.
        return df['campaign'].astype(int)


@register
class CappedCampaign(Feature):
    name = 'capped_campaign'
    version = 'v1'
    inputs = ('campaign',)
    owner = OWNER
    description = f'Contacts in this campaign, clipped at the training {CAP_QUANTILE:.0%} quantile'

    def __init__(self) -> None:
        super().__init__()
        self._capper = Capper(CAP_QUANTILE)

    def _fit(self, train: pd.DataFrame) -> None:
        # Step 1: learn the cap from TRAINING rows only. Validation, test and live data
        # reuse this value, so they never influence it.
        self._capper.fit(train['campaign'])

    def _transform(self, df: pd.DataFrame) -> pd.Series:
        # Step 2: clip anything above the learned cap; values below it are unchanged.
        return self._capper.transform(df['campaign'])

    def _state(self) -> Dict:
        # Step 3: save the cap so the endpoint clips with exactly the training value.
        return self._capper.to_dict()

    def _load_state(self, state: Dict) -> None:
        self._capper = Capper.from_dict(state)
