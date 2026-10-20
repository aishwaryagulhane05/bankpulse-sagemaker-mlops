"""Campaign features: how hard the bank pushed this customer in the current campaign.

    campaign_intensity v1   number of contacts in this campaign (raw count)
    capped_campaign v1      the same count, clipped at the training data's 99th percentile
    capped_campaign v2      log(1 + capped count): the v1 cap, then a log to shrink the tail

The raw count has a long tail (most customers get 1-3 calls, a few get 40+). v2 is an
upgrade of v1 that lives next to it: config/data_prep.yaml picks the version, so a model
trained on v1 keeps getting v1 values while v2 is evaluated.
"""

from typing import Dict

import pandas as pd

from bankpulse_core.encoding import Capper, log1p
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


@register
class CappedCampaignV2(CappedCampaign):
    """v2: the same learned cap as v1, followed by log1p.

    The log spreads out the common 1-3 contacts and squeezes the rare high counts, which
    suits linear models; tree models such as XGBoost mostly ignore monotonic transforms,
    so v2 is expected to matter more for the logistic regression baseline.
    """

    version = 'v2'
    description = f'log(1 + contacts in this campaign), after clipping at the training {CAP_QUANTILE:.0%} quantile'

    def _transform(self, df: pd.DataFrame) -> pd.Series:
        # Step 2 (v2): clip with the learned cap exactly as v1 does, then take log1p.
        return log1p(super()._transform(df))
