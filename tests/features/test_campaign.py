"""Tests for features.campaign."""

import json

import numpy as np
import pandas as pd
import pytest

from bankpulse_core.feature_base import FeatureError, check_inputs, get_feature, resolve
from features.campaign import CampaignIntensity, CappedCampaign, CappedCampaignV2

# 100 training rows with counts 1..100, so the 99th percentile is 99.01.
TRAIN = pd.DataFrame({'campaign': range(1, 101)})
# Later rows (another split, or live traffic) with values far above anything in training.
LATER = pd.DataFrame({'campaign': [1, 50, 500]}, index=[7, 8, 9])


def test_campaign_intensity_is_the_raw_count():
    result = CampaignIntensity().fit(TRAIN).transform(LATER)
    assert result['campaign_intensity'].tolist() == [1, 50, 500]
    assert result.index.equals(LATER.index)


def test_cap_is_learned_on_training_data_only():
    feature = CappedCampaign().fit(TRAIN)
    assert feature.state()['cap'] == pytest.approx(99.01)
    # Transforming later data must not move the cap.
    feature.transform(LATER)
    assert feature.state()['cap'] == pytest.approx(99.01)


def test_values_above_cap_are_clipped_and_others_unchanged():
    result = CappedCampaign().fit(TRAIN).transform(LATER)
    assert result['capped_campaign'].tolist() == pytest.approx([1, 50, 99.01])
    assert result.index.equals(LATER.index)


def test_capped_state_round_trips_through_json():
    trained = CappedCampaign().fit(TRAIN)
    restored = CappedCampaign().load_state(json.loads(json.dumps(trained.state())))
    pd.testing.assert_frame_equal(trained.transform(LATER), restored.transform(LATER))


def test_capped_campaign_must_be_fitted():
    with pytest.raises(FeatureError, match='must be fitted'):
        CappedCampaign().transform(LATER)


@pytest.mark.parametrize('name, cls', [
    ('campaign_intensity', CampaignIntensity),
    ('capped_campaign', CappedCampaign),
])
def test_registered_and_inputs_allowed(name, cls):
    assert get_feature(name, 'v1') is cls
    check_inputs(cls, allowed=['campaign', 'pdays'], forbidden=['duration'])


# ---- capped_campaign v2 (feature upgrade) -------------------------------------------------

def test_v2_is_log1p_of_the_v1_capped_value():
    v1 = CappedCampaign().fit(TRAIN).transform(LATER)['capped_campaign']
    v2 = CappedCampaignV2().fit(TRAIN).transform(LATER)['capped_campaign']
    assert v2.tolist() == pytest.approx(np.log1p(v1).tolist())
    assert v2.index.equals(LATER.index)


def test_v1_and_v2_are_registered_side_by_side():
    assert get_feature('capped_campaign', 'v1') is CappedCampaign
    assert get_feature('capped_campaign', 'v2') is CappedCampaignV2
    assert resolve({'capped_campaign': 'v2'})[0].version == 'v2'


def test_v2_learns_the_same_cap_and_state_round_trips():
    v2 = CappedCampaignV2().fit(TRAIN)
    assert v2.state() == CappedCampaign().fit(TRAIN).state()
    restored = CappedCampaignV2().load_state(json.loads(json.dumps(v2.state())))
    pd.testing.assert_frame_equal(v2.transform(LATER), restored.transform(LATER))
