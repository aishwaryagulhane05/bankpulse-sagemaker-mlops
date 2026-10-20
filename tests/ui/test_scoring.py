"""Tests for ui.scoring with a fake endpoint (no AWS access)."""

import io
import sys
import types

import pandas as pd
import pytest

from bankpulse_core import feature_base
from bankpulse_core.feature_base import Feature, register
from bankpulse_core.pipeline import build_schema
from ui import scoring


class Doubled(Feature):
    name, version, inputs, owner, description = 'doubled', 'v1', ('campaign',), 'tests', 'Twice campaign'

    def _transform(self, df):
        return df['campaign'] * 2


@pytest.fixture
def schema(monkeypatch):
    monkeypatch.setattr(feature_base, '_REGISTRY', {})
    monkeypatch.setitem(sys.modules, 'fake_ui_features', types.ModuleType('fake_ui_features'))
    register(Doubled)
    fitted = Doubled().fit(pd.DataFrame({'campaign': [1]}))
    return build_schema([fitted], ['doubled'], 'y', 'yes', 'fake_ui_features')


class FakeRuntime:
    """Returns 0.1 * the first input value per row, comma-separated like some XGBoost versions."""

    def __init__(self):
        self.calls = []

    def invoke_endpoint(self, EndpointName, ContentType, Accept, Body):
        self.calls.append(Body)
        values = [round(float(line.split(',')[0]) / 10, 4) for line in Body.strip().splitlines()]
        return {'Body': io.BytesIO(','.join(str(v) for v in values).encode())}


def test_score_encodes_records_and_batches_requests(schema, monkeypatch):
    monkeypatch.setattr(scoring, 'BATCH', 2)
    runtime = FakeRuntime()
    scores = scoring.score(runtime, 'ep', [{'campaign': 1}, {'campaign': 2}, {'campaign': 4}], schema)
    assert scores == [0.2, 0.4, 0.8]
    assert runtime.calls == ['2\n4\n', '8\n']            # encoded by the schema's feature, 2 per request


def test_score_rejects_wrong_number_of_scores(schema):
    class Short(FakeRuntime):
        def invoke_endpoint(self, **kwargs):
            return {'Body': io.BytesIO(b'0.5')}

    with pytest.raises(ValueError, match='1 scores for 2 customers'):
        scoring.score(Short(), 'ep', [{'campaign': 1}, {'campaign': 2}], schema)


def test_parse_scores_accepts_newlines_and_commas():
    assert scoring.parse_scores('0.1\n0.2\n') == [0.1, 0.2]
    assert scoring.parse_scores('0.1,0.2') == [0.1, 0.2]


def test_rank_flags_and_orders_customers():
    ranked = scoring.rank(pd.DataFrame({'id': ['a', 'b', 'c']}), [0.05, 0.9, 0.3], threshold=0.25)
    assert ranked['id'].tolist() == ['b', 'c', 'a']
    assert ranked['rank'].tolist() == [1, 2, 3] and ranked['call_first'].tolist() == [True, True, False]


def test_form_options_and_required_inputs_come_from_contract_and_schema(schema):
    from bankpulse_core.contract import load_contract
    from pathlib import Path

    contract = load_contract(Path(__file__).resolve().parents[2] / 'config' / 'contract.yaml')
    options = scoring.form_options(contract)
    assert 'duration' not in options and 'y' not in options
    assert options['poutcome'] == ['failure', 'nonexistent', 'success']
    assert scoring.required_inputs(schema) == ['campaign']
