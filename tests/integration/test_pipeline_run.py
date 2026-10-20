"""End-to-end tests: pipeline.run() on the real dataset with the project's config."""

import copy
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from bankpulse_core.config import load_config
from bankpulse_core.io import file_hash, read_json
from bankpulse_core.pipeline import run

REPO = Path(__file__).resolve().parents[2]
CONFIG = load_config(REPO / 'config' / 'data_prep.yaml')


def config_writing_to(out_dir):
    config = copy.deepcopy(CONFIG)
    config['output']['processed_dir'] = str(out_dir)
    return config


@pytest.fixture(scope='module')
def first_run(tmp_path_factory):
    return run(config_writing_to(tmp_path_factory.mktemp('run1')))


def test_same_raw_file_twice_gives_identical_output_hashes(first_run, tmp_path):
    second = run(config_writing_to(tmp_path))
    first_hashes = {name: file_hash(path) for name, path in first_run['outputs'].items()}
    second_hashes = {name: file_hash(path) for name, path in second['outputs'].items()}
    assert first_hashes == second_hashes


def test_outputs_have_expected_shapes(first_run):
    outputs, n_columns = first_run['outputs'], first_run['n_columns']
    train = pd.read_csv(outputs['train'], header=None)
    test_features = pd.read_csv(outputs['test_features'], header=None)
    test_labels = pd.read_csv(outputs['test_labels'], header=None)
    # 41,188 rows split 70 / 15 / 15 in time order.
    assert len(train) == 28831 and len(test_features) == len(test_labels) == 6179
    # Label first in train; features only in the Batch Transform input.
    assert train.shape[1] == n_columns + 1 and test_features.shape[1] == n_columns == 73
    assert set(train[0]) == {0, 1} and set(test_labels[0]) == {0, 1}


def test_features_are_fitted_on_train_split_only(first_run):
    schema = read_json(first_run['outputs']['feature_schema'])
    states = {feature['name']: feature['state'] for feature in schema['features']}
    raw = pd.read_csv(CONFIG['data']['path'], sep=';')
    train = raw.iloc[:28831]
    assert states['capped_campaign']['cap'] == float(train['campaign'].quantile(0.99))
    # "sep" only appears after the training period, so it is not a learned category.
    assert 'sep' not in states['contact_timing']['month']['categories']


def test_run_metadata_traces_inputs_and_outputs(first_run):
    metadata = read_json(first_run['run_metadata'])
    assert metadata['inputs']['raw_data']['sha256'] == file_hash(CONFIG['data']['path'])
    assert set(metadata['outputs']) == set(first_run['outputs'])
    assert metadata['features']['capped_campaign'] == 'v1'
    assert metadata['split_summary']['train'] == first_run['split_summary']['train']


def test_broken_feature_fails_the_run_and_writes_nothing(tmp_path, monkeypatch):
    # A deliberately broken input: a feature package whose feature outputs text.
    from bankpulse_core import feature_base
    from bankpulse_core.feature_base import Feature, register
    from bankpulse_core.pipeline import PipelineError

    monkeypatch.setattr(feature_base, '_REGISTRY', dict(feature_base._REGISTRY))
    package = types.ModuleType('broken_features')

    @register
    class JobAsText(Feature):
        name, version, inputs, owner, description = 'job_as_text', 'v1', ('job',), 'tests', 'Raw job text'

        def _transform(self, df):
            return df['job']

    monkeypatch.setitem(sys.modules, 'broken_features', package)
    config = config_writing_to(tmp_path)
    config['feature_package'] = 'broken_features'
    config['features'] = {'age': 'v1', 'job_as_text': 'v1'}

    with pytest.raises(PipelineError, match=r"non-numeric columns \['job_as_text'\]"):
        run(config)
    assert not any(tmp_path.iterdir())


def test_encoding_raw_test_records_matches_batch_input_file(first_run):
    # The endpoint path (raw records -> encode_records) must give byte-identical rows to the
    # batch path (pipeline.run -> test_features.csv), or live scores would differ from evaluation.
    from bankpulse_core.pipeline import encode_records, to_csv_payload

    schema = read_json(first_run['outputs']['feature_schema'])
    raw = pd.read_csv(CONFIG['data']['path'], sep=';')
    newest = raw.iloc[-200:].drop(columns=['y', 'duration'])           # "new" customers, no label
    with open(first_run['outputs']['test_features'], encoding='utf-8') as handle:
        batch_rows = handle.read().splitlines()[-200:]
    assert to_csv_payload(encode_records(newest.to_dict('records'), schema)).splitlines() == batch_rows
