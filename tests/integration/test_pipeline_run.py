"""End-to-end tests: pipeline.run() on the real dataset with the project's config."""

import copy
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
