"""Tests for bankpulse_core.lineage."""

import re
import subprocess
from pathlib import Path

import pytest

from bankpulse_core import lineage
from bankpulse_core.config import load_config
from bankpulse_core.io import file_hash, read_json
from bankpulse_core.lineage import start_run

REPO = Path(__file__).resolve().parents[2]
CONFIG = {'split': {'train': 0.7, 'test': 0.3}, 'features': {'age': 'v1'}}


def test_run_metadata_holds_raw_hash_config_commit_and_feature_versions(tmp_path):
    # The acceptance check for BP-301, with the project's real config and raw file.
    config = load_config(REPO / 'config' / 'data_prep.yaml')
    run = start_run(config)
    run.record_file('inputs', 'raw_data', config['data']['path'])
    run.record('features', dict(config['features']))
    saved = read_json(run.save(tmp_path / 'run_metadata.json'))

    assert saved['inputs']['raw_data']['sha256'] == file_hash(config['data']['path'])
    assert saved['config'] == config
    assert re.fullmatch(r'[0-9a-f]{40}', saved['code']['git_commit'])
    assert isinstance(saved['code']['uncommitted_changes'], bool)
    assert saved['features']['capped_campaign'] == 'v1' and len(saved['features']) == 12


def test_save_adds_finish_time_and_environment(tmp_path):
    saved = read_json(start_run(CONFIG).save(tmp_path / 'nested' / 'run_metadata.json'))
    assert saved['finished_at'] >= saved['started_at']
    assert saved['duration_seconds'] >= 0
    assert set(saved['environment']) == {'python', 'pandas', 'numpy'}
    assert re.fullmatch(r'\d{8}T\d{6}Z-[0-9a-f]{8}', saved['run_id'])


def test_output_files_are_recorded_with_hashes(tmp_path):
    output = tmp_path / 'train.csv'
    output.write_text('1,30\n0,45\n')
    run = start_run(CONFIG).record_file('outputs', 'train', output)
    assert run.metadata['outputs']['train'] == {'path': str(output), 'sha256': file_hash(output)}


def test_two_runs_get_different_ids():
    assert start_run(CONFIG).metadata['run_id'] != start_run(CONFIG).metadata['run_id']


def test_non_json_values_are_rejected_when_recorded():
    run = start_run(CONFIG)
    with pytest.raises(TypeError, match="'split_summary' is not JSON-safe"):
        run.record('split_summary', {'train': {1, 2}})
    with pytest.raises(TypeError, match="'config' is not JSON-safe"):
        start_run({'path': Path('data')})


def test_reserved_keys_cannot_be_overwritten():
    with pytest.raises(KeyError, match='set by the run itself'):
        start_run(CONFIG).record('inputs', {})


def test_unknown_file_kind_is_rejected(tmp_path):
    (tmp_path / 'x.csv').write_text('a\n')
    with pytest.raises(ValueError, match='kind must be one of'):
        start_run(CONFIG).record_file('artifacts', 'x', tmp_path / 'x.csv')


def test_missing_git_is_recorded_as_none(monkeypatch):
    def no_git(*args, **kwargs):
        raise FileNotFoundError('git')

    monkeypatch.setattr(lineage.subprocess, 'run', no_git)
    assert start_run(CONFIG).metadata['code'] == {'git_commit': None, 'uncommitted_changes': None}


def test_git_error_is_recorded_as_none(monkeypatch):
    def not_a_repo(*args, **kwargs):
        raise subprocess.CalledProcessError(128, 'git')

    monkeypatch.setattr(lineage.subprocess, 'run', not_a_repo)
    assert start_run(CONFIG).metadata['code']['git_commit'] is None
