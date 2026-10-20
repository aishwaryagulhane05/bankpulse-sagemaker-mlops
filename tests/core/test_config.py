"""Tests for bankpulse_core.config."""

import copy
from pathlib import Path

import pytest
import yaml

from bankpulse_core.config import ConfigError, load_config, validate_config

REPO = Path(__file__).resolve().parents[2]
VALID = {
    'data': {'path': 'data/raw/file.csv', 'separator': ';'},
    'contract': 'config/contract.yaml',
    'split': {'train': 0.7, 'validation': 0.15, 'test': 0.15},
    'output': {'processed_dir': 'data/processed', 'reports_dir': 'reports'},
    'feature_package': 'features',
    'features': {'age': 'v1'},
}


def with_change(section, key, value):
    """A copy of VALID with one setting replaced."""
    config = copy.deepcopy(VALID)
    if key is None:
        config[section] = value
    else:
        config[section][key] = value
    return config


def write_config(tmp_path, config, name='data_prep.yaml'):
    """Write config to <tmp>/config/<name>, mimicking the repository layout."""
    (tmp_path / 'config').mkdir(exist_ok=True)
    path = tmp_path / 'config' / name
    path.write_text(yaml.safe_dump(config) if config is not None else '', encoding='utf-8')
    return path


def test_valid_config_passes():
    validate_config(VALID)


def test_fractions_that_do_not_sum_to_one_raise():
    with pytest.raises(ConfigError, match='split: Fractions must add up to 1'):
        validate_config(with_change('split', None, {'train': 0.7, 'validation': 0.15, 'test': 0.05}))


@pytest.mark.parametrize('section, key, name', [
    ('data', 'path', 'data.path'),
    ('output', 'reports_dir', 'output.reports_dir'),
    ('contract', None, 'contract'),
])
def test_missing_setting_is_named(section, key, name):
    with pytest.raises(ConfigError, match=f'missing {name}'):
        validate_config(with_change(section, key, None))


def test_bad_feature_version_raises():
    with pytest.raises(ConfigError, match='versions must look like'):
        validate_config(with_change('features', None, {'age': 1}))


def test_bad_log_level_raises():
    with pytest.raises(ConfigError, match='log_level'):
        validate_config({**VALID, 'log_level': 'LOUD'})


def test_load_resolves_local_paths_against_repo_root(tmp_path):
    path = write_config(tmp_path, with_change('output', 'processed_dir', 's3://bucket/processed'))
    config = load_config(path)
    assert config['data']['path'] == str(tmp_path / 'data/raw/file.csv')
    assert config['contract'] == str(tmp_path / 'config/contract.yaml')
    assert config['output']['processed_dir'] == 's3://bucket/processed'     # S3 left unchanged
    assert config['log_level'] == 'INFO'


def test_empty_file_reports_first_missing_setting(tmp_path):
    with pytest.raises(ConfigError, match='missing data.path'):
        load_config(write_config(tmp_path, None, 'empty.yaml'))


def test_project_config_loads_and_points_at_real_files():
    config = load_config(REPO / 'config' / 'data_prep.yaml')
    assert Path(config['data']['path']).is_file()
    assert Path(config['contract']).is_file()
    assert len(config['features']) == 12
