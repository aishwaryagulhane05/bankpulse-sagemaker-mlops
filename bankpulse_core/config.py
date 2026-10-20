"""Config: load and check settings from config/data_prep.yaml.

    config = load_config('config/data_prep.yaml')

load_config() reads the YAML, checks it with validate_config(), and turns the local paths
into absolute paths, so notebooks, tests and jobs can run from any working directory.
"""

import re
from pathlib import Path
from typing import Dict, Mapping, Optional, Union

import yaml

from bankpulse_core.split import _check_fractions

# Settings the pipeline needs, as (section, key); key None means a top-level setting.
REQUIRED = (
    ('data', 'path'), ('data', 'separator'),
    ('contract', None), ('split', None),
    ('output', 'processed_dir'), ('output', 'reports_dir'),
    ('feature_package', None), ('features', None),
)

# Path settings that load_config() resolves against the repository root (optional ones may be absent).
PATH_KEYS = (('data', 'path'), ('data', 'batches'), ('contract', None), ('output', 'processed_dir'),
             ('output', 'reports_dir'))

# Optional rolling window over monthly batches: how many of the newest batches go to each split.
WINDOW_KEYS = ('train_batches', 'validation_batches', 'test_batches')

LOG_LEVELS = ('DEBUG', 'INFO', 'WARNING', 'ERROR')
_VERSION_PATTERN = re.compile(r'^v\d+$')


class ConfigError(ValueError):
    """The data prep config is missing a setting or has an invalid value."""


def _get(config: Mapping, section: str, key: Optional[str]):
    value = config.get(section)
    return value if key is None else (value or {}).get(key)


def validate_config(config: Mapping) -> None:
    """Raise ConfigError if a required setting is missing or a value is invalid."""
    # Step 1: every required setting is present; the error names the missing setting.
    for section, key in REQUIRED:
        if _get(config, section, key) in (None, ''):
            raise ConfigError(f'Config is missing {section}.{key}' if key else f'Config is missing {section}')

    # Step 2: split fractions, checked with the same rules time_split() applies.
    try:
        _check_fractions(config['split'])
    except ValueError as exc:
        raise ConfigError(f'split: {exc}') from exc

    # Step 3: features is a name -> version map, e.g. {"age": "v1"}.
    features = config['features']
    if not isinstance(features, Mapping):
        raise ConfigError('features must map feature names to versions, e.g. "age: v1"')
    bad = {name: version for name, version in features.items() if not _VERSION_PATTERN.match(str(version))}
    if bad:
        raise ConfigError(f'features: versions must look like "v1", got {bad}')

    # Step 4: an optional rolling window needs a whole positive number of batches per split.
    window = config.get('window')
    if window is not None:
        bad = {k: window.get(k) for k in WINDOW_KEYS if not (isinstance(window.get(k), int) and window.get(k) > 0)}
        if bad:
            raise ConfigError(f'window: {", ".join(WINDOW_KEYS)} must be positive integers, got {bad}')

    # Step 5: the log level is optional, but must be one logging understands.
    level = config.get('log_level', 'INFO')
    if level not in LOG_LEVELS:
        raise ConfigError(f'log_level must be one of {LOG_LEVELS}, got {level!r}')


def load_config(path: Union[str, Path]) -> Dict:
    """Read, check and return the config, with local paths made absolute.

    Paths in the file are relative to the repository root, which is the parent of the
    config/ folder holding the file. S3 paths (s3://...) are left unchanged.
    """
    path = Path(path).resolve()
    # Step 1: read the YAML; an empty file becomes {} so validation reports what's missing.
    with open(path, encoding='utf-8') as handle:
        config = yaml.safe_load(handle) or {}

    # Step 2: check everything before any value is used.
    validate_config(config)

    # Step 3: make local paths absolute against the repository root.
    root = path.parent.parent
    for section, key in PATH_KEYS:
        value = _get(config, section, key)
        if value is None:                      # optional setting not used in this config
            continue
        value = str(value)
        if not value.startswith('s3://'):
            value = str(root / value)
        if key is None:
            config[section] = value
        else:
            config[section][key] = value

    # Step 4: fill in optional settings so callers can rely on them.
    config.setdefault('log_level', 'INFO')
    return config
