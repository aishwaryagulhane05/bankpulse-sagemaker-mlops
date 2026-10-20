"""Lineage: run metadata (raw data hash, config, code version, feature versions, output hashes).

    run = start_run(config)
    run.record_file('inputs', 'raw_data', raw_path)
    run.record('features', {'age': 'v1', ...})
    run.record_file('outputs', 'train', train_path)
    run.save(processed_dir / 'run_metadata.json')

run_metadata.json answers "which data, code and settings produced these files?", so a
model trained on them can be traced back and reproduced.
"""

import json
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from bankpulse_core.io import file_hash, write_json

# The repository root, where git is asked for the code version.
_REPO_ROOT = Path(__file__).resolve().parents[1]

# Keys set by start_run(); record() may not overwrite them.
_RESERVED = ('run_id', 'started_at', 'finished_at', 'duration_seconds', 'code', 'environment', 'config',
             'inputs', 'outputs')
FILE_KINDS = ('inputs', 'outputs')


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _git(*args: str) -> Optional[str]:
    """Output of a git command in the repository, or None if git or the repository is unavailable."""
    try:
        result = subprocess.run(['git', *args], cwd=_REPO_ROOT, capture_output=True, text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _code_version() -> Dict[str, Any]:
    """Git commit and whether the working tree had uncommitted changes."""
    commit = _git('rev-parse', 'HEAD')
    status = _git('status', '--porcelain') if commit else None
    # Uncommitted changes mean the commit alone doesn't reproduce the run; record that.
    return {'git_commit': commit, 'uncommitted_changes': bool(status) if status is not None else None}


def _check_json_safe(key: str, value: Any) -> None:
    try:
        json.dumps(value)
    except TypeError as exc:
        raise TypeError(f'Lineage value for {key!r} is not JSON-safe: {exc}') from exc


class Run:
    """Metadata for one pipeline run; create it with start_run()."""

    def __init__(self, config: Mapping) -> None:
        # Step 1: config first, so a non-JSON value fails before any work is done.
        _check_json_safe('config', config)
        self._started = _now()
        # Step 2: what produced the run: settings, code version and library versions.
        self.metadata: Dict[str, Any] = {
            'run_id': f'{self._started:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}',
            'started_at': self._started.isoformat(),
            'code': _code_version(),
            'environment': {'python': platform.python_version(), 'pandas': pd.__version__, 'numpy': np.__version__},
            'config': config,
            'inputs': {},
            'outputs': {},
        }

    def record(self, key: str, value: Any) -> 'Run':
        """Store any JSON-safe value, e.g. feature versions or the split summary."""
        if key in _RESERVED:
            raise KeyError(f'{key!r} is set by the run itself; use record_file() for inputs and outputs')
        _check_json_safe(key, value)
        self.metadata[key] = value
        return self

    def record_file(self, kind: str, name: str, path) -> 'Run':
        """Store a file's path and SHA-256 under 'inputs' or 'outputs'."""
        if kind not in FILE_KINDS:
            raise ValueError(f'kind must be one of {FILE_KINDS}, got {kind!r}')
        self.metadata[kind][name] = {'path': str(path), 'sha256': file_hash(path)}
        return self

    def save(self, path) -> Path:
        """Add the finish time and write run_metadata.json."""
        finished = _now()
        self.metadata['finished_at'] = finished.isoformat()
        self.metadata['duration_seconds'] = round((finished - self._started).total_seconds(), 3)
        return write_json(self.metadata, path)


def start_run(config: Mapping) -> Run:
    """Begin recording a run for the given (already loaded) config."""
    return Run(config)
