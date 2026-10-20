"""Ingest: accept new data files as versioned batches, or reject them with a report.

    result = ingest_file('s3://.../raw/incoming/2010-11.csv', contract, 's3://.../raw/batches')
    data, sizes = load_batches('s3://.../raw/batches')

Accepted files are copied to <batches>/<arrival-id>.csv and listed, in arrival order, in
<batches>/manifest.json with their row count, positive rate and SHA-256. Nothing is ever
deleted or overwritten: a file already in the manifest (same hash) is skipped, and a file
that breaks the data contract is recorded in <batches>/../rejected/ with every problem.
Works with local paths and S3 URIs alike (via io.py).
"""

from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Dict, List, Tuple

import pandas as pd

from bankpulse_core.contract import Contract, ContractError, check_rules, check_schema
from bankpulse_core.io import PathLike, file_hash, read_bytes, read_csv, read_json, write_bytes, write_json

MANIFEST = 'manifest.json'


def _posix(path: PathLike) -> str:
    return str(path).replace('\\', '/').rstrip('/')


def _join(base: PathLike, *parts: str) -> str:
    """base/part1/part2 for both local paths and s3:// URIs."""
    return '/'.join([_posix(base), *parts])


def _parent(path: PathLike) -> str:
    return _posix(path).rsplit('/', 1)[0]


def read_manifest(batches: PathLike) -> Dict:
    """The batch manifest, or an empty one for a new batches folder."""
    try:
        return read_json(_join(batches, MANIFEST))
    except Exception as error:
        # A missing file locally, or NoSuchKey in S3: no batch has been accepted yet.
        if isinstance(error, FileNotFoundError) or 'NoSuchKey' in str(error):
            return {'batches': []}
        raise


def month_batches(df: pd.DataFrame, month_column: str = 'month') -> List[pd.DataFrame]:
    """Split time-ordered rows into calendar-month batches: a new batch starts when the month changes."""
    batch_number = (df[month_column] != df[month_column].shift()).cumsum()
    return [part for _, part in df.groupby(batch_number, sort=True)]


def ingest_file(source: PathLike, contract: Contract, batches: PathLike, separator: str = ';') -> Dict:
    """Validate one new file and add it as the next batch; returns what happened and why."""
    manifest = read_manifest(batches)
    digest = file_hash(source)
    name = PurePosixPath(_posix(source)).stem

    # Step 1: the same file twice must not count twice (e.g. an upload event delivered again).
    if any(batch['sha256'] == digest for batch in manifest['batches']):
        return {'status': 'duplicate', 'source': str(source), 'sha256': digest}

    # Step 2: the data contract decides; every problem is reported, and nothing is half-added.
    data = read_csv(source, sep=separator)
    try:
        check_schema(data, contract)
        check_rules(data, contract)
    except ContractError as error:
        report = {'status': 'rejected', 'source': str(source), 'sha256': digest,
                  'checked_at': datetime.now(timezone.utc).isoformat(), 'problems': str(error).splitlines()[1:]}
        write_json(report, _join(_parent(batches), 'rejected', f'{name}.json'))
        return report

    # Step 3: copy the exact bytes in as the next batch (so it keeps the source's hash), then
    # record it in the manifest.
    arrival = f'{len(manifest["batches"]) + 1:04d}-{name}'
    write_bytes(read_bytes(source), _join(batches, f'{arrival}.csv'))
    entry = {'id': arrival, 'file': f'{arrival}.csv', 'source': str(source), 'rows': int(len(data)),
             'positive_rate': round(float((data[contract.target].astype(str) == contract.positive_label).mean()), 4),
             'sha256': digest, 'added_at': datetime.now(timezone.utc).isoformat()}
    manifest['batches'].append(entry)
    write_json(manifest, _join(batches, MANIFEST))
    return {'status': 'accepted', **entry}


def load_batches(batches: PathLike, separator: str = ';') -> Tuple[pd.DataFrame, List[int]]:
    """Every accepted batch concatenated in arrival order, plus each batch's row count."""
    manifest = read_manifest(batches)
    if not manifest['batches']:
        raise ValueError(f'No batches in {batches}')
    parts = [read_csv(_join(batches, batch['file']), sep=separator) for batch in manifest['batches']]
    return pd.concat(parts, ignore_index=True), [len(part) for part in parts]
