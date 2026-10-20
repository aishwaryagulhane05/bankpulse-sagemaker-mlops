"""I/O: read and write local files and S3 objects, compute file hashes.

    read_csv / write_csv     tables
    read_json / write_json   reports, feature schema, run metadata
    file_hash                SHA-256 of a file, recorded in lineage

Local paths only for now; S3 support is added in BP-403.

Files are always written with "\n" line endings and UTF-8, so the same data gives the
same bytes (and the same hash) on Windows, macOS and SageMaker Studio (Linux).
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Union

import pandas as pd

PathLike = Union[str, Path]

# Read files in 1 MB pieces when hashing, so large files don't have to fit in memory.
_CHUNK_SIZE = 1024 * 1024


def _prepare_output(path: PathLike) -> Path:
    """Create the parent folder, so callers don't have to."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: PathLike, sep: str = ',') -> pd.DataFrame:
    """Read a CSV file with a header row."""
    return pd.read_csv(path, sep=sep)


def write_csv(df: pd.DataFrame, path: PathLike, header: bool = True, sep: str = ',') -> Path:
    """Write df without its index; header=False gives the XGBoost training format."""
    path = _prepare_output(path)
    # lineterminator='\n': pandas otherwise uses "\r\n" on Windows, which changes the hash.
    df.to_csv(path, sep=sep, header=header, index=False, lineterminator='\n', encoding='utf-8')
    return path


def read_json(path: PathLike) -> Any:
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def write_json(obj: Any, path: PathLike) -> Path:
    """Write JSON-safe data, indented for readable diffs; raises TypeError otherwise."""
    path = _prepare_output(path)
    # Step 1: serialise first, so a non-JSON value fails before an empty file is created.
    text = json.dumps(obj, indent=2, ensure_ascii=False) + '\n'
    # Step 2: newline='\n' keeps the bytes identical on every operating system.
    with open(path, 'w', encoding='utf-8', newline='\n') as handle:
        handle.write(text)
    return path


def file_hash(path: PathLike) -> str:
    """SHA-256 of the file's bytes, as 64 hex characters."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b''):
            digest.update(chunk)
    return digest.hexdigest()
