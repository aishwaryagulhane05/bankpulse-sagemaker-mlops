"""I/O: read and write local files and S3 objects, compute file hashes.

    read_csv / write_csv     tables
    read_json / write_json   reports, feature schema, run metadata
    file_hash                SHA-256 of a file, recorded in lineage
    upload_file              copy a local file to S3
    list_objects             S3 URIs under a prefix

Every path may be local or an S3 URI ("s3://bucket/key"). S3 access uses boto3's
default credential chain: the LabRole in SageMaker Studio, or an AWS profile on a laptop.
No keys are ever passed in code or config (see issue BP-400 for the set-up).

Files are always written with "\n" line endings and UTF-8, so the same data gives the
same bytes (and the same hash) on Windows, macOS and SageMaker Studio (Linux).
"""

import functools
import hashlib
import io
import json
from pathlib import Path
from typing import Any, List, Tuple, Union

import pandas as pd

PathLike = Union[str, Path]

# Read files in 1 MB pieces when hashing, so large files don't have to fit in memory.
_CHUNK_SIZE = 1024 * 1024
_S3_PREFIX = 's3://'


# ---- S3 helpers --------------------------------------------------------------------------

def is_s3(path: PathLike) -> bool:
    return str(path).startswith(_S3_PREFIX)


def split_s3_uri(uri: str) -> Tuple[str, str]:
    """'s3://bucket/a/b.csv' -> ('bucket', 'a/b.csv')."""
    bucket, _, key = str(uri)[len(_S3_PREFIX):].partition('/')
    if not str(uri).startswith(_S3_PREFIX) or not bucket:
        raise ValueError(f'Not an S3 URI: {uri!r} (expected s3://bucket/key)')
    return bucket, key


@functools.lru_cache(maxsize=1)
def _s3():
    """One S3 client per process, created on first use so local work never needs boto3 set up."""
    import boto3

    return boto3.client('s3')


def _s3_call(method: str, **kwargs):
    """Call an S3 client method, turning a missing-credentials error into set-up guidance."""
    from botocore.exceptions import NoCredentialsError

    try:
        return getattr(_s3(), method)(**kwargs)
    except NoCredentialsError as exc:
        raise RuntimeError('No AWS credentials found. In SageMaker Studio they come from the LabRole; '
                           'on a laptop, set AWS_PROFILE to your lab profile (see issue BP-400).') from exc


def _read_bytes(path: PathLike) -> bytes:
    if is_s3(path):
        bucket, key = split_s3_uri(path)
        return _s3_call('get_object', Bucket=bucket, Key=key)['Body'].read()
    return Path(path).read_bytes()


def _write_bytes(data: bytes, path: PathLike) -> Union[str, Path]:
    """Write bytes locally (creating the folder) or to S3; returns the path written."""
    if is_s3(path):
        bucket, key = split_s3_uri(path)
        _s3_call('put_object', Bucket=bucket, Key=key, Body=data)
        return str(path)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


# ---- tables and JSON ----------------------------------------------------------------------

def read_csv(path: PathLike, sep: str = ',', header: Union[int, None] = 0) -> pd.DataFrame:
    """Read a CSV file; header=None for files without a header row (XGBoost format)."""
    return pd.read_csv(io.BytesIO(_read_bytes(path)), sep=sep, header=header)


def write_csv(df: pd.DataFrame, path: PathLike, header: bool = True, sep: str = ',') -> Union[str, Path]:
    """Write df without its index; header=False gives the XGBoost training format."""
    # lineterminator='\n': pandas otherwise uses "\r\n" on Windows, which changes the hash.
    text = df.to_csv(sep=sep, header=header, index=False, lineterminator='\n')
    return _write_bytes(text.encode('utf-8'), path)


def read_json(path: PathLike) -> Any:
    return json.loads(_read_bytes(path).decode('utf-8'))


def write_json(obj: Any, path: PathLike) -> Union[str, Path]:
    """Write JSON-safe data, indented for readable diffs; raises TypeError otherwise."""
    # Serialise first, so a non-JSON value fails before an empty file is created.
    text = json.dumps(obj, indent=2, ensure_ascii=False) + '\n'
    return _write_bytes(text.encode('utf-8'), path)


# ---- hashes, uploads and listings -------------------------------------------------------------

def file_hash(path: PathLike) -> str:
    """SHA-256 of the file's bytes, as 64 hex characters (local file or S3 object)."""
    if is_s3(path):
        return hashlib.sha256(_read_bytes(path)).hexdigest()
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b''):
            digest.update(chunk)
    return digest.hexdigest()


def upload_file(local_path: PathLike, uri: str) -> str:
    """Copy a local file to an S3 URI (multipart for large files); returns the URI."""
    bucket, key = split_s3_uri(uri)
    if not key or key.endswith('/'):
        raise ValueError(f'{uri!r} needs a full object key, e.g. s3://bucket/train/train.csv')
    _s3_call('upload_file', Filename=str(local_path), Bucket=bucket, Key=key)
    return uri


def list_objects(prefix_uri: str) -> List[str]:
    """Every object under an S3 prefix, as sorted s3:// URIs (all pages)."""
    bucket, prefix = split_s3_uri(prefix_uri)
    uris = []
    token = None
    while True:
        # list_objects_v2 returns at most 1,000 keys per call; follow the continuation token.
        kwargs = {'Bucket': bucket, 'Prefix': prefix, **({'ContinuationToken': token} if token else {})}
        page = _s3_call('list_objects_v2', **kwargs)
        uris += [f'{_S3_PREFIX}{bucket}/{item["Key"]}' for item in page.get('Contents', [])]
        if not page.get('IsTruncated'):
            return sorted(uris)
        token = page['NextContinuationToken']
