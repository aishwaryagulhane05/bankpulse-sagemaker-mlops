"""Tests for bankpulse_core.io."""

import io
from pathlib import Path

import pandas as pd
import pytest

from bankpulse_core import io as bankpulse_io
from bankpulse_core.io import file_hash, list_objects, read_csv, read_json, split_s3_uri, upload_file, write_csv, write_json

FRAME = pd.DataFrame({'age': [30, 45], 'job': ['admin.', 'unknown'], 'euribor3m': [4.857, 0.635]})


def test_csv_round_trip_gives_identical_data(tmp_path):
    path = write_csv(FRAME, tmp_path / 'nested' / 'folder' / 'data.csv')    # parent folders created
    pd.testing.assert_frame_equal(read_csv(path), FRAME)


def test_csv_with_semicolon_separator(tmp_path):
    path = write_csv(FRAME, tmp_path / 'data.csv', sep=';')
    assert path.read_text(encoding='utf-8').splitlines()[0] == 'age;job;euribor3m'
    pd.testing.assert_frame_equal(read_csv(path, sep=';'), FRAME)


def test_csv_without_header_for_xgboost(tmp_path):
    path = write_csv(FRAME[['age', 'euribor3m']], tmp_path / 'train.csv', header=False)
    assert path.read_text(encoding='utf-8') == '30,4.857\n45,0.635\n'


def test_files_use_unix_line_endings(tmp_path):
    assert b'\r\n' not in write_csv(FRAME, tmp_path / 'data.csv').read_bytes()
    assert b'\r\n' not in write_json({'a': [1, 2]}, tmp_path / 'data.json').read_bytes()


def test_json_round_trip(tmp_path):
    report = {'rows': 41188, 'positive_rate': 0.1127, 'columns': ['age', 'job'], 'note': 'café'}
    assert read_json(write_json(report, tmp_path / 'reports' / 'report.json')) == report


def test_non_json_value_raises_and_writes_nothing(tmp_path):
    with pytest.raises(TypeError):
        write_json({'bad': {1, 2}}, tmp_path / 'bad.json')
    assert not (tmp_path / 'bad.json').exists()


def test_hash_is_stable_and_detects_a_one_byte_change(tmp_path):
    first = write_csv(FRAME, tmp_path / 'a.csv')
    second = write_csv(FRAME, tmp_path / 'b.csv')
    assert file_hash(first) == file_hash(second)
    assert len(file_hash(first)) == 64

    changed = tmp_path / 'c.csv'
    changed.write_bytes(first.read_bytes().replace(b'30', b'31'))
    assert file_hash(changed) != file_hash(first)


# ---- S3 (a fake client: no AWS account or credentials needed) ------------------------------

class FakeS3:
    """In-memory stand-in for the boto3 S3 client methods io.py uses."""

    def __init__(self, page_size=1000):
        self.objects, self.page_size = {}, page_size

    def put_object(self, Bucket, Key, Body):
        self.objects[(Bucket, Key)] = bytes(Body)

    def get_object(self, Bucket, Key):
        return {'Body': io.BytesIO(self.objects[(Bucket, Key)])}

    def upload_file(self, Filename, Bucket, Key):
        self.objects[(Bucket, Key)] = Path(Filename).read_bytes()

    def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None):
        keys = sorted(k for b, k in self.objects if b == Bucket and k.startswith(Prefix))
        start = int(ContinuationToken or 0)
        page = keys[start:start + self.page_size]
        more = start + self.page_size < len(keys)
        return {'Contents': [{'Key': k} for k in page], 'IsTruncated': more,
                **({'NextContinuationToken': str(start + self.page_size)} if more else {})}


@pytest.fixture
def s3(monkeypatch):
    fake = FakeS3()
    monkeypatch.setattr(bankpulse_io, '_s3', lambda: fake)
    return fake


def test_split_s3_uri():
    assert split_s3_uri('s3://bucket/a/b.csv') == ('bucket', 'a/b.csv')
    with pytest.raises(ValueError, match='Not an S3 URI'):
        split_s3_uri('data/a.csv')


def test_csv_and_json_round_trip_through_s3(s3):
    uri = 's3://bucket/bank-campaign-mlops/raw/data.csv'
    assert write_csv(FRAME, uri) == uri
    pd.testing.assert_frame_equal(read_csv(uri), FRAME)
    write_json({'rows': 2}, 's3://bucket/reports/r.json')
    assert read_json('s3://bucket/reports/r.json') == {'rows': 2}


def test_s3_object_has_same_bytes_and_hash_as_local_file(s3, tmp_path):
    local = write_csv(FRAME, tmp_path / 'data.csv', header=False)
    uri = upload_file(local, 's3://bucket/train/train.csv')
    assert file_hash(uri) == file_hash(local)
    pd.testing.assert_frame_equal(read_csv(uri, header=None), read_csv(local, header=None))


def test_upload_needs_a_full_key(s3, tmp_path):
    local = write_csv(FRAME, tmp_path / 'data.csv')
    with pytest.raises(ValueError, match='full object key'):
        upload_file(local, 's3://bucket/train/')


def test_list_objects_follows_every_page(monkeypatch):
    fake = FakeS3(page_size=2)
    monkeypatch.setattr(bankpulse_io, '_s3', lambda: fake)
    for name in ('a', 'b', 'c', 'd', 'e'):
        fake.put_object('bucket', f'train/{name}.csv', b'1')
    fake.put_object('bucket', 'test/x.csv', b'1')
    assert list_objects('s3://bucket/train/') == [f's3://bucket/train/{n}.csv' for n in 'abcde']


def test_missing_credentials_give_set_up_guidance(monkeypatch):
    from botocore.exceptions import NoCredentialsError

    class NoCredentials:
        def get_object(self, **kwargs):
            raise NoCredentialsError()

    monkeypatch.setattr(bankpulse_io, '_s3', lambda: NoCredentials())
    with pytest.raises(RuntimeError, match='BP-400'):
        read_csv('s3://bucket/raw/data.csv')
