"""Tests for bankpulse_core.io."""

import pandas as pd
import pytest

from bankpulse_core.io import file_hash, read_csv, read_json, write_csv, write_json

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
