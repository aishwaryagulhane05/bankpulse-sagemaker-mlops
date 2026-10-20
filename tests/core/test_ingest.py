"""Tests for bankpulse_core.ingest, with the project's real contract and monthly batches of the real data."""

from pathlib import Path

import pandas as pd
import pytest

from bankpulse_core.contract import load_contract
from bankpulse_core.ingest import ingest_file, load_batches, month_batches, read_manifest
from bankpulse_core.io import file_hash, read_json

REPO = Path(__file__).resolve().parents[2]
CONTRACT = load_contract(REPO / 'config' / 'contract.yaml')
RAW = pd.read_csv(REPO / 'data' / 'raw' / 'bank-additional-full.csv', sep=';')


def write_month(tmp_path, frame, name):
    path = tmp_path / 'incoming' / f'{name}.csv'
    path.parent.mkdir(exist_ok=True)
    frame.to_csv(path, sep=';', index=False)
    return path


def test_history_splits_into_26_calendar_months():
    months = month_batches(RAW)
    assert len(months) == 26
    assert months[0]['month'].iloc[0] == 'may' and months[-1]['month'].iloc[0] == 'nov'
    assert sum(len(m) for m in months) == len(RAW)


def test_valid_file_is_accepted_byte_for_byte_and_listed(tmp_path):
    months = month_batches(RAW)
    source = write_month(tmp_path, months[-1], '2010-11')
    result = ingest_file(source, CONTRACT, tmp_path / 'batches')
    assert result['status'] == 'accepted' and result['id'] == '0001-2010-11'
    assert result['rows'] == 128 and result['positive_rate'] == 0.4531
    assert file_hash(tmp_path / 'batches' / '0001-2010-11.csv') == file_hash(source)
    assert [b['id'] for b in read_manifest(tmp_path / 'batches')['batches']] == ['0001-2010-11']


def test_same_file_twice_is_a_duplicate(tmp_path):
    source = write_month(tmp_path, month_batches(RAW)[-1], '2010-11')
    ingest_file(source, CONTRACT, tmp_path / 'batches')
    assert ingest_file(source, CONTRACT, tmp_path / 'batches')['status'] == 'duplicate'
    assert len(read_manifest(tmp_path / 'batches')['batches']) == 1


def test_bad_file_is_rejected_with_a_report_and_not_added(tmp_path):
    bad = month_batches(RAW)[-1].assign(y='maybe')
    result = ingest_file(write_month(tmp_path, bad, 'broken'), CONTRACT, tmp_path / 'batches')
    assert result['status'] == 'rejected'
    assert any("y: unexpected values ['maybe']" in problem for problem in result['problems'])
    assert read_json(tmp_path / 'rejected' / 'broken.json')['status'] == 'rejected'
    assert read_manifest(tmp_path / 'batches') == {'batches': []}


def test_load_batches_in_arrival_order(tmp_path):
    for i, month in enumerate(month_batches(RAW)[-3:]):
        ingest_file(write_month(tmp_path, month, f'm{i}'), CONTRACT, tmp_path / 'batches')
    data, sizes = load_batches(tmp_path / 'batches')
    assert sizes == [303, 204, 128]                       # Sep, Oct, Nov 2010
    assert data['month'].tolist()[0] == 'sep' and data['month'].tolist()[-1] == 'nov'


def test_no_batches_is_an_error(tmp_path):
    with pytest.raises(ValueError, match='No batches'):
        load_batches(tmp_path / 'batches')
