"""Tests for bankpulse_core.contract."""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from bankpulse_core.contract import (ContractError, check_rules, check_schema, drop_forbidden, load_contract,
                                     quality_report)

REPO = Path(__file__).resolve().parents[2]

# A small contract with every kind of check.
CONTRACT_YAML = {
    'target': 'y',
    'positive_label': 'yes',
    'forbidden_columns': ['duration'],
    'sentinels': {'pdays': 999},
    'columns': {
        'age': {'type': 'int', 'min': 17, 'max': 100},
        'job': {'type': 'category', 'values': ['admin.', 'unknown']},
        'duration': {'type': 'int', 'min': 0},
        'pdays': {'type': 'int', 'min': 0, 'max': 999},
        'previous': {'type': 'int', 'min': 0},
        'euribor3m': {'type': 'float'},
        'y': {'type': 'category', 'values': ['no', 'yes']},
    },
    'rules': [{
        'name': 'days_since_contact_needs_previous_contacts',
        'description': 'A real pdays value means the customer was contacted before',
        'when': {'column': 'pdays', 'not_equals': 999},
        'then': {'column': 'previous', 'min': 1},
    }],
}

GOOD = pd.DataFrame({
    'age': [30, 45, 60, 30], 'job': ['admin.', 'unknown', 'admin.', 'admin.'], 'duration': [100, 200, 300, 100],
    'pdays': [999, 3, 999, 999], 'previous': [0, 1, 0, 0], 'euribor3m': [4.8, 1.3, 0.7, 4.8],
    'y': ['no', 'yes', 'no', 'no'],
})


def write(tmp_path, data):
    path = tmp_path / 'contract.yaml'
    # sort_keys=False keeps the column order, as in the real contract file.
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
    return path


@pytest.fixture
def contract(tmp_path):
    return load_contract(write(tmp_path, CONTRACT_YAML))


# ---- load_contract ------------------------------------------------------------------

def test_allowed_columns_exclude_target_and_forbidden(contract):
    assert contract.allowed_columns == ['age', 'job', 'pdays', 'previous', 'euribor3m']


def test_unquoted_yes_no_in_contract_is_rejected(tmp_path):
    path = tmp_path / 'contract.yaml'
    path.write_text('target: y\npositive_label: "yes"\ncolumns:\n  y: {type: category, values: [no, yes]}\n')
    with pytest.raises(ContractError, match='quote "yes" and "no"'):
        load_contract(path)


def test_rule_with_unknown_column_is_rejected(tmp_path):
    data = json.loads(json.dumps(CONTRACT_YAML))
    data['rules'][0]['then']['column'] = 'prev'
    with pytest.raises(ContractError, match="unknown column 'prev'"):
        load_contract(write(tmp_path, data))


# ---- check_schema -------------------------------------------------------------------

def test_good_data_passes(contract):
    check_schema(GOOD, contract)
    check_rules(GOOD, contract)


def test_bad_label_is_rejected(contract):
    with pytest.raises(ContractError, match=r"y: unexpected values \['maybe'\]"):
        check_schema(GOOD.assign(y=['no', 'maybe', 'no', 'no']), contract)


def test_missing_and_unexpected_columns_are_reported(contract):
    with pytest.raises(ContractError, match=r"missing columns \['job'\][\s\S]*unexpected columns \['extra'\]"):
        check_schema(GOOD.drop(columns='job').assign(extra=1), contract)


def test_wrong_type_and_out_of_range_values_are_reported(contract):
    bad = GOOD.assign(age=[30, 120, 10, 30], euribor3m=['a', 'b', 'c', 'd'])
    with pytest.raises(ContractError) as error:
        check_schema(bad, contract)
    message = str(error.value)
    # Every problem is listed at once.
    assert 'age: 1 values below the minimum 17' in message
    assert 'age: 1 values above the maximum 100' in message
    assert 'euribor3m: expected float' in message


def test_missing_values_are_reported(contract):
    with pytest.raises(ContractError, match='job: 1 missing values'):
        check_schema(GOOD.assign(job=['admin.', None, 'admin.', 'admin.']), contract)


# ---- check_rules, drop_forbidden ---------------------------------------------------------

def test_broken_rule_is_reported_with_count_and_first_row(contract):
    broken = GOOD.assign(pdays=[999, 3, 5, 999], previous=[0, 1, 0, 0])
    with pytest.raises(ContractError, match='days_since_contact_needs_previous_contacts: 1 rows, first at index 2'):
        check_rules(broken, contract)


def test_drop_forbidden_removes_duration(contract):
    cleaned = drop_forbidden(GOOD, contract)
    assert 'duration' not in cleaned.columns
    assert 'duration' in GOOD.columns                 # the input is not modified


# ---- quality_report --------------------------------------------------------------------

def test_quality_report_values_and_json_safe(contract):
    report = quality_report(GOOD, contract)
    assert report == {
        'rows': 4, 'columns': 7, 'duplicate_rows': 1, 'positive_rate': 0.25,
        'missing_values': {}, 'unknown_share': {'job': 0.25}, 'sentinel_share': {'pdays': 0.75},
    }
    json.dumps(report)


# ---- the project's contract and data -------------------------------------------------------

def test_project_contract_loads_and_real_data_passes():
    contract = load_contract(REPO / 'config' / 'contract.yaml')
    data = pd.read_csv(REPO / 'data' / 'raw' / 'bank-additional-full.csv', sep=';')
    check_schema(data, contract)
    check_rules(data, contract)
    assert len(contract.rules) == 4
