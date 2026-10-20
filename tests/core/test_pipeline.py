"""Tests for bankpulse_core.pipeline."""

import json
import sys
import types

import pandas as pd
import pytest

from bankpulse_core.feature_base import Feature, FeatureError
from bankpulse_core.io import read_json
from bankpulse_core.pipeline import (PipelineError, build_matrix, build_schema, check_outputs, encode_records,
                                     encode_target, to_csv_payload, write_outputs)


# Two small, unregistered features: core tests never import the features package.
class Doubled(Feature):
    name, version, inputs, owner, description = 'doubled', 'v1', ('x',), 'tests', 'Twice x'

    def _transform(self, df):
        return df['x'] * 2


class Capped(Feature):
    name, version, inputs, owner, description = 'capped', 'v1', ('x',), 'tests', 'x capped at the training max'

    def _fit(self, train):
        self.cap = float(train['x'].max())

    def _transform(self, df):
        return df['x'].clip(upper=self.cap)

    def _state(self):
        return {'cap': self.cap}

    def _load_state(self, state):
        self.cap = state['cap']


TRAIN =pd.DataFrame({'x': [1, 2, 3], 'y': ['no', 'yes', 'no']}, index=[0, 1, 2])
TEST = pd.DataFrame({'x': [10, 1], 'y': ['yes', 'no']}, index=[3, 4])


def fitted():
    return [Doubled().fit(TRAIN), Capped().fit(TRAIN)]


def test_matrix_joins_features_in_order_and_keeps_rows():
    matrix = build_matrix(fitted(), TEST)
    assert list(matrix.columns) == ['doubled', 'capped']
    assert matrix.loc[3].tolist() == [20, 3] and matrix.index.tolist() == [3, 4]


def test_duplicate_output_columns_are_rejected():
    with pytest.raises(PipelineError, match=r"share column names \['doubled'\]"):
        build_matrix([Doubled().fit(TRAIN), Doubled().fit(TRAIN)], TEST)


def test_target_is_encoded_as_one_for_positive_label():
    assert encode_target(TRAIN['y'], 'yes').tolist() == [0, 1, 0]


def test_schema_holds_column_order_and_feature_states():
    features = fitted()
    schema = build_schema(features, ['doubled', 'capped'], 'y', 'yes', 'tests')
    assert schema['columns'] == ['doubled', 'capped']
    assert schema['features'][1] == {'name': 'capped', 'version': 'v1', 'inputs': ['x'], 'state': {'cap': 3.0}}
    json.dumps(schema)


def write_all(tmp_path):
    features = fitted()
    splits = {'train': TRAIN, 'validation': TEST, 'test': TEST}
    matrices = {name: build_matrix(features, part) for name, part in splits.items()}
    labels = {name: encode_target(part['y'], 'yes') for name, part in splits.items()}
    schema = build_schema(features, list(matrices['train'].columns), 'y', 'yes', 'tests')
    return write_outputs(matrices, labels, schema, tmp_path)


def test_train_and_validation_files_are_label_first_without_header(tmp_path):
    written = write_all(tmp_path)
    assert (tmp_path / 'train' / 'train.csv').read_text() == '0,2,1\n1,4,2\n0,6,3\n'
    assert (tmp_path / 'validation' / 'validation.csv').read_text() == '1,20,3\n0,2,1\n'
    assert written['train'] == str(tmp_path / 'train' / 'train.csv')


def test_batch_input_has_features_only_and_labels_are_separate(tmp_path):
    write_all(tmp_path)
    assert (tmp_path / 'batch-input' / 'test_features.csv').read_text() == '20,3\n2,1\n'
    assert (tmp_path / 'labels' / 'test_labels.csv').read_text() == '1\n0\n'
    assert (tmp_path / 'batch-input' / 'validation_features.csv').exists()


def test_each_training_channel_folder_holds_one_file(tmp_path):
    # A training channel reads every file under its prefix, so nothing else may sit there.
    write_all(tmp_path)
    for channel in ('train', 'validation'):
        assert [p.name for p in (tmp_path / channel).iterdir()] == [f'{channel}.csv']


def test_schema_file_is_written(tmp_path):
    written = write_all(tmp_path)
    assert read_json(written['feature_schema'])['columns'] == ['doubled', 'capped']
    assert set(written) == {'train', 'validation', 'validation_features', 'validation_labels',
                            'test_features', 'test_labels', 'feature_schema'}


# ---- check_outputs ---------------------------------------------------------------------

GOOD_MATRIX = pd.DataFrame({'a': [1, 2], 'b': [0.5, 1.5]}, index=[7, 8])
GOOD_LABELS = pd.Series([0, 1], index=[7, 8])
SCHEMA = {'columns': ['a', 'b']}


def check(matrix=GOOD_MATRIX, labels=GOOD_LABELS, schema=SCHEMA):
    check_outputs({'test': matrix}, {'test': labels}, schema, forbidden=['duration'])


def test_good_outputs_pass():
    check()


@pytest.mark.parametrize('matrix, message', [
    (GOOD_MATRIX[['b', 'a']], 'columns differ from the schema order'),
    (GOOD_MATRIX.assign(b=['x', 'y']), r"non-numeric columns \['b'\]"),
    (GOOD_MATRIX.assign(b=[True, False]), r"non-numeric columns \['b'\]"),
    (GOOD_MATRIX.assign(b=[float('nan'), 1.0]), r"missing or infinite values in \['b'\]"),
    (GOOD_MATRIX.assign(b=[float('inf'), 1.0]), r"missing or infinite values in \['b'\]"),
])
def test_bad_matrix_is_rejected(matrix, message):
    with pytest.raises(PipelineError, match=message):
        check(matrix=matrix)


@pytest.mark.parametrize('column', ['duration', 'duration__capped'])
def test_forbidden_output_column_is_rejected(column):
    matrix = GOOD_MATRIX.rename(columns={'b': column})
    with pytest.raises(PipelineError, match='forbidden columns in output'):
        check(matrix=matrix, schema={'columns': ['a', column]})


def test_labels_must_be_binary_and_aligned():
    with pytest.raises(PipelineError, match=r'labels must be 0/1, got \[0, 2\]'):
        check(labels=pd.Series([0, 2], index=[7, 8]))
    with pytest.raises(PipelineError, match='different rows'):
        check(labels=pd.Series([0, 1], index=[0, 1]))


def test_all_problems_are_listed_at_once():
    with pytest.raises(PipelineError) as error:
        check(matrix=GOOD_MATRIX[['b', 'a']].assign(a=['x', 'y']), labels=pd.Series([5, 1], index=[7, 8]))
    assert str(error.value).count('\n- ') == 3


# ---- encode_records ---------------------------------------------------------------------------

@pytest.fixture
def schema(monkeypatch):
    """A schema for Doubled and Capped, registered in a throwaway registry and fake package."""
    from bankpulse_core import feature_base

    monkeypatch.setattr(feature_base, '_REGISTRY', {})
    monkeypatch.setitem(sys.modules, 'fake_features', types.ModuleType('fake_features'))
    feature_base.register(Doubled)
    feature_base.register(Capped)
    return json.loads(json.dumps(build_schema(fitted(), ['doubled', 'capped'], 'y', 'yes', 'fake_features')))


def test_records_are_encoded_with_training_state(schema):
    # The cap learned on TRAIN (3) applies to a live record with x = 10.
    matrix = encode_records([{'x': 10}, {'x': 1}], schema)
    assert matrix.values.tolist() == [[20, 3], [2, 1]]


def test_csv_payload_has_no_header_or_index(schema):
    assert to_csv_payload(encode_records(pd.DataFrame({'x': [10, 1]}), schema)) == '20,3\n2,1\n'


def test_record_missing_an_input_is_rejected(schema):
    with pytest.raises(FeatureError, match=r"needs column\(s\) \['x'\]"):
        encode_records([{'age': 30}], schema)


def test_no_records_is_rejected(schema):
    with pytest.raises(PipelineError, match='No records'):
        encode_records([], schema)


def test_schema_column_mismatch_is_rejected(schema):
    with pytest.raises(PipelineError, match='differ from feature_schema.json'):
        encode_records([{'x': 1}], {**schema, 'columns': ['capped', 'doubled']})
