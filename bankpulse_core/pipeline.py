"""Pipeline: runs ingest -> validate -> split -> registered features -> write, driven by config.

    result = run(load_config('config/data_prep.yaml'))

Building blocks (this module):
    build_matrix(features, df)    fitted features -> one numeric table, in config order
    encode_target(series, label)  "yes"/"no" -> 1/0
    build_schema(...)             column order + every feature's state (feature_schema.json)
    write_outputs(...)            files in the built-in XGBoost CSV format

Output layout (one folder per S3 prefix, see below):
    train/train.csv                          label first, no header   -> training channel
    validation/validation.csv                label first, no header   -> validation channel
    batch-input/<split>_features.csv         features only, no header -> Batch Transform input
    labels/<split>_labels.csv                label only, no header    -> evaluation
    feature-schema/feature_schema.json       column order and feature states -> endpoint

A SageMaker training channel reads every file under its S3 prefix, so the label-first
files each get a folder of their own; feature-only and label files never sit next to them.
"""

import importlib
import logging
from pathlib import Path
from typing import Dict, Mapping, Sequence, Union

import pandas as pd

from bankpulse_core.contract import check_rules, check_schema, drop_forbidden, load_contract
from bankpulse_core.feature_base import Feature, check_inputs, resolve
from bankpulse_core.io import read_csv, write_csv, write_json
from bankpulse_core.lineage import start_run
from bankpulse_core.log_utils import log_stage, setup_logging
from bankpulse_core.split import check_no_overlap, split_summary, time_split

logger = logging.getLogger(__name__)

# Splits whose label-first file is a training-job channel.
CHANNEL_SPLITS = ('train', 'validation')
SCHEMA_VERSION = 1


class PipelineError(ValueError):
    """The pipeline produced or received data it can't safely write."""


def build_matrix(features: Sequence[Feature], df: pd.DataFrame) -> pd.DataFrame:
    """Join the outputs of fitted features, side by side, in the given (config) order."""
    # Step 1: each feature returns columns indexed like df, so concat lines rows up.
    matrix = pd.concat([feature.transform(df) for feature in features], axis=1)
    # Step 2: two features writing the same column name would silently shadow each other.
    duplicated = matrix.columns[matrix.columns.duplicated()].tolist()
    if duplicated:
        raise PipelineError(f'Feature outputs share column names {duplicated}; rename one of the features')
    return matrix


def encode_target(series: pd.Series, positive_label: str) -> pd.Series:
    """1 where the label equals positive_label, else 0 (XGBoost binary:logistic labels)."""
    return (series.astype(str) == str(positive_label)).astype(int).rename(series.name)


def build_schema(features: Sequence[Feature], columns: Sequence[str], target: str, positive_label: str) -> Dict:
    """Everything the endpoint needs to encode a raw record exactly like the training data."""
    return {
        'schema_version': SCHEMA_VERSION,
        'target': target,
        'positive_label': str(positive_label),
        # Column order of every feature file; the model's inputs are positional.
        'columns': list(columns),
        # Rebuilt at inference time with get_feature(name, version)().load_state(state).
        'features': [{'name': f.name, 'version': f.version, 'inputs': list(f.inputs), 'state': f.state()}
                     for f in features],
    }


def write_outputs(matrices: Mapping[str, pd.DataFrame], labels: Mapping[str, pd.Series], schema: Mapping,
                  out_dir: Union[str, Path]) -> Dict[str, str]:
    """Write every output file; returns output name -> path (for lineage)."""
    out_dir = Path(out_dir)
    written: Dict[str, str] = {}
    for split, matrix in matrices.items():
        if split in CHANNEL_SPLITS:
            # Built-in XGBoost CSV: label in the first column, no header, no index.
            labelled = pd.concat([labels[split], matrix], axis=1)
            written[split] = str(write_csv(labelled, out_dir / split / f'{split}.csv', header=False))
        if split != 'train':
            # Batch Transform scores feature-only rows; labels are kept apart for evaluation.
            written[f'{split}_features'] = str(write_csv(matrix, out_dir / 'batch-input' / f'{split}_features.csv',
                                                         header=False))
            written[f'{split}_labels'] = str(write_csv(labels[split].to_frame(),
                                                       out_dir / 'labels' / f'{split}_labels.csv', header=False))
    written['feature_schema'] = str(write_json(schema, out_dir / 'feature-schema' / 'feature_schema.json'))
    return written


def check_outputs(matrices: Mapping[str, pd.DataFrame], labels: Mapping[str, pd.Series], schema: Mapping,
                  forbidden: Sequence[str]) -> None:
    """Raise PipelineError listing every reason the outputs are unsafe to write.

    XGBoost reads columns by position and accepts any number, so a wrong column order, a
    text value turned into NaN or a leaked column would train silently on bad data.
    """
    problems = []
    for split, matrix in matrices.items():
        # Step 1: every split has exactly the schema's columns, in the schema's order.
        if list(matrix.columns) != list(schema['columns']):
            problems.append(f'{split}: columns differ from the schema order')
        # Step 2: no output column is a forbidden (leakage) column or derived from one by name.
        leaked = [c for c in matrix.columns if any(c == f or c.startswith(f'{f}__') for f in forbidden)]
        if leaked:
            problems.append(f'{split}: forbidden columns in output {leaked}')
        # Step 3: every column is numeric (bool and text would be written as words).
        not_numeric = [c for c in matrix.columns if not pd.api.types.is_numeric_dtype(matrix[c])
                       or pd.api.types.is_bool_dtype(matrix[c])]
        if not_numeric:
            problems.append(f'{split}: non-numeric columns {not_numeric[:5]}')
        else:
            # Step 4: no missing or infinite values (only checkable once columns are numeric).
            bad = matrix.columns[~matrix.apply(lambda column: column.notna() & column.abs().ne(float('inf'))).all()]
            if len(bad):
                problems.append(f'{split}: missing or infinite values in {list(bad)[:5]}')
        # Step 5: labels are 0/1 and line up with the feature rows.
        label = labels[split]
        if not label.index.equals(matrix.index):
            problems.append(f'{split}: labels and features have different rows')
        values = sorted(label.unique().tolist())          # .tolist() gives plain ints for the message
        if not set(values) <= {0, 1}:
            problems.append(f'{split}: labels must be 0/1, got {values}')
    if problems:
        raise PipelineError('Outputs failed checks, nothing was written:\n- ' + '\n- '.join(problems))


def run(config: Mapping) -> Dict:
    """Run data preparation end to end and return the written paths and the split summary.

    Steps: read -> check contract -> drop leakage -> time split -> fit features on train
    -> transform every split -> write files -> save run metadata. Every step is logged
    with its duration, and any failed check stops the run before files are written.
    """
    setup_logging(config['log_level'])
    lineage = start_run(config)
    out_dir = Path(config['output']['processed_dir'])

    # Step 1: read the raw file and record its hash, so outputs can be traced back to it.
    with log_stage('read raw data'):
        raw = read_csv(config['data']['path'], sep=config['data']['separator'])
        lineage.record_file('inputs', 'raw_data', config['data']['path'])

    # Step 2: the data must match the contract; then leakage columns are removed.
    with log_stage('check contract'):
        contract = load_contract(config['contract'])
        check_schema(raw, contract)
        check_rules(raw, contract)
        data = drop_forbidden(raw, contract)

    # Step 3: time-ordered split, with no row in two splits.
    with log_stage('split'):
        splits = time_split(data, config['split'])
        check_no_overlap(splits)
        lineage.record('split_summary', split_summary(splits, contract.target, contract.positive_label))

    # Step 4: build the configured features. Importing the feature package registers them;
    # the framework itself never imports it. Each feature may only read contract columns,
    # and everything a feature learns comes from the TRAIN split.
    with log_stage('fit features'):
        importlib.import_module(config['feature_package'])
        features = resolve(config['features'])
        for feature in features:
            check_inputs(feature, contract.allowed_columns, contract.forbidden_columns)
            feature.fit(splits['train'])
        lineage.record('features', {feature.name: feature.version for feature in features})

    # Step 5: the same fitted features transform every split, so all splits share columns.
    with log_stage('transform splits'):
        matrices = {name: build_matrix(features, part) for name, part in splits.items()}
        labels = {name: encode_target(part[contract.target], contract.positive_label)
                  for name, part in splits.items()}
        schema = build_schema(features, list(matrices['train'].columns), contract.target, contract.positive_label)
        # Last gate: nothing is written unless every split passes.
        check_outputs(matrices, labels, schema, contract.forbidden_columns)

    # Step 6: write the files and record every output's hash.
    with log_stage('write outputs'):
        written = write_outputs(matrices, labels, schema, out_dir)
        for name, path in written.items():
            lineage.record_file('outputs', name, path)
        metadata_path = lineage.save(out_dir / 'run_metadata.json')

    logger.info('%d feature columns, %s', len(schema['columns']),
                ', '.join(f'{name}: {len(part)} rows' for name, part in splits.items()))
    return {'outputs': written, 'run_metadata': str(metadata_path),
            'split_summary': lineage.metadata['split_summary'], 'n_columns': len(schema['columns'])}
