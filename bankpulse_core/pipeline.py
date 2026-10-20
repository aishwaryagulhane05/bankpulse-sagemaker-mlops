"""Pipeline: runs ingest -> validate -> split -> registered features -> write, driven by config.

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

from pathlib import Path
from typing import Dict, Mapping, Sequence, Union

import pandas as pd

from bankpulse_core.feature_base import Feature
from bankpulse_core.io import write_csv, write_json

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
