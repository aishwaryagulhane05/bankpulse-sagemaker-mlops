"""Pipeline step 3 (Processing, XGBoost container): challenger vs champion on the newest holdout.

Both models score the SAME raw customers, each encoded with its OWN feature_schema.json (a
retrained model may have learned different category lists or caps). Each model's threshold
is chosen on the validation batches, then both are evaluated once on the holdout batches.

Writes to /opt/ml/processing/evaluation:
    evaluation.json     both models' metrics, the comparison, and challenger_ok (1/0) for the Condition step
    model_quality.json  the challenger's holdout metrics in SageMaker's model-quality format (registry)
"""

import argparse
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import xgboost as xgb  # noqa: E402

from bankpulse_core.config import load_config  # noqa: E402
from bankpulse_core.evaluation import choose_threshold, compare, evaluate  # noqa: E402
from bankpulse_core.ingest import load_batches  # noqa: E402
from bankpulse_core.io import read_json, write_json  # noqa: E402
from bankpulse_core.monitoring import SUBGROUP_RATIO_ALERT, group_from_one_hot, subgroup_rates  # noqa: E402
from bankpulse_core.pipeline import encode_records  # noqa: E402
from bankpulse_core.split import window_split  # noqa: E402

BASE = Path('/opt/ml/processing')


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batches', required=True)
    parser.add_argument('--window', default='12,2,2')
    parser.add_argument('--margin', type=float, default=0.01, help='allowed shortfall in holdout PR-AUC')
    parser.add_argument('--base', default=str(BASE), help='folder holding the mounted inputs and outputs')
    return parser.parse_args(argv)


def load_booster(folder: Path) -> xgb.Booster:
    """The model inside model.tar.gz (built-in XGBoost saves it as 'xgboost-model')."""
    with tarfile.open(folder / 'model.tar.gz') as tar:
        tar.extractall(folder)
    booster = xgb.Booster()
    booster.load_model(str(folder / 'xgboost-model'))
    return booster


def score_model(booster, schema, validation, holdout):
    """Threshold on validation, then holdout metrics and age-band selection rates."""
    encoded = {name: encode_records(part.drop(columns=['y']), schema) for name, part in
               (('validation', validation), ('holdout', holdout))}
    labels = {name: part['y'].eq(schema['positive_label']).astype(int).to_numpy() for name, part in
              (('validation', validation), ('holdout', holdout))}
    scores = {name: booster.predict(xgb.DMatrix(matrix.to_numpy())) for name, matrix in encoded.items()}
    threshold = choose_threshold(labels['validation'], scores['validation'])
    metrics = evaluate(labels['holdout'], scores['holdout'], threshold)
    groups = subgroup_rates(scores['holdout'], group_from_one_hot(encoded['holdout'], 'age_band'), threshold)
    return {**metrics, 'subgroup_min_max_ratio': groups['min_max_ratio'], 'subgroups': groups['groups']}


def main(argv=None) -> dict:
    args = parse_args(argv)
    base = Path(args.base)
    config = load_config(HERE / 'config' / 'data_prep.yaml')

    # Step 1: the same rolling window the challenger was trained with; raw rows, no duration.
    data, sizes = load_batches(args.batches, separator=config['data']['separator'])
    train, val, test = (int(n) for n in args.window.split(','))
    splits = window_split(data.drop(columns=['duration']), sizes, train, val, test)

    # Step 2: each model with its own encoding.
    results = {}
    for role in ('challenger', 'champion'):
        booster = load_booster(base / role / 'model')
        schema = read_json(base / role / 'schema' / 'feature_schema.json')
        results[role] = score_model(booster, schema, splits['validation'], splits['test'])

    # Step 3: the decision rule from docs/retraining.md, section 4.
    comparison = compare(results['challenger'], results['champion'])
    accurate = comparison['pr_auc']['difference'] >= -args.margin
    fair = (results['challenger']['subgroup_min_max_ratio'] >= results['champion']['subgroup_min_max_ratio']
            or results['challenger']['subgroup_min_max_ratio'] >= SUBGROUP_RATIO_ALERT)
    evaluation = {'challenger_ok': int(accurate and fair), 'accurate': accurate, 'fair': fair, 'margin': args.margin,
                  'holdout_rows': int(len(splits['test'])), 'holdout_positive_rate': results['challenger']['positive_rate'],
                  **results, 'comparison': comparison}

    out = base / 'evaluation'
    write_json(evaluation, out / 'evaluation.json')
    holdout, cm = results['challenger'], results['challenger']['confusion_matrix']
    write_json({'binary_classification_metrics': {
        'auc': {'value': holdout['roc_auc'], 'standard_deviation': 'NaN'},
        'precision': {'value': holdout['precision'], 'standard_deviation': 'NaN'},
        'recall': {'value': holdout['recall'], 'standard_deviation': 'NaN'},
        'f1': {'value': holdout['f1'], 'standard_deviation': 'NaN'},
        'confusion_matrix': {'0': {'0': cm['tn'], '1': cm['fp']}, '1': {'0': cm['fn'], '1': cm['tp']}},
    }}, out / 'model_quality.json')
    print('challenger PR-AUC', results['challenger']['pr_auc'], '| champion PR-AUC', results['champion']['pr_auc'],
          '| challenger_ok', evaluation['challenger_ok'])
    return evaluation


if __name__ == '__main__':
    main()
