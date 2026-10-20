"""Choose the XGBoost hyperparameters on the validation split, locally (no AWS needed).

Run from anywhere:
    python scripts/tune_xgboost.py

Uses xgboost 1.7.6, the same version as SageMaker's built-in container (1.7-1), on the files
pipeline.run() writes, so a setting chosen here behaves the same in the training job.
Only validation PR-AUC is used to choose; the test split is never looked at.
Writes <output.reports_dir>/xgboost_tuning.json; the best setting goes into notebook 02.
"""

import itertools
from pathlib import Path

import xgboost as xgb

from bankpulse_core.config import load_config
from bankpulse_core.io import read_csv, write_json
from bankpulse_core.pipeline import run

CONFIG_PATH = Path(__file__).resolve().parents[1] / 'config' / 'data_prep.yaml'

# Fixed settings: the same as the SageMaker job.
FIXED = {'objective': 'binary:logistic', 'eval_metric': 'aucpr', 'subsample': 0.8, 'seed': 0}
GRID = {'max_depth': [2, 3, 4, 6], 'min_child_weight': [1, 5, 20, 50], 'eta': [0.05, 0.1],
        'colsample_bytree': [0.5, 0.8]}
NUM_ROUND, EARLY_STOPPING = 500, 30


def matrix(path):
    data = read_csv(path, header=None)
    return xgb.DMatrix(data.iloc[:, 1:].to_numpy(), label=data.iloc[:, 0].to_numpy())


def main() -> None:
    # Step 1: the same train / validation files the training job reads.
    config = load_config(CONFIG_PATH)
    outputs = run(config)['outputs']
    train, validation = matrix(outputs['train']), matrix(outputs['validation'])

    # Step 2: every combination, each stopped early on validation PR-AUC.
    results = []
    for values in itertools.product(*GRID.values()):
        params = {**FIXED, **dict(zip(GRID, values))}
        booster = xgb.train(params, train, NUM_ROUND, evals=[(validation, 'validation')],
                            early_stopping_rounds=EARLY_STOPPING, verbose_eval=False)
        results.append({**dict(zip(GRID, values)), 'validation_aucpr': round(booster.best_score, 4),
                        'best_round': booster.best_iteration + 1})

    # Step 3: rank by validation PR-AUC and save.
    results.sort(key=lambda r: r['validation_aucpr'], reverse=True)
    write_json({'features': config['features'], 'fixed': FIXED, 'num_round': NUM_ROUND,
                'early_stopping_rounds': EARLY_STOPPING, 'best': results[0], 'top10': results[:10]},
               Path(config['output']['reports_dir']) / 'xgboost_tuning.json')
    print('best on validation:', results[0])


if __name__ == '__main__':
    main()
