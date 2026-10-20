"""Baseline model: logistic regression on the same features, trained locally.

Run from anywhere (laptop or a SageMaker Studio terminal):
    python scripts/baseline.py

Steps: run the data pipeline -> train on train -> choose the threshold on validation ->
evaluate once on test. Writes <output.reports_dir>/baseline_metrics.json, the bar the
SageMaker XGBoost model has to beat in notebook 02.
"""

import logging
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bankpulse_core.config import load_config
from bankpulse_core.evaluation import choose_threshold, evaluate
from bankpulse_core.io import read_csv, write_json
from bankpulse_core.log_utils import log_stage
from bankpulse_core.pipeline import run

CONFIG_PATH = Path(__file__).resolve().parents[1] / 'config' / 'data_prep.yaml'
# Fixed so the baseline is the same on every run and machine.
RANDOM_STATE = 42

logger = logging.getLogger('bankpulse_core.scripts.baseline')


def labelled(path):
    """Split a label-first, headerless XGBoost file into (features, labels)."""
    data = read_csv(path, header=None)
    return data.iloc[:, 1:].to_numpy(), data.iloc[:, 0].to_numpy()


def main() -> None:
    # Step 1: build the same files the SageMaker training job will use.
    config = load_config(CONFIG_PATH)
    outputs = run(config)['outputs']
    x_train, y_train = labelled(outputs['train'])
    x_validation, y_validation = labelled(outputs['validation'])
    x_test = read_csv(outputs['test_features'], header=None).to_numpy()
    y_test = read_csv(outputs['test_labels'], header=None)[0].to_numpy()

    # Step 2: scaling puts counts, rates and 0/1 columns on one scale, so the
    # regularisation treats every feature evenly.
    with log_stage('train baseline'):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
        model.fit(x_train, y_train)

    # Step 3: threshold from validation, then a single evaluation on test.
    with log_stage('evaluate baseline'):
        threshold = choose_threshold(y_validation, model.predict_proba(x_validation)[:, 1])
        metrics = {
            'model': 'logistic_regression',
            'validation': evaluate(y_validation, model.predict_proba(x_validation)[:, 1], threshold),
            'test': evaluate(y_test, model.predict_proba(x_test)[:, 1], threshold),
        }
        path = write_json(metrics, Path(config['output']['reports_dir']) / 'baseline_metrics.json')
    logger.info('test PR-AUC %.4f, lift %.2fx, F1 %.4f at threshold %.3f; saved %s', metrics['test']['pr_auc'],
                metrics['test']['lift_top_decile'], metrics['test']['f1'], threshold, path)


if __name__ == '__main__':
    main()
