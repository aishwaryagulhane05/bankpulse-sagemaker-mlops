# AWS run artifacts

Files downloaded from the AWS Academy Learner Lab run on 2026-10-20, kept here because the lab account and its S3 bucket expire.

| File | Created by | Notes |
|---|---|---|
| `feature_schema.json` | Feature pipeline (notebook 01) | Schema v1, 68 encoded columns |
| `xgboost_metrics.json` | Training + evaluation (notebook 02) | Job `bankpulse-xgboost-2026-10-20-19-38-10-728`; test PR-AUC 0.4673, threshold 0.1064 |
| `model_comparison.json` | Evaluation (notebook 02) | Candidate 0.4673 vs baseline 0.4712 PR-AUC |
| `validation_features.csv.out` | Batch Transform (notebook 02) | One probability per validation row |
| `test_features.csv.out` | Batch Transform (notebook 02) | One probability per test row |
| `model_quality.json` | Model Registry metrics (notebook 03) | AUC 0.629, F1 0.369 |
| `bankpulse-term-deposit-v1-MODELCARD.pdf` | Model Card export (notebook 03) | Card for model `bankpulse-term-deposit-v1` |
| `monitoring_report.json` | Drift report (notebook 05) | Generated 2026-10-20T19:55:16Z over 200 captured records |

`xgboost_metrics.json`, `model_comparison.json` and `monitoring_report.json` are identical to the copies in [`reports/`](../../reports).
