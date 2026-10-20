# Screenshots

Screenshots from the run in the AWS Academy Learner Lab (`us-east-1`), in notebook order. The account ID shown belongs to a temporary lab account. Back to the [README](../../README.md).

## Notebook 01: S3 data lake

**01-sagemaker-session-role-bucket.png**: SageMaker session from the laptop: region `us-east-1`, role `LabRole`, the default bucket and the data-lake prefix.

![01-sagemaker-session-role-bucket.png](01-sagemaker-session-role-bucket.png)

**01-s3-data-lake-folders.png**: S3 data lake `bank-campaign-mlops/`: one folder per stage (batch-input, batch-predictions, feature-schema, inference-capture, labels, model artifacts, ...).

![01-s3-data-lake-folders.png](01-s3-data-lake-folders.png)

**01-s3-data-lake-folders-continued.png**: The rest of the data lake: raw, reports, retraining, run-metadata, train and validation.

![01-s3-data-lake-folders-continued.png](01-s3-data-lake-folders-continued.png)

**01-s3-labels.png**: `labels/`: validation and test labels, kept apart from the features so Batch Transform only sees inputs.

![01-s3-labels.png](01-s3-labels.png)

## Notebook 02: training job and Batch Transform

**02-training-job-completed.png**: Training job `bankpulse-xgboost-…-728`: **Completed** in 2 min 36 s, reading the `train/` and `validation/` channels.

![02-training-job-completed.png](02-training-job-completed.png)

**02-training-job-metrics.png**: Metrics reported by the built-in XGBoost container: validation PR-AUC 0.2259, train PR-AUC 0.2178.

![02-training-job-metrics.png](02-training-job-metrics.png)

**02-training-job-input-channels.png**: Input channels: `train` and `validation`, `text/csv`, from the data lake.

![02-training-job-input-channels.png](02-training-job-input-channels.png)

**02-training-job-output-artifact.png**: Output artifact `model.tar.gz`, and the job running as `LabRole`.

![02-training-job-output-artifact.png](02-training-job-output-artifact.png)

**02-s3-model-artifacts.png**: `model-artifacts/` in S3: the first training run and the final one (economic indicators removed).

![02-s3-model-artifacts.png](02-s3-model-artifacts.png)

**02-batch-transform-jobs.png**: Batch Transform jobs **Completed**: the final run's validation and test jobs (`…-19-41-37-*`), plus the first run's. The `xgb-*` jobs belong to another project in the same lab.

![02-batch-transform-jobs.png](02-batch-transform-jobs.png)

**02-s3-batch-predictions.png**: `batch-predictions/`: one probability per customer for validation and test (`.out` files).

![02-s3-batch-predictions.png](02-s3-batch-predictions.png)

## Notebook 03: Model Registry and Model Card

**03-model-registry-group.png**: Model Registry group `bankpulse-term-deposit`, with its description and 5 versions.

![03-model-registry-group.png](03-model-registry-group.png)

**03-model-registry-versions.png**: Versions: **v1 Approved** (notebook 03), v2–v4 **Pending manual approval** (retraining pipeline). v5 was approved later in Studio.

![03-model-registry-versions.png](03-model-registry-versions.png)

**03-model-card-overview.png**: Model Card `bankpulse-term-deposit-v1` (Draft): XGBoost 1.7-1 container, binary classification, model description.

![03-model-card-overview.png](03-model-card-overview.png)

**03-model-cards-list.png**: Model Cards list: `bankpulse-term-deposit-v1` with risk rating **Medium**. The other card belongs to another project.

![03-model-cards-list.png](03-model-cards-list.png)

## Notebook 04: real-time endpoint and Data Capture

**04-sagemaker-model-from-package.png**: The SageMaker Model created at deployment from **model package version 1** (the approved registry version).

![04-sagemaker-model-from-package.png](04-sagemaker-model-from-package.png)

**04-endpoints-inservice.png**: Endpoint `bankpulse-term-deposit-20261021-011916` **InService**.

![04-endpoints-inservice.png](04-endpoints-inservice.png)

**04-endpoint-summary.png**: Endpoint summary: real-time, the invocation URL, and model container logs.

![04-endpoint-summary.png](04-endpoint-summary.png)

**04-endpoint-config-data-capture.png**: Endpoint configuration: **Data capture Active**, 100% of requests and responses, written to `inference-capture/`.

![04-endpoint-config-data-capture.png](04-endpoint-config-data-capture.png)

**04-scored-customers.png**: The newest 200 customers scored by the endpoint: 82 above the threshold of 0.106; top 10 shown with their real outcome.

![04-scored-customers.png](04-scored-customers.png)

**04-s3-inference-capture.png**: `inference-capture/` with one folder per endpoint.

![04-s3-inference-capture.png](04-s3-inference-capture.png)

**04-capture-jsonl-files.png**: Data Capture files (`.jsonl`) by date and hour; each line holds the request rows and returned scores.

![04-capture-jsonl-files.png](04-capture-jsonl-files.png)

## Notebook 05: monitoring and CloudWatch

**05-s3-monitoring-report.png**: The monitoring report saved to `monitoring-reports/<timestamp>/monitoring_report.json`.

![05-s3-monitoring-report.png](05-s3-monitoring-report.png)

**05-cloudwatch-alarms-overview.png**: CloudWatch overview: the `BankPulse/ModelMonitoring` alarm and its recent `max_feature_psi` value.

![05-cloudwatch-alarms-overview.png](05-cloudwatch-alarms-overview.png)

**05-cloudwatch-drift-alarm.png**: Drift alarm `…-feature-drift`: max feature PSI 3.43 against the 0.2 threshold (the red line).

![05-cloudwatch-drift-alarm.png](05-cloudwatch-drift-alarm.png)

## Notebook 06: continuous training

**06-eventbridge-retrain-rules.png**: EventBridge rules `bankpulse-retrain-on-drift` and `bankpulse-retrain-on-upload` (the monthly rule is further down the list), all **Enabled**.

![06-eventbridge-retrain-rules.png](06-eventbridge-retrain-rules.png)

**06-registry-version-2-lineage.png**: Lineage of version 2, registered by the pipeline: datasets → training job → model → version, approval and model group.

![06-registry-version-2-lineage.png](06-registry-version-2-lineage.png)

**06-registry-version-5-metrics.png**: Version 5 (month-26 run): Train and Evaluate **Complete**, pending approval, with its training metrics.

![06-registry-version-5-metrics.png](06-registry-version-5-metrics.png)

**06-registry-version-5-approved.png**: Version 5 **Approved** in Studio (a manual decision; approval doesn't deploy anything).

![06-registry-version-5-approved.png](06-registry-version-5-approved.png)

**06-registry-version-5-lineage.png**: Lineage of version 5: two datasets and the image → training job → model → version 5 → model group.

![06-registry-version-5-lineage.png](06-registry-version-5-lineage.png)

**06-endpoint-unchanged-after-retraining.png**: After all retraining runs, the endpoint is still **InService** and unchanged (it serves version 1 until a deliberate deploy).

![06-endpoint-unchanged-after-retraining.png](06-endpoint-unchanged-after-retraining.png)

## Results UI

**ui-overview.png**: Streamlit results UI (`streamlit run ui/app.py`), Overview: the problem, why it matters, key numbers, and the live connection to the lab.

![ui-overview.png](ui-overview.png)

## Still to capture

- Notebook 01: object details of `train/train.csv`
- Notebook 02: training job hyperparameters (Configurations tab); evaluation table, PR curve and confusion matrix (also in `notebooks/executed/02_training_evaluation.ipynb`)
- Notebook 03: Model Card evaluation and additional-information sections
- Notebook 05: CloudWatch dashboard `bankpulse-model-monitoring`; Clarify / Model Monitor results (also in `notebooks/executed/05_bias_drift_monitoring.ipynb`); SageMaker endpoints list empty after clean-up
- Notebook 06: the `bankpulse-retrain` pipeline graph and execution list in Studio
- UI: SageMaker tracking, Monitoring and Score customers sections
