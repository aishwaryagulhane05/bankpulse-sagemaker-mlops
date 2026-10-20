# bankpulse-sagemaker-mlops

An end-to-end MLOps system that predicts whether a bank customer will subscribe to a term deposit, **before** the call is made. It uses a tested Python framework for the data, and AWS SageMaker for training, batch scoring, the model registry, the Model Card, a real-time endpoint with Data Capture, and monitoring with Amazon CloudWatch.

> **Status:** complete. Every stage has run end to end in an AWS Academy Learner Lab (`us-east-1`), and the framework, pipeline and UI have 266 passing tests.

![Architecture: local framework, SageMaker components in the AWS lab, and monitoring](architecture/architecture.png)

## Results at a glance

| | Logistic baseline | SageMaker XGBoost |
|---|---|---|
| Validation PR-AUC | 0.222 | **0.228** |
| Test PR-AUC | **0.471** | 0.467 |
| Test F1 at the validation threshold | 0.306 | **0.369** |
| Test recall | 0.224 | **0.302** |
| Test lift, top 10% | 1.19× | **1.24×** |
| Subscribers found in test (true positives) | 533 | **718** |

- **Comparison:** both models use the same 68 inputs. The threshold is chosen on validation, and test is evaluated once.
- **Outcome:** XGBoost finds 35% more subscribers at the chosen threshold. Ranking quality (PR-AUC) is essentially tied.
- **Endpoint check:** the real-time endpoint returns exactly the same scores as Batch Transform for the same customers (maximum difference 0.0).
- **Monitoring:** live traffic shows strong drift (17 drifted inputs, prediction PSI 2.95). It also shows an age-band selection ratio of 0.30, which triggers a fairness review (see [Monitoring](#monitoring)).

## Problem

A bank runs phone campaigns to sell term deposits, and only about 11% of contacted customers subscribe. The model scores each customer's probability of subscribing, so the bank can call the most promising customers first.

- **Type:** binary classification (target `y`: yes / no).
- **Leakage:** only information known *before* the call may be used, so `duration` (call length) is forbidden.
- **Time:** the data is in date order. Splits are time-ordered: the oldest 70% of rows train the model, the next 15% validate it, and the newest 15% test it. A random split would leak future behaviour into training.

## Dataset

[UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) (Moro, Cortez & Rita, 2014), file `bank-additional-full.csv`.

| Property | Value |
|---|---|
| Records | 41,188 phone contacts, May 2008 – November 2010 |
| Columns | 20 inputs + target `y` |
| Positive rate | 11.27% overall; **5.6% train, 10.7% validation, 38.5% test** (strong drift) |
| Quality | no missing values, 12 duplicate rows, `unknown` in 6 categorical columns (up to 20.9% for `default`), `pdays = 999` (never contacted) in 96.3% of rows |
| Location | `data/raw/` (column descriptions in `bank-additional-names.txt`) |

## How it works

### 1. Data framework (local Python, tested)

| Module | What it does |
|---|---|
| `bankpulse_core/contract.py` | Checks the raw file against `config/contract.yaml`: 21 columns, their types, allowed values and ranges, 4 business rules, and the forbidden `duration`. It also produces the quality report. |
| `bankpulse_core/split.py` | Time-ordered 70 / 15 / 15 split, with checks for overlap and time order. |
| `bankpulse_core/feature_base.py` | Feature interface and versioned registry, keyed by `(name, version)`. |
| `bankpulse_core/encoding.py` | One-hot encoding with an `__other` column for unseen values, quantile capping, and saved state for each. |
| `features/*.py` | Versioned business features (see [Features](#features)). |
| `bankpulse_core/pipeline.py` | `run(config)` does read → validate → split → fit features on train → transform → output checks → XGBoost CSVs. `encode_records()` encodes live records for the endpoint. |
| `bankpulse_core/lineage.py` | `run_metadata.json`: raw-data and output SHA-256 hashes, git commit, config, feature versions. |
| `bankpulse_core/evaluation.py` | Batch Transform parsing, threshold choice on validation, PR-AUC, lift and comparison. |
| `bankpulse_core/monitoring.py` | Data Capture parsing, PSI drift, prediction drift, age-band selection rates, the report. |
| `bankpulse_core/aws.py`, `io.py` | Lab session and `LabRole`, local and S3 I/O, using the default credential chain (never keys in code). |

The core never imports `features/`. `config/data_prep.yaml` chooses which features and versions run, so a new or upgraded feature needs no pipeline change.

### 2. SageMaker in the AWS lab (notebooks)

| Notebook | SageMaker / AWS components | Result in the lab |
|---|---|---|
| `01_data_preparation` | S3 data lake, SageMaker session, `LabRole` | Quality report, pipeline outputs uploaded to `bank-campaign-mlops/`, hash verified in S3 |
| `02_training_evaluation` | **Training Job** (built-in XGBoost 1.7-1, `ml.m5.large`), **Batch Transform** (2 jobs) | Job completed in 156 billable seconds; evaluation vs the baseline |
| `03_model_registry` | **Model Registry** (package group, version with model metrics, approval), **Model Card** | Version 1 **Approved**; card `bankpulse-term-deposit-v1` |
| `04_inference_capture` | **Model**, **Endpoint configuration**, **Real-time endpoint**, **Data Capture**, `InvokeEndpoint` | 200 new customers scored; capture files in S3 |
| `05_bias_drift_monitoring` | **CloudWatch** metrics, alarm and dashboard; **Model Monitor** baseline; **Clarify** (attempted) | Monitoring report; 4 metrics, alarm and dashboard; clean-up cell |
| `06_continuous_training` | **SageMaker Pipelines**, **EventBridge** triggers, Model Registry version history | 4 successful retraining runs; versions 2-5 registered as PendingManualApproval |

## Features

Nine features are enabled (68 model inputs). None reads `duration`, and every learned value is fitted on the training split only and saved in `feature_schema.json`.

| Feature | Version | Inputs | Output | Learned on train |
|---|---|---|---|---|
| `previously_contacted` | v1 | `pdays` | 1 unless `pdays` is the 999 "never contacted" value | – |
| `prior_success` | v1 | `poutcome` | 1 if the previous campaign succeeded | – |
| `previous_contacts` | v1 | `previous` | Raw count | – |
| `campaign_intensity` | v1 | `campaign` | Raw count | – |
| `capped_campaign` | v1 (v2 available) | `campaign` | Count clipped at the 99th percentile (v2: then `log1p`) | Cap |
| `age` | v1 | `age` | Raw age | – |
| `age_band` | v1 | `age` | One-hot `<30`, `30-44`, `45-59`, `60+` | – |
| `categorical_profile` | v1 | `job`, `marital`, `education`, `default`, `housing`, `loan` | One-hot | Category lists |
| `contact_timing` | v1 | `contact`, `month`, `day_of_week` | One-hot | Category lists |

**Registered but not enabled:** `euribor_rate`, `employment_level` and `economic_indicators`.
- **Why:** they describe the *period* of the call, not the customer. With a time-ordered split, every later value is outside the training range (feature PSI 8–17), so XGBoost used them as a clock.
- **Effect:** validation PR-AUC was 0.177 with them and 0.229 without (`scripts/tune_xgboost.py`).

Other conventions:
- **Column names:** `<feature>__<category>`, or `<feature>__<column>__<category>` for features over several columns.
- **Unseen values:** a value never seen in training goes to `__other`, while `unknown` stays a normal category. For example, September appears only after the training period, so 570 test rows go to `contact_timing__month__other`.

## Model training and selection

- **Hyperparameters:** chosen by `scripts/tune_xgboost.py`, a 64-setting grid scored on **validation PR-AUC only**, using local XGBoost 1.7.6, the container's version. Best: `max_depth=4, eta=0.05, min_child_weight=1, colsample_bytree=0.5, subsample=0.8`, with early stopping on validation `aucpr`.
- **Threshold:** 0.106, the best F1 on validation, reused unchanged on test.
- **Approval rule:**
  - **Non-inferiority:** approve if test PR-AUC is at most 0.01 below the logistic baseline on the same features. The result was −0.0039, so the version was approved.
  - **Disclosure:** the margin was set after the first results, and the registry's approval description says so. A stricter "must beat the baseline" rule would have rejected this version.
- **Reports:** `reports/baseline_metrics.json`, `xgboost_tuning.json`, `xgboost_metrics.json`, `model_comparison.json`.

## Monitoring

`reports/monitoring_report.json`, built from the endpoint's captured traffic (the newest 200 customers) and compared with the training data:

| Metric | Value | Status |
|---|---|---|
| Max feature PSI | 3.43 (`previous_contacts`) | alert: 17 inputs above 0.2 |
| Prediction PSI | 2.95 (mean score 0.07 → 0.25) | alert |
| Share of customers above the threshold | 41% | |
| Age-band selection ratio, min/max | **0.30** | alert (below 0.8) |

The age-band selection rates at the threshold are: 60+ 77%, under 30 54%, 30–44 31%, 45–59 23%.
- **Possible cause:** this partly reflects real differences in subscription rates by age in this period.
- **Action:** it still needs a fairness review before the scores are used to decide who is called.
- **Caveat:** the live sample is small (200 customers).

**CloudWatch:**
- **Metrics:** the four headline values under `BankPulse/ModelMonitoring`, with the endpoint as the dimension.
- **Alarm:** `<endpoint>-feature-drift` fires when max PSI is above 0.2.
- **Dashboard:** `bankpulse-model-monitoring`.

**Managed SageMaker monitoring** (results in `reports/managed_monitoring_attempts.json`):
- Model Monitor baseline job: **completed**.
- Monitoring schedule and Clarify bias job: **rejected by AWS**, with "in maintenance mode and not available to new customers". The custom report above covers the same checks.

## Continuous training

New data keeps arriving after deployment, so the model retrains on recent data while the endpoint keeps serving the approved version. The design is in [docs/retraining.md](docs/retraining.md).

```
new month in raw/incoming/ ─┐
monthly schedule ───────────┼─> EventBridge ─> SageMaker Pipeline "bankpulse-retrain":
drift alarm (PSI > 0.2) ────┘     Prepare (ingest + rolling window) -> Train -> Evaluate (challenger vs champion)
                                  -> Check -> Register (PendingManualApproval) -> a person approves -> deploy
```

- **Ingest** (`bankpulse_core/ingest.py`):
  - each new file is checked against the data contract;
  - accepted files are copied byte for byte to `raw/batches/` and listed in `manifest.json`;
  - duplicates are skipped, and invalid files are reported in `raw/rejected/`;
  - nothing is deleted.
- **Rolling window** (`split.window_split`): train on the latest 12 months, validate on the next 2, hold out the newest 2. Older months age out.
- **Champion vs challenger** (`pipelines/steps/evaluate.py`): both models score the same holdout, each with its own feature schema and its own threshold from validation.
- **Rule:** register only if holdout PR-AUC is no more than 0.01 below the champion's, and the age-band selection ratio is not worse than the champion's or is at least 0.8.
- **Triggers** (`pipelines/triggers.py`): `bankpulse-retrain-on-upload`, `-monthly` and `-on-drift`, running as `LabRole`.
- **Safe deployment** (`pipelines/deploy.py`, not wired yet):
  - blue/green canary (10% for 10 minutes), with auto-rollback on 5XX and p99-latency alarms;
  - keeps Data Capture, and records the version's feature schema on the endpoint;
  - one-call manual rollback.

**Replay in the lab.** The history was split into its 26 calendar months. Months 1-23 were loaded as history, and months 24, 25 and 26 then arrived one at a time; each upload to `raw/incoming/` started the pipeline by itself.

| Execution | Trigger | Holdout (newest 2 months) | Champion (v1) PR-AUC | Challenger PR-AUC | Challenger lift | Age-band ratio | Result |
|---|---|---|---|---|---|---|---|
| `3wjs5sh014r7` | manual, month 24 | 536 rows | 0.588 | **0.751** | 1.76× | 0.83 | version 2 |
| `pyc11ujznp6j` | **S3 upload**, month 25 | 507 rows | 0.489 | **0.732** | 1.71× | 0.87 | version 3 |
| `akymeu81c6aa` | manual, `capped_campaign v2` | 507 rows | 0.489 | **0.732** | 1.71× | 0.87 | version 4 |
| `9ccdjy4g6i50` | **S3 upload**, month 26 | 332 rows | 0.474 | **0.660** | 1.53× | – | version 5 |

**What the replay showed:**
- **Retraining wins every time.** On every arrival, the model retrained on the latest 12 months beats the 2008-trained champion by 0.16-0.24 PR-AUC.
- **The v2 feature changes nothing.** `capped_campaign v2` (log1p) gives exactly the same results as v1, because tree splits don't change under a monotonic transform, so v1 stays.
- **Nothing is live yet.** Versions 2-5 wait for approval; the endpoint still serves version 1.

## Running it

### Locally (no AWS needed)

```bash
pip install -r requirements.txt
pip install -e .
pytest -q
```

- `python scripts/quality_report.py`: contract checks and the quality report.
- `python scripts/baseline.py`: runs the pipeline and the logistic baseline.
- `python scripts/tune_xgboost.py`: the validation-only hyperparameter search.

Outputs go to `data/processed/` (git-ignored) and `reports/`.

### Connecting to the AWS lab from a laptop

No credentials are stored in the repository. Each lab session:

1. Start the lab, open **AWS Details → AWS CLI: Show**, and paste the three values into `~/.aws/credentials` under `[bankpulse-lab]`. The file has no extension: in Notepad, choose "Save as type: All files".
2. Once only: put `[profile bankpulse-lab]` and `region = us-east-1` in `~/.aws/config`.
3. Check the connection. It prints no secrets:
   ```powershell
   .venv\Scripts\Activate.ps1
   python scripts\check_aws.py
   ```
   The code picks the `bankpulse-lab` profile automatically, and uses `LabRole` in the signed-in account. Outside Studio, `get_execution_role()` would return the sign-in role, which SageMaker cannot use. Set `BANKPULSE_ROLE_ARN` only to use another role.
4. Run notebooks 01 → 05 in order. Each saves what the next needs in `data/processed/sagemaker_state.json`, which is git-ignored because it holds ARNs.

The notebooks also run unchanged in SageMaker Studio, where Studio's execution role is used.

### Cost and clean-up

- **Billed only while running:** the training job and the two transform jobs.
- **Billed every hour until deleted:** the real-time endpoint (`ml.m5.large`, about $0.12/hour). Ending the lab session does **not** stop it.
- **To clean up:** in notebook 05, run the setup cell, set `CLEAN_UP = True` in the last cell, and run only that cell. It deletes the endpoint, its configuration and model (and a monitoring schedule, if one exists) and prints the endpoints still running.
- **Kept, costing nothing or almost nothing:** S3 objects, the registry, the Model Card and CloudWatch.

## Results UI

A Streamlit app for tracking the project in SageMaker and querying the endpoint:

```bash
streamlit run ui/app.py
```

Pick a section in the sidebar:

| Section | What it shows |
|---|---|
| Overview | What the project is, the problem, why it matters, who it is for, how it helps the end user, and what it must not be used for |
| SageMaker tracking | **Live, read-only** status from the lab: S3 data lake folders, training job (status, billable time, metrics, hyperparameters), Batch Transform jobs, Model Registry versions and approval notes, Model Card, endpoint and Data Capture, each with a link to the console |
| Model results | Baseline vs XGBoost per split, confusion matrices, hyperparameter search |
| Monitoring | Feature PSI chart, age-band selection rates, alerts, and the live CloudWatch alarm state |
| Score customers | One customer from a form (pre-call fields only), or the newest N customers, scored by the **live endpoint**. Shows probability, threshold decision, ranking and score distribution |

- **Offline (no lab credentials):** the app shows the saved reports.
- **Connected (`~/.aws/credentials` holds a valid `[bankpulse-lab]` profile):** the app also reads live status. It only describes and lists resources, never changes them. Scoring calls the endpoint while it is InService; those requests are recorded by Data Capture too.

## Screenshots

From the run in the AWS Academy lab. Click a thumbnail for the full image. All captions are in [docs/screenshots](docs/screenshots/README.md), and the executed notebooks with every output are in [`notebooks/executed/`](notebooks/executed/). A video walkthrough goes in [docs/video-demo](docs/video-demo/README.md).

**Notebook 01: S3 data lake**

<table>
<tr><td width="50%"><a href="docs/screenshots/01-sagemaker-session-role-bucket.png"><img src="docs/screenshots/01-sagemaker-session-role-bucket.png" width="100%" alt="01-sagemaker-session-role-bucket.png"></a><br><sub>Session, LabRole and bucket</sub></td><td width="50%"><a href="docs/screenshots/01-s3-data-lake-folders.png"><img src="docs/screenshots/01-s3-data-lake-folders.png" width="100%" alt="01-s3-data-lake-folders.png"></a><br><sub>S3 data lake folders</sub></td></tr>
<tr><td width="50%"><a href="docs/screenshots/01-s3-data-lake-folders-continued.png"><img src="docs/screenshots/01-s3-data-lake-folders-continued.png" width="100%" alt="01-s3-data-lake-folders-continued.png"></a><br><sub>S3 data lake folders (continued)</sub></td><td width="50%"><a href="docs/screenshots/01-s3-labels.png"><img src="docs/screenshots/01-s3-labels.png" width="100%" alt="01-s3-labels.png"></a><br><sub>Labels kept apart from features</sub></td></tr>
</table>

**Notebook 02: training job and Batch Transform**

<table>
<tr><td width="50%"><a href="docs/screenshots/02-training-job-completed.png"><img src="docs/screenshots/02-training-job-completed.png" width="100%" alt="02-training-job-completed.png"></a><br><sub>Training job Completed</sub></td><td width="50%"><a href="docs/screenshots/02-training-job-metrics.png"><img src="docs/screenshots/02-training-job-metrics.png" width="100%" alt="02-training-job-metrics.png"></a><br><sub>Training metrics (PR-AUC)</sub></td></tr>
<tr><td width="50%"><a href="docs/screenshots/02-training-job-input-channels.png"><img src="docs/screenshots/02-training-job-input-channels.png" width="100%" alt="02-training-job-input-channels.png"></a><br><sub>Train and validation channels</sub></td><td width="50%"><a href="docs/screenshots/02-training-job-output-artifact.png"><img src="docs/screenshots/02-training-job-output-artifact.png" width="100%" alt="02-training-job-output-artifact.png"></a><br><sub>model.tar.gz and LabRole</sub></td></tr>
<tr><td width="50%"><a href="docs/screenshots/02-s3-model-artifacts.png"><img src="docs/screenshots/02-s3-model-artifacts.png" width="100%" alt="02-s3-model-artifacts.png"></a><br><sub>Model artifacts in S3</sub></td><td width="50%"><a href="docs/screenshots/02-batch-transform-jobs.png"><img src="docs/screenshots/02-batch-transform-jobs.png" width="100%" alt="02-batch-transform-jobs.png"></a><br><sub>Batch Transform jobs Completed</sub></td></tr>
<tr><td width="50%"><a href="docs/screenshots/02-s3-batch-predictions.png"><img src="docs/screenshots/02-s3-batch-predictions.png" width="100%" alt="02-s3-batch-predictions.png"></a><br><sub>Batch predictions (.out)</sub></td></tr>
</table>

**Notebook 03: Model Registry and Model Card**

<table>
<tr><td width="50%"><a href="docs/screenshots/03-model-registry-group.png"><img src="docs/screenshots/03-model-registry-group.png" width="100%" alt="03-model-registry-group.png"></a><br><sub>Model Registry group</sub></td><td width="50%"><a href="docs/screenshots/03-model-registry-versions.png"><img src="docs/screenshots/03-model-registry-versions.png" width="100%" alt="03-model-registry-versions.png"></a><br><sub>Versions: v1 Approved, v2-v4 Pending</sub></td></tr>
<tr><td width="50%"><a href="docs/screenshots/03-model-card-overview.png"><img src="docs/screenshots/03-model-card-overview.png" width="100%" alt="03-model-card-overview.png"></a><br><sub>Model Card overview</sub></td><td width="50%"><a href="docs/screenshots/03-model-cards-list.png"><img src="docs/screenshots/03-model-cards-list.png" width="100%" alt="03-model-cards-list.png"></a><br><sub>Model Card, risk Medium</sub></td></tr>
</table>

**Notebook 04: real-time endpoint and Data Capture**

<table>
<tr><td width="50%"><a href="docs/screenshots/04-sagemaker-model-from-package.png"><img src="docs/screenshots/04-sagemaker-model-from-package.png" width="100%" alt="04-sagemaker-model-from-package.png"></a><br><sub>Model from package version 1</sub></td><td width="50%"><a href="docs/screenshots/04-endpoints-inservice.png"><img src="docs/screenshots/04-endpoints-inservice.png" width="100%" alt="04-endpoints-inservice.png"></a><br><sub>Endpoint InService</sub></td></tr>
<tr><td width="50%"><a href="docs/screenshots/04-endpoint-summary.png"><img src="docs/screenshots/04-endpoint-summary.png" width="100%" alt="04-endpoint-summary.png"></a><br><sub>Endpoint summary</sub></td><td width="50%"><a href="docs/screenshots/04-endpoint-config-data-capture.png"><img src="docs/screenshots/04-endpoint-config-data-capture.png" width="100%" alt="04-endpoint-config-data-capture.png"></a><br><sub>Data Capture active, 100%</sub></td></tr>
<tr><td width="50%"><a href="docs/screenshots/04-scored-customers.png"><img src="docs/screenshots/04-scored-customers.png" width="100%" alt="04-scored-customers.png"></a><br><sub>Scored customers</sub></td><td width="50%"><a href="docs/screenshots/04-s3-inference-capture.png"><img src="docs/screenshots/04-s3-inference-capture.png" width="100%" alt="04-s3-inference-capture.png"></a><br><sub>inference-capture/ in S3</sub></td></tr>
<tr><td width="50%"><a href="docs/screenshots/04-capture-jsonl-files.png"><img src="docs/screenshots/04-capture-jsonl-files.png" width="100%" alt="04-capture-jsonl-files.png"></a><br><sub>Captured requests (.jsonl)</sub></td></tr>
</table>

**Notebook 05: monitoring and CloudWatch**

<table>
<tr><td width="50%"><a href="docs/screenshots/05-s3-monitoring-report.png"><img src="docs/screenshots/05-s3-monitoring-report.png" width="100%" alt="05-s3-monitoring-report.png"></a><br><sub>Monitoring report in S3</sub></td><td width="50%"><a href="docs/screenshots/05-cloudwatch-alarms-overview.png"><img src="docs/screenshots/05-cloudwatch-alarms-overview.png" width="100%" alt="05-cloudwatch-alarms-overview.png"></a><br><sub>CloudWatch alarms overview</sub></td></tr>
<tr><td width="50%"><a href="docs/screenshots/05-cloudwatch-drift-alarm.png"><img src="docs/screenshots/05-cloudwatch-drift-alarm.png" width="100%" alt="05-cloudwatch-drift-alarm.png"></a><br><sub>Drift alarm: PSI 3.43 > 0.2</sub></td></tr>
</table>

**Notebook 06: continuous training**

<table>
<tr><td width="50%"><a href="docs/screenshots/06-eventbridge-retrain-rules.png"><img src="docs/screenshots/06-eventbridge-retrain-rules.png" width="100%" alt="06-eventbridge-retrain-rules.png"></a><br><sub>EventBridge retraining rules</sub></td><td width="50%"><a href="docs/screenshots/06-registry-version-2-lineage.png"><img src="docs/screenshots/06-registry-version-2-lineage.png" width="100%" alt="06-registry-version-2-lineage.png"></a><br><sub>Version 2 lineage</sub></td></tr>
<tr><td width="50%"><a href="docs/screenshots/06-registry-version-5-metrics.png"><img src="docs/screenshots/06-registry-version-5-metrics.png" width="100%" alt="06-registry-version-5-metrics.png"></a><br><sub>Version 5 metrics</sub></td><td width="50%"><a href="docs/screenshots/06-registry-version-5-approved.png"><img src="docs/screenshots/06-registry-version-5-approved.png" width="100%" alt="06-registry-version-5-approved.png"></a><br><sub>Version 5 approved</sub></td></tr>
<tr><td width="50%"><a href="docs/screenshots/06-registry-version-5-lineage.png"><img src="docs/screenshots/06-registry-version-5-lineage.png" width="100%" alt="06-registry-version-5-lineage.png"></a><br><sub>Version 5 lineage</sub></td><td width="50%"><a href="docs/screenshots/06-endpoint-unchanged-after-retraining.png"><img src="docs/screenshots/06-endpoint-unchanged-after-retraining.png" width="100%" alt="06-endpoint-unchanged-after-retraining.png"></a><br><sub>Endpoint unchanged after retraining</sub></td></tr>
</table>

**Results UI**

<table>
<tr><td width="50%"><a href="docs/screenshots/ui-overview.png"><img src="docs/screenshots/ui-overview.png" width="100%" alt="ui-overview.png"></a><br><sub>Results UI overview</sub></td></tr>
</table>

## Repository structure

```
bankpulse-sagemaker-mlops/
├── architecture/        make_diagram.py -> architecture.png
├── bankpulse_core/      FRAMEWORK: config, io, aws, contract, split, encoding, feature_base,
│                        pipeline, lineage, evaluation, monitoring, log_utils
├── features/            FEATURES: contact_history, campaign, economic, client_profile
├── config/              data_prep.yaml (data, split, S3 prefix, enabled features), contract.yaml
├── data/raw/            dataset and column descriptions (data/processed/ is generated, git-ignored)
├── notebooks/           01-06 without outputs; executed/ holds the same notebooks with the lab run's outputs
├── pipelines/           retrain_pipeline (SageMaker Pipeline), steps/ (prepare, evaluate), triggers, deploy
├── docs/                retraining.md: continuous-training design
├── reports/             quality, baseline, tuning, XGBoost, comparison, monitoring, managed-monitoring results
├── scripts/             check_aws, quality_report, baseline, tune_xgboost, simulate_arrivals
├── ui/                  Streamlit results UI: app, tracking (read-only AWS status), scoring, content
├── tests/               core/, features/, integration/, ui/, pipelines/ (266 tests)
├── pyproject.toml       pip install -e .
└── requirements.txt
```

### S3 layout

```
s3://sagemaker-us-east-1-<account>/bank-campaign-mlops/
├── raw/                 original CSV
├── train/  validation/  label-first XGBoost files (one per folder = one training channel)
├── batch-input/         feature-only files for Batch Transform
├── labels/              validation and test labels for evaluation
├── feature-schema/      column order and feature states, used by the endpoint
├── run-metadata/        lineage of the pipeline run
├── model-artifacts/     model.tar.gz from the training job
├── batch-predictions/   Batch Transform .out files
├── reports/             metrics, comparison, model-quality JSON for the registry
├── inference-capture/   Data Capture JSON Lines from the endpoint
├── model-monitor/       Model Monitor baseline statistics and constraints
└── monitoring-reports/  dated monitoring reports
```

## Limitations

- **Drift dominates.** The test period's subscription rate (38%) is about seven times training's (6%), so absolute scores don't transfer between periods. The model should be retrained on recent data, and the drift alarm is the trigger.
- **Modest ranking quality.** Test PR-AUC is about 0.47 against a 0.38 base rate, and lift is about 1.2× once drift is accounted for. The model helps prioritise calls; it doesn't predict individual outcomes.
- **Small live sample.** Monitoring used 200 captured customers. Its rates per age band are indicative only.
- **One bank, 2008–2010.** The data is from a single Portuguese bank during a financial crisis.
- **Lab limits.** Clarify and Model Monitor schedules are unavailable to new accounts, so bias and drift checks use the project's own code.
- **Deployment of a retrained version is a manual step.** Retraining, evaluation and registration run automatically. Switching the endpoint to an approved version (`pipelines/deploy.py`) is implemented and tested with fake clients, but not wired to approvals or run in the lab yet, so the endpoint keeps serving version 1.

## Responsible use

The model is meant for ranking call lists that campaign staff review. It is **not** for credit, lending, pricing or eligibility decisions (see the Model Card). Age and job are inputs, so selection rates by age band are monitored, and a ratio below 0.8 requires review.

## Reference

Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems, 62*, 22–31. https://doi.org/10.1016/j.dss.2014.03.001
