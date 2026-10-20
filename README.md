# bankpulse-sagemaker-mlops

An end-to-end MLOps system that predicts customer term deposit subscriptions using XGBoost, with AWS SageMaker for training, model registry, batch and real-time inference, and Amazon S3 and CloudWatch for monitoring.

> **Status:** Phase 2 (feature engine) complete. The time-ordered splitter, encoding helpers, feature interface and registry, and 12 versioned features are implemented and tested. The other framework modules and the notebooks are still labelled placeholders; the SageMaker work is being built step by step in AWS SageMaker Studio.

## Problem

A bank runs phone campaigns to sell term deposits, and only about 11% of contacted customers subscribe. The goal is to predict, **before a call is made**, the probability that a customer will subscribe, so the bank can call the most promising customers first.

- **Type:** binary classification (target `y`: yes / no)
- **Key constraint:** only information known before the call may be used, so `duration` (call length) is excluded to prevent target leakage.

## Dataset

[UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) (Moro, Cortez & Rita, 2014), `bank-additional-full.csv`.

| Property | Value |
|---|---|
| Records | 41,188 phone contacts, May 2008 – November 2010 |
| Inputs | 20 (client, campaign, previous contact and economic indicators) |
| Target | `y`, subscribed to a term deposit (about 11% yes) |
| Format | `;`-separated CSV, rows in date order |
| Location | `data/raw/bank-additional-full.csv` |
| Column descriptions | `data/raw/bank-additional-names.txt` |

Notes: `pdays = 999` means the client was not previously contacted, and several categorical columns use the value `unknown`.

## Planned architecture

```
Amazon S3 (data lake)
  raw → processed → chronological train / validation / test
        │
SageMaker Studio (JupyterLab): preprocessing, leakage removal, encoding
        │
SageMaker Training Job (built-in XGBoost) → model artifact in S3
        │
Batch Transform → evaluation (PR-AUC, precision, recall, F1)
        │
Model Registry + Model Card (versioning, approval, documentation)
        │
Real-time endpoint ──→ Data capture → S3
        │
Drift and bias monitoring → S3 reports → CloudWatch metrics and alarms
```

## SageMaker and AWS components

| Component | Purpose | Notebook |
|---|---|---|
| Amazon S3 | Data lake for raw data, splits, model artifacts, predictions, captures and reports | 01 |
| SageMaker Studio / JupyterLab | Development environment | All |
| SageMaker Training Job (XGBoost) | Managed model training | 02 |
| SageMaker Batch Transform | Score validation and test sets | 02 |
| SageMaker Model Registry | Model versioning and approval | 03 |
| SageMaker Model Cards | Intended use, training and evaluation documentation | 03 |
| SageMaker Models, Endpoint Configurations, Real-Time Endpoints | Online predictions | 04 |
| SageMaker Data Capture | Log requests and predictions to S3 | 04 |
| SageMaker Clarify / Model Bias Monitor | Managed bias analysis (may be blocked in the AWS Academy lab) | 05 |
| Amazon CloudWatch | Custom monitoring metrics and alarms | 05 |
| IAM (LabRole) | Permissions for all SageMaker resources | All |

## Repository structure

```
bankpulse-sagemaker-mlops/
├── data/raw/                     Dataset and column descriptions
├── notebooks/
│   ├── 01_data_preparation.ipynb       S3 data lake, leakage removal, encoding, chronological split
│   ├── 02_training_evaluation.ipynb    XGBoost training job and Batch Transform evaluation
│   ├── 03_model_registry.ipynb         Model Registry and Model Card
│   ├── 04_inference_capture.ipynb      Real-time endpoint with data capture
│   └── 05_bias_drift_monitoring.ipynb  Drift, bias and CloudWatch monitoring
├── bankpulse_core/               FRAMEWORK: generic, reusable, no business logic
│   ├── config.py                 Load and check settings
│   ├── io.py                     Local and S3 read/write, file hashes
│   ├── contract.py               Data contract and validation rules
│   ├── split.py                  Time-based splits
│   ├── encoding.py               Encoding, capping, unseen categories
│   ├── feature_base.py           Feature interface and registry
│   ├── pipeline.py               Config-driven pipeline runner
│   ├── lineage.py                Run metadata and versions
│   ├── evaluation.py             Batch Transform parsing, threshold, metrics
│   ├── monitoring.py             Drift and subgroup checks
│   └── log_utils.py              Shared logging setup
├── features/                     FEATURES: business logic, one module per group
│   ├── contact_history.py        previously_contacted, prior_success, previous_contacts
│   ├── campaign.py               campaign_intensity, capped_campaign
│   ├── economic.py               euribor_rate, employment_level, economic_indicators
│   └── client_profile.py         age, age_band, categorical_profile, contact_timing
├── config/
│   ├── data_prep.yaml            Data source, splits, enabled features and versions
│   └── contract.yaml             Allowed/forbidden columns and rules
├── tests/
│   ├── core/                     Framework tests
│   └── features/                 One test file per feature module
├── reports/                      Monitoring reports and evaluation results
├── architecture/                 Architecture diagram
├── pyproject.toml                Installable package: pip install -e .
└── requirements.txt
```

### Framework vs features

- **`bankpulse_core/`** holds reusable functions shared by every feature and notebook. It never imports from `features/`.
- **`features/`** holds feature definitions. Each feature has a name, version, required inputs and owner, and uses the core for I/O, validation, encoding and fitting on training data only.
- **`config/data_prep.yaml`** decides which features and versions run, so adding or upgrading a feature means adding a module and switching it on in config, not editing the pipeline.
- **Notebooks** only call the framework (`pipeline.run()`); they hold no reusable logic.

To make both packages importable in notebooks and tests, run `pip install -e .` from the repository root.

## Features

All 12 features are enabled at v1 in `config/data_prep.yaml`. None of them reads `duration`, and every learned value (caps, category lists) is fitted on the training split only and saved as JSON.

| Feature | Version | Inputs | Output | Learns from training |
|---|---|---|---|---|
| `previously_contacted` | v1 | `pdays` | 1 if `pdays` is not the 999 "never contacted" value | – |
| `prior_success` | v1 | `poutcome` | 1 if the previous campaign succeeded | – |
| `previous_contacts` | v1 | `previous` | Raw count | – |
| `campaign_intensity` | v1 | `campaign` | Raw count | – |
| `capped_campaign` | v1 | `campaign` | Count clipped at the 99th percentile | Cap |
| `euribor_rate` | v1 | `euribor3m` | Raw value (time-dependent, watch for drift) | – |
| `employment_level` | v1 | `nr.employed` | Raw value (time-dependent, watch for drift) | – |
| `economic_indicators` | v1 | `emp.var.rate`, `cons.price.idx`, `cons.conf.idx` | One column each (time-dependent, watch for drift) | – |
| `age` | v1 | `age` | Raw age | – |
| `age_band` | v1 | `age` | One-hot `<30`, `30-44`, `45-59`, `60+` | – |
| `categorical_profile` | v1 | `job`, `marital`, `education`, `default`, `housing`, `loan` | One-hot | Category lists |
| `contact_timing` | v1 | `contact`, `month`, `day_of_week` | One-hot | Category lists |

- **Column names:** single-column features use the feature name. Multi-column outputs use `<feature>__<category>`, or `<feature>__<column>__<category>` when several raw columns are encoded.
- **Unseen values:** a category not seen in training goes to that column's `__other` column, while the dataset's own `unknown` stays a normal category. For example, `month = sep` appears only after the training period.
- **Output:** fitted on the training split, the 12 features give 73 columns.

## Running the tests

```bash
pip install -r requirements.txt
pip install -e .
pytest -v
```

Tests run locally before each merge; there is no hosted CI.

## Planned S3 layout

```
s3://<sagemaker-bucket>/bank-campaign-mlops/
├── raw/  processed/  train/  validation/  test/
├── model-artifacts/  feature-schema/  batch-predictions/
└── inference-capture/  monitoring-reports/
```

## Environment

- **Runs in:** notebooks run locally in VS Code and create SageMaker resources in the AWS Academy Learner Lab, using the lab's `LabRole`. They also run unchanged in SageMaker Studio.
- **Region:** `us-east-1`.
- **Instance type:** `ml.m5.large` for training, Batch Transform and the endpoint.
- **Python packages:** listed in `requirements.txt` (`pip install -r requirements.txt`, then `pip install -e .`).

### Connecting to the AWS lab from a laptop

No credentials are ever stored in the repository. Each lab session:

1. Start the lab, open **AWS Details → AWS CLI: Show**, and paste the three values into `~/.aws/credentials` under `[bankpulse-lab]`. The file has no extension: in Notepad, choose "Save as type: All files".
2. Once only: put `[profile bankpulse-lab]` and `region = us-east-1` in `~/.aws/config`.
3. In PowerShell, from the repository folder:

   ```powershell
   .venv\Scripts\Activate.ps1
   $env:AWS_PROFILE = "bankpulse-lab"
   $env:BANKPULSE_ROLE_ARN = "arn:aws:iam::<account-id>:role/LabRole"   # IAM -> Roles -> LabRole
   $env:SAGEMAKER_SUPPRESS_V2_WARNING = "1"
   python scripts\check_aws.py
   ```

   The script checks the credentials, region, role, SageMaker access and default bucket without printing any secret. Outside Studio, `sagemaker.get_execution_role()` does not work, so notebooks read the role from `BANKPULSE_ROLE_ARN`.
4. Lab credentials expire when the session ends. Paste fresh values next time, and click **End Lab** when finished.
- **Cost:** delete the endpoint, endpoint configuration and model when finished, because real-time endpoints are billed every hour they run.

## Evaluation approach

The classes are imbalanced (about 11% positive), so accuracy alone is misleading. The model will be evaluated with PR-AUC, precision, recall, F1-score and a confusion matrix. The decision threshold is chosen on the validation set, and the final results are reported once on the held-out test set.

## Reference

Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems, 62*, 22–31. https://doi.org/10.1016/j.dss.2014.03.001
