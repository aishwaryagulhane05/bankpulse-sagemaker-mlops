# Continuous training: keeping the deployed model current

**Status:** design (Phase 11, issue BP-1101). It describes how new data reaches the model while an endpoint is serving, and how a retrained model safely replaces the deployed one. Tickets BP-1102 to BP-1108 implement it.

## Why the model must be retrained

The data shows how quickly a frozen model goes stale:

| Period | Subscription rate | What the deployed model sees |
|---|---|---|
| Training (oldest 70%) | 5.6% | what it learned |
| Validation (next 15%) | 10.7% | lift 2.7× |
| Test (newest 15%) | 38.5% | lift falls to 1.2× |
| Live traffic (newest 200) | 50% | 17 inputs drifted (PSI up to 3.4); prediction PSI 2.95 |

A model trained once in 2008 describes 2008. Customers, campaigns and the economy keep changing, so the model has to be retrained on recent data. The new version must be checked before it serves anyone, and it must be easy to undo.

## The loop at a glance

```
 new monthly file                monthly schedule            drift alarm (PSI > 0.2)
        |                               |                              |
        v                               v                              v
 s3://.../raw/incoming/ ------> EventBridge rule -------> start SageMaker Pipeline "bankpulse-retrain"
                                                                       |
   +-------------------------------------------------------------------+
   v
 1. Validate     contract checks on the new file -> accepted to raw/batches/ or moved to raw/rejected/
 2. Prepare      rolling window: train on the last K batches, validate on the next, hold out the newest
 3. Train        XGBoost training job (challenger), same container and tuned settings
 4. Evaluate     challenger AND current champion scored on the same newest holdout
 5. Decide       challenger PR-AUC >= champion - 0.01  and  age-band ratio not worse than champion's
                   no  -> stop, write the report, keep the champion
                   yes -> register the challenger in the Model Registry as PendingManualApproval
 6. Approve      a person reviews the metrics and approves it in the registry
                   |
                   v  (EventBridge: model package state change -> Approved)
 7. Deploy       new endpoint config; blue/green update: canary 10% of traffic for 10 minutes,
                 automatic rollback if the error or latency alarms fire; Data Capture stays on
 8. Re-baseline  monitoring reference = the new training window; drift is measured against it
```

The endpoint keeps serving the champion during steps 1–6. Customers only reach the new model in step 7, and only gradually.

## 1. When retraining starts (BP-1105)

| Trigger | How | Why |
|---|---|---|
| New data file | EventBridge rule on *S3 Object Created* under `bank-campaign-mlops/raw/incoming/` | Learn from the latest campaign as soon as its outcomes are known |
| Schedule | EventBridge schedule, monthly | Retrain even when no one uploads a file or drift is slow |
| Drift alarm | The existing CloudWatch alarm `<endpoint>-feature-drift` (max PSI > 0.2) | React when live customers no longer look like the training data |

All three start the same pipeline with the same parameters, so there is one retraining path, not three. An execution that is already running blocks a second start. The pipeline checks for running executions first, so repeated alarms don't stack up runs.

## 2. Ingesting a new file (BP-1102)

- **Landing:** a new file lands in `raw/incoming/`. The validate step runs the existing contract checks (`check_schema` and `check_rules` from `bankpulse_core/contract.py`).
- **Accepted:** the file is copied to `raw/batches/<arrival-id>.csv` and appended to `raw/batches/manifest.json` (arrival order, row count, SHA-256, positive rate).
- **Rejected:** the file goes to `raw/rejected/` with a JSON report of every problem. The pipeline stops, and no model changes.
- **Outcome labels:** files must contain outcome labels (`y`), so a batch arrives only after its calls have finished. Unlabelled live traffic is used for drift monitoring, never for training.

## 3. What the new model is trained on (BP-1103)

The training data is a **rolling window** over the batches, in arrival order:

```
batches:   b1 b2 b3 ... b12 b13 | b14       | b15
           <---- train -------> | validate  | holdout (newest)
```

- **Train:** the most recent K batches (config `window.train_batches`, e.g. 12 months). Older batches age out, so 2008 stops shaping 2011's scores.
- **Validate:** the next batch. Early stopping and the decision threshold are chosen here, as today.
- **Holdout:** the newest batch. It is used only to compare challenger and champion, which have both never seen it.
- **Features:** the same `config/data_prep.yaml` features and versions. Category lists and caps are re-learned from the new training window and saved in the new version's `feature_schema.json`.
- **Hyperparameters:** reused from `reports/xgboost_tuning.json`. Every few months (or if the challenger keeps losing), `scripts/tune_xgboost.py` runs again on the current window.

## 4. Champion vs challenger (BP-1104)

The deployed model (the **champion**) and the newly trained one (the **challenger**) are scored on the **same newest holdout batch**. Comparing each against its own test set would mix model quality with data drift.

The challenger is registered only if:
1. its holdout PR-AUC is at most 0.01 below the champion's (the same non-inferiority rule as notebook 03), and
2. its age-band selection ratio is not lower than the champion's, or is at least 0.8.

The pipeline writes a comparison report either way, and it is attached to the model package. A human approves in the registry. The pipeline never deploys on its own, because a person must be accountable for which model decides who gets called.

## 5. Each version carries its own encoding

A retrained model may learn different category lists or caps, so the endpoint must always encode customers with **the schema of the version it is serving**:

- **At registration:** the pipeline stores the S3 URI of that version's `feature_schema.json` in the model package's custom metadata (`feature_schema_uri`).
- **At deployment:** step 7 reads it and records it on the endpoint (tag `feature_schema_uri`).
- **Clients:** `encode_records`, the UI's scoring section and notebook 04 load the schema from the endpoint's tag instead of a fixed path. A version mismatch is then impossible.

## 6. Safe deployment and rollback (BP-1106)

When a version is approved, EventBridge (*SageMaker Model Package State Change*, status `Approved`) calls a small deploy function. It:

1. creates a SageMaker Model and a new endpoint configuration from the approved package, with Data Capture enabled;
2. calls `UpdateEndpoint` with a **blue/green deployment configuration**:
   - canary: 10% of traffic to the new version for 10 minutes, then all traffic;
   - **auto-rollback alarms:** `Invocation5XXErrors > 0` and `ModelLatency p99 > 500 ms`. If either fires during the canary, SageMaker moves all traffic back to the old fleet by itself.
3. keeps the previous endpoint configuration. **Manual rollback** is one call, `UpdateEndpoint(EndpointConfigName=<previous>)`, and the registry marks the replaced version accordingly.

The endpoint name never changes, so applications calling it need no change and no downtime.

## 7. After the switch

- **Monitoring reference:** the monitoring reference becomes the new training window, so PSI measures drift from what the *current* model learned.
- **Lineage:**
  - each pipeline execution writes `run_metadata.json` with the batch hashes in the window, the git commit and the feature versions;
  - the model package links to that execution, so any score can be traced back to the exact data it was trained on.
- **Version history in the UI:** the SageMaker tracking section lists every registered version with its holdout metrics and approval note (BP-1107).

## 8. Proving it works (BP-1107)

The historical file is split into time-ordered monthly batches. The last three are replayed as new arrivals, one at a time. Each arrival should:

- trigger the pipeline from the S3 upload, with nothing started by hand;
- register a challenger, or stop with a report explaining why not;
- after approval, update the endpoint through the canary, with no failed requests.

The result is a version history (v1, v2, v3, ...) with the data window, metrics and decision for each, visible in the registry and the UI.

## 9. Cost and lab limits (BP-1108)

**What a run costs.** Each execution runs three `ml.m5.large` jobs back to back, each billed only while it runs:

| Step | Time in the lab |
|---|---|
| Prepare | about 5 minutes (mostly instance start-up and `pip install`) |
| Train | about 3 minutes |
| Evaluate | about 4 minutes |

There is no Batch Transform step, because Evaluate scores both models directly with XGBoost. Nothing keeps running after the execution ends. The canary in step 7 briefly runs two endpoint fleets.

**Limits built in:**
- every job has `max_run` of 1 hour;
- the window caps the training data at 12 months;
- a run with nothing new in `incoming/` still retrains on the same window, so the monthly schedule is the only trigger that fires without new data.

**Clean-up (notebook 05, `CLEAN_UP = True`):**
- **deletes:** the monitoring schedule, endpoint, endpoint config and model, which are the things that bill by the hour;
- **disables, without deleting:** the three `bankpulse-retrain-*` rules, so no new executions start after the lab;
- **keeps:** the pipeline definition, registry versions and S3 data. They cost nothing, or close to nothing, and they are the project's record.

**When this runs for longer than a lab:**
- keep the last 3 registry versions and their endpoint configs;
- add an S3 lifecycle rule that moves `inference-capture/` and `retraining/executions/` objects older than 90 days to cheaper storage.

**AWS Academy:**
- Lab credentials expire, but EventBridge rules, the pipeline and the endpoint keep running under `LabRole`. The endpoint keeps billing after the lab session ends.
- `LabRole` trusts `events.amazonaws.com`, so EventBridge can start the pipeline as `LabRole` (checked in BP-1105).
- The processing container (XGBoost 1.7-1, Python 3.9, NumPy 1.24) needs `pandas==2.0.3`. Newer pandas wheels require NumPy 2 and fail to import.
- Clarify and Model Monitor schedules are unavailable to new accounts, so the evaluate and monitoring steps use this project's own code.

## 10. Ticket map

| Ticket | Delivers |
|---|---|
| BP-1102 | `bankpulse_core/ingest.py`: validate, accept or reject, manifest |
| BP-1103 | Rolling window split in `split.py`, `window` config |
| BP-1104 | `pipelines/retrain_pipeline.py`: the SageMaker Pipeline (steps 1–5), notebook 06 |
| BP-1105 | EventBridge rules: S3 upload, monthly schedule, drift alarm |
| BP-1106 | Deploy function: blue/green canary, auto-rollback, schema tag |
| BP-1107 | Replay of three monthly arrivals; version history in the UI |
| BP-1108 | Cost controls, clean-up, lab notes |
