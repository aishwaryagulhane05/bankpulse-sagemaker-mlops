# Video demo: BankPulse app

A short screen recording of the BankPulse results UI and the SageMaker resources behind it. Back to the [README](../../README.md).

**Video:** _not recorded yet._ Add the file here as `bankpulse-demo.mp4`, or link it below if it's hosted elsewhere.

<!-- When recorded, replace the line above with, for example:
**Video:** [bankpulse-demo.mp4](bankpulse-demo.mp4) (5 min)
or a YouTube / Google Drive link.
-->

## Before recording

1. Start the lab, paste fresh credentials into `~/.aws/credentials`, and run `python scripts\check_aws.py` (all checks must pass).
2. Check the endpoint is **InService**. If it was cleaned up, redeploy it with notebook 04.
3. Start the app: `streamlit run ui/app.py`, then open http://localhost:8501 (or the port it prints).
4. Click **Refresh live status** in the sidebar, so the app reads the current lab session.
5. Record the browser window at 1080p. Hide bookmarks and close unrelated tabs.

## Storyboard (about 5 minutes)

| Time | Show | Say |
|---|---|---|
| 0:00 | **Overview** section | The problem (1 in 9 calls ends in a subscription), who uses this (campaign managers, agents, the MLOps team, risk and compliance), and the key numbers: +35% subscribers found, 1.24× lift. |
| 0:45 | Architecture diagram (Overview, bottom) | The local framework prepares the data; SageMaker trains, registers, serves and monitors; CloudWatch alerts. |
| 1:15 | **SageMaker tracking** | Live from the lab: data lake folders, training job Completed, Batch Transform, registry versions (v1 Approved, retrained versions), Model Card, endpoint InService with Data Capture. |
| 2:00 | Tracking → **6. Continuous training** | Three EventBridge triggers, and the pipeline runs. Each new month retrains on the latest 12 months and registers a challenger for approval. |
| 2:40 | **Model results** | XGBoost vs the logistic baseline on the same features: ties on PR-AUC, more subscribers found (F1 0.37 vs 0.31). Thresholds are chosen on validation only. |
| 3:15 | **Monitoring** | Drift between live traffic and the training data (17 inputs above PSI 0.2), the age-band selection ratio of 0.30 (flagged for review), and the CloudWatch alarm. |
| 3:50 | **Score customers**: one customer | Fill in the form (pre-call fields only), score with the live endpoint, and show the probability and call-first decision. |
| 4:20 | **Score customers**: newest 200 | Score the batch: 82 flagged, 60% of them subscribed vs 50% overall. Show the score distribution. |
| 4:50 | Close | Responsible use: ranks marketing calls only, not credit decisions. Retraining is approval-gated. |

## Optional extra clips

- SageMaker Studio → Pipelines → `bankpulse-retrain`: one execution graph, all steps green.
- Uploading a month file to `raw/incoming/`, then the new pipeline execution appearing by itself.
- The CloudWatch dashboard `bankpulse-model-monitoring`.

## File size

GitHub rejects files over 100 MB and warns above 50 MB. Keep the MP4 under 50 MB (5 minutes at 1080p with H.264 is usually 20–40 MB), or host it (YouTube, unlisted; or Google Drive) and link it here.
