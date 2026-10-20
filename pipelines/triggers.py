"""EventBridge triggers that start the retraining pipeline (docs/retraining.md, section 1).

    python -m pipelines.triggers            # create or update the three rules

    bankpulse-retrain-on-upload   S3 "Object Created" under <prefix>/raw/incoming/  -> start pipeline
    bankpulse-retrain-monthly     schedule: 03:00 UTC on the 1st of every month      -> start pipeline
    bankpulse-retrain-on-drift    the feature-drift alarm enters ALARM                -> start pipeline

All three run as LabRole (which trusts events.amazonaws.com). Turning on EventBridge events
for the bucket keeps any notification settings the bucket already has.
"""

import json
from pathlib import Path
from typing import Dict, List

from bankpulse_core.aws import lab_boto_session, read_state
from bankpulse_core.config import load_config

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_NAME = 'bankpulse-retrain'


def rules(bucket: str, prefix: str, alarm: str) -> List[Dict]:
    """The three rule definitions (name, description, and an event pattern or a schedule)."""
    return [
        {'Name': 'bankpulse-retrain-on-upload',
         'Description': 'A new monthly data file landed in raw/incoming/: retrain',
         'EventPattern': json.dumps({'source': ['aws.s3'], 'detail-type': ['Object Created'],
                                     'detail': {'bucket': {'name': [bucket]},
                                                'object': {'key': [{'prefix': f'{prefix}/raw/incoming/'}]}}})},
        {'Name': 'bankpulse-retrain-monthly',
         'Description': 'Monthly retraining, even without new uploads or drift',
         'ScheduleExpression': 'cron(0 3 1 * ? *)'},
        {'Name': 'bankpulse-retrain-on-drift',
         'Description': 'Live data drifted from the training data (max PSI > 0.2): retrain',
         'EventPattern': json.dumps({'source': ['aws.cloudwatch'], 'detail-type': ['CloudWatch Alarm State Change'],
                                     'detail': {'alarmName': [alarm], 'state': {'value': ['ALARM']}}})},
    ]


def enable_bucket_events(s3, bucket: str) -> None:
    """Send the bucket's S3 events to EventBridge, keeping any existing notification settings."""
    current = s3.get_bucket_notification_configuration(Bucket=bucket)
    current.pop('ResponseMetadata', None)
    if 'EventBridgeConfiguration' not in current:
        s3.put_bucket_notification_configuration(Bucket=bucket,
                                                 NotificationConfiguration={**current, 'EventBridgeConfiguration': {}})


def install(events, s3, pipeline_arn: str, role: str, bucket: str, prefix: str, alarm: str) -> List[str]:
    """Create or update the rules and point each at the pipeline; returns the rule names."""
    enable_bucket_events(s3, bucket)
    names = []
    for rule in rules(bucket, prefix, alarm):
        events.put_rule(State='ENABLED', **rule)
        events.put_targets(Rule=rule['Name'], Targets=[{
            'Id': 'retrain-pipeline', 'Arn': pipeline_arn, 'RoleArn': role,
            'SageMakerPipelineParameters': {'PipelineParameterList': []},     # pipeline defaults
        }])
        names.append(rule['Name'])
    return names


def main() -> None:
    config = load_config(ROOT / 'config' / 'data_prep.yaml')
    state = read_state(Path(config['output']['processed_dir']) / 'sagemaker_state.json')
    session = lab_boto_session()
    pipeline_arn = session.client('sagemaker').describe_pipeline(PipelineName=PIPELINE_NAME)['PipelineArn']
    names = install(session.client('events'), session.client('s3'), pipeline_arn, state['role'], state['bucket'],
                    state['prefix'], state['cloudwatch']['alarm'])
    print('rules ->', PIPELINE_NAME, ':', ', '.join(names))


if __name__ == '__main__':
    main()
