"""Tests for pipelines.deploy and pipelines.triggers with fake AWS clients (nothing is called in AWS)."""

import json

import pytest

from pipelines import deploy, triggers


class Recorder:
    """Fake boto3 client: records every call and returns canned responses."""

    def __init__(self, responses=None):
        self.calls, self.responses = [], responses or {}

    def __getattr__(self, name):
        def call(**kwargs):
            self.calls.append((name, kwargs))
            response = self.responses.get(name, {})
            return response(**kwargs) if callable(response) else response
        return call

    def names(self):
        return [name for name, _ in self.calls]


def fake_sagemaker(status='Approved'):
    return Recorder({
        'describe_model_package': {'ModelApprovalStatus': status, 'ModelPackageVersion': 2,
                                   'CustomerMetadataProperties': {'feature_schema_uri': 's3://b/v2/schema.json'}},
        'describe_endpoint': {'EndpointConfigName': 'config-v1', 'EndpointArn': 'arn:endpoint/live'},
        'describe_endpoint_config': {'DataCaptureConfig': {'EnableCapture': True, 'InitialSamplingPercentage': 100}},
    })


def test_deploy_runs_a_blue_green_canary_with_auto_rollback():
    sm, cw = fake_sagemaker(), Recorder()
    result = deploy.deploy_approved(sm, cw, 'arn:package/2', 'live', 'arn:role/LabRole')

    assert sm.names() == ['describe_model_package', 'create_model', 'describe_endpoint', 'describe_endpoint_config',
                          'create_endpoint_config', 'update_endpoint', 'add_tags']
    config = dict(sm.calls)['create_endpoint_config']
    assert config['DataCaptureConfig']['EnableCapture'] is True                    # monitoring continues
    update = dict(sm.calls)['update_endpoint']
    assert update['EndpointName'] == 'live'                                        # same name: no client change
    policy = update['DeploymentConfig']['BlueGreenUpdatePolicy']['TrafficRoutingConfiguration']
    assert policy['Type'] == 'CANARY' and policy['CanarySize']['Value'] == 10
    assert [a['AlarmName'] for a in update['DeploymentConfig']['AutoRollbackConfiguration']['Alarms']] == \
        ['live-5xx-errors', 'live-latency-p99']
    assert result['previous_config'] == 'config-v1' and result['feature_schema_uri'] == 's3://b/v2/schema.json'
    assert 'delete' not in ' '.join(sm.names())                                    # nothing is ever deleted


def test_unapproved_version_is_never_deployed():
    sm = fake_sagemaker(status='PendingManualApproval')
    with pytest.raises(ValueError, match='not Approved'):
        deploy.deploy_approved(sm, Recorder(), 'arn:package/2', 'live', 'arn:role/LabRole')
    assert sm.names() == ['describe_model_package']


def test_rollback_serves_the_previous_config():
    sm = Recorder()
    deploy.rollback(sm, 'live', 'config-v1')
    assert sm.calls == [('update_endpoint', {'EndpointName': 'live', 'EndpointConfigName': 'config-v1'})]


def test_handler_ignores_other_events(monkeypatch):
    monkeypatch.setenv('GROUP', 'bankpulse-term-deposit')
    event = {'detail': {'ModelApprovalStatus': 'Rejected', 'ModelPackageGroupName': 'bankpulse-term-deposit'}}
    assert deploy.handler(event, sm=Recorder(), cw=Recorder())['skipped'] is True


def test_trigger_rules_cover_upload_schedule_and_drift():
    rules = {r['Name']: r for r in triggers.rules('bucket', 'bank-campaign-mlops', 'live-feature-drift')}
    upload = json.loads(rules['bankpulse-retrain-on-upload']['EventPattern'])
    assert upload['detail']['object']['key'] == [{'prefix': 'bank-campaign-mlops/raw/incoming/'}]
    assert rules['bankpulse-retrain-monthly']['ScheduleExpression'] == 'cron(0 3 1 * ? *)'
    drift = json.loads(rules['bankpulse-retrain-on-drift']['EventPattern'])
    assert drift['detail'] == {'alarmName': ['live-feature-drift'], 'state': {'value': ['ALARM']}}


def test_bucket_events_keep_existing_notifications():
    s3 = Recorder({'get_bucket_notification_configuration': lambda **kw: {
        'QueueConfigurations': [{'QueueArn': 'arn:sqs'}], 'ResponseMetadata': {}}})
    triggers.enable_bucket_events(s3, 'bucket')
    _, put = s3.calls[1]
    assert put['NotificationConfiguration'] == {'QueueConfigurations': [{'QueueArn': 'arn:sqs'}],
                                                'EventBridgeConfiguration': {}}


def test_install_points_every_rule_at_the_pipeline():
    events = Recorder()
    s3 = Recorder({'get_bucket_notification_configuration': lambda **kw: {'EventBridgeConfiguration': {}}})
    names = triggers.install(events, s3, 'arn:pipeline/bankpulse-retrain', 'arn:role/LabRole', 'bucket', 'p', 'alarm')
    assert len(names) == 3 and 'put_bucket_notification_configuration' not in s3.names()   # already enabled
    targets = [kwargs['Targets'][0] for name, kwargs in events.calls if name == 'put_targets']
    assert {t['Arn'] for t in targets} == {'arn:pipeline/bankpulse-retrain'}
    assert all(t['RoleArn'] == 'arn:role/LabRole' for t in targets)
