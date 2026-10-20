"""Tests for ui.tracking with fake AWS clients (no AWS access)."""

from datetime import datetime, timedelta, timezone

from ui import tracking

T0 = datetime(2026, 10, 20, 19, 38, tzinfo=timezone.utc)


class FakeSageMaker:
    def describe_training_job(self, TrainingJobName):
        return {'TrainingJobStatus': 'Completed', 'ResourceConfig': {'InstanceType': 'ml.m5.large'},
                'BillableTimeInSeconds': 156, 'TrainingStartTime': T0, 'TrainingEndTime': T0 + timedelta(minutes=3),
                'FinalMetricDataList': [{'MetricName': 'validation:aucpr', 'Value': 0.22591}],
                'HyperParameters': {'max_depth': '4', '_tuning_objective_metric': ''},
                'ModelArtifacts': {'S3ModelArtifacts': 's3://b/model.tar.gz'}}

    def describe_transform_job(self, TransformJobName):
        return {'TransformJobStatus': 'Completed', 'TransformResources': {'InstanceType': 'ml.m5.large'},
                'TransformStartTime': T0, 'TransformEndTime': T0 + timedelta(minutes=4, seconds=30)}

    def list_model_packages(self, **kwargs):
        return {'ModelPackageSummaryList': [{'ModelPackageArn': 'arn:pkg/1'}]}

    def describe_model_package(self, ModelPackageName):
        return {'ModelPackageVersion': 1, 'ModelApprovalStatus': 'Approved', 'CreationTime': T0,
                'ApprovalDescription': 'Approved under a non-inferiority rule'}

    def describe_endpoint(self, EndpointName):
        if EndpointName == 'gone':
            raise RuntimeError(f'Could not find endpoint "{EndpointName}".')
        return {'EndpointStatus': 'InService', 'EndpointConfigName': 'cfg', 'CreationTime': T0,
                'DataCaptureConfig': {'EnableCapture': True, 'CurrentSamplingPercentage': 100},
                'ProductionVariants': [{'CurrentInstanceCount': 1}]}


class FakePaginator:
    def paginate(self, Bucket, Prefix):
        yield {'Contents': [
            {'Key': f'{Prefix}train/train.csv', 'Size': 4_000_000, 'LastModified': T0},
            {'Key': f'{Prefix}batch-input/test_features.csv', 'Size': 1_000_000, 'LastModified': T0},
            {'Key': f'{Prefix}batch-input/validation_features.csv', 'Size': 1_000_000,
             'LastModified': T0 + timedelta(hours=1)},
        ]}


class FakeS3:
    def get_paginator(self, name):
        return FakePaginator()


class FakeCloudWatch:
    def describe_alarms(self, AlarmNames):
        return {'MetricAlarms': [{'StateValue': 'ALARM', 'StateReason': 'PSI 3.4 > 0.2', 'Threshold': 0.2,
                                  'MetricName': 'max_feature_psi'}]}

    def get_metric_data(self, MetricDataQueries, **kwargs):
        return {'MetricDataResults': [{'Label': q['Label'], 'Values': [3.42691] if i == 0 else []}
                                      for i, q in enumerate(MetricDataQueries)]}


def test_data_lake_groups_objects_by_folder():
    lake = tracking.data_lake(FakeS3(), 'bucket', 'bank-campaign-mlops')
    assert lake['batch-input']['objects'] == 2 and lake['batch-input']['size_mb'] == 2.0
    assert lake['batch-input']['last_modified'] == T0 + timedelta(hours=1)
    assert set(lake) == {'train', 'batch-input'}


def test_training_job_summary():
    job = tracking.training_job(FakeSageMaker(), 'job-1')
    assert job['status'] == 'Completed' and job['minutes'] == 3.0
    assert job['metrics'] == {'validation:aucpr': 0.2259}
    assert job['hyperparameters'] == {'max_depth': '4'}            # internal "_" parameters hidden


def test_transform_and_registry_and_endpoint():
    sm = FakeSageMaker()
    assert tracking.transform_jobs(sm, {'test': 't-1'})['test']['minutes'] == 4.5
    assert tracking.registry(sm, 'group')[0]['status'] == 'Approved'
    endpoint = tracking.endpoint(sm, 'live')
    assert endpoint['status'] == 'InService' and endpoint['data_capture'] and endpoint['capture_percent'] == 100


def test_deleted_endpoint_is_reported_not_raised():
    assert tracking.endpoint(FakeSageMaker(), 'gone') == {'name': 'gone', 'status': 'Deleted'}


def test_alarm_and_latest_metrics():
    cw = FakeCloudWatch()
    assert tracking.alarm(cw, 'a')['state'] == 'ALARM'
    values = tracking.latest_metrics(cw, 'ns', 'ep', ['max_feature_psi', 'prediction_psi'], T0, T0)
    assert values == {'max_feature_psi': 3.4269, 'prediction_psi': None}


def test_safe_turns_errors_into_a_message():
    def expired():
        raise PermissionError('ExpiredToken')

    assert tracking.safe(expired) == {'error': 'PermissionError: ExpiredToken'}
    assert tracking.safe(lambda: 42) == 42
