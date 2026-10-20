"""Read-only status of the project's SageMaker resources, for the tracking tab.

Every function takes boto3 clients and returns plain dicts. Only describe/list calls are
made, so opening the UI never creates, changes or deletes anything in AWS. A failure
(expired lab credentials, a deleted endpoint) is returned as {'error': ...} instead of
raising, so one missing resource doesn't break the whole page.
"""

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


def safe(call: Callable[[], Any]) -> Any:
    """Run a read-only AWS call; return {'error': message} if it fails."""
    try:
        return call()
    except Exception as error:                    # botocore ClientError, NoCredentialsError, ...
        return {'error': f'{type(error).__name__}: {error}'}


def _minutes(start: Optional[datetime], end: Optional[datetime]) -> Optional[float]:
    return round((end - start).total_seconds() / 60, 1) if start and end else None


def data_lake(s3, bucket: str, prefix: str) -> Dict[str, Dict[str, Any]]:
    """Objects and total size per top-level folder under s3://bucket/prefix/."""
    folders: Dict[str, Dict[str, Any]] = {}
    for page in s3.get_paginator('list_objects_v2').paginate(Bucket=bucket, Prefix=f'{prefix}/'):
        for item in page.get('Contents', []):
            folder = item['Key'][len(prefix) + 1:].split('/', 1)[0]
            entry = folders.setdefault(folder, {'objects': 0, 'size_mb': 0.0, 'last_modified': None})
            entry['objects'] += 1
            entry['size_mb'] += item['Size'] / 1e6
            if entry['last_modified'] is None or item['LastModified'] > entry['last_modified']:
                entry['last_modified'] = item['LastModified']
    for entry in folders.values():
        entry['size_mb'] = round(entry['size_mb'], 2)
    return folders


def training_job(sm, name: str) -> Dict[str, Any]:
    job = sm.describe_training_job(TrainingJobName=name)
    return {
        'name': name,
        'status': job['TrainingJobStatus'],
        'instance': job['ResourceConfig']['InstanceType'],
        'billable_seconds': job.get('BillableTimeInSeconds'),
        'minutes': _minutes(job.get('TrainingStartTime'), job.get('TrainingEndTime')),
        'metrics': {m['MetricName']: round(m['Value'], 4) for m in job.get('FinalMetricDataList', [])},
        'hyperparameters': {k: v for k, v in job.get('HyperParameters', {}).items() if not k.startswith('_')},
        'model_artifact': job.get('ModelArtifacts', {}).get('S3ModelArtifacts'),
    }


def transform_jobs(sm, names: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """split -> status of its Batch Transform job."""
    result = {}
    for split, name in names.items():
        job = sm.describe_transform_job(TransformJobName=name)
        result[split] = {'name': name, 'status': job['TransformJobStatus'],
                         'instance': job['TransformResources']['InstanceType'],
                         'minutes': _minutes(job.get('TransformStartTime'), job.get('TransformEndTime'))}
    return result


def registry(sm, group: str) -> List[Dict[str, Any]]:
    """Every version in the model package group, newest first."""
    packages = sm.list_model_packages(ModelPackageGroupName=group, SortBy='CreationTime', SortOrder='Descending')
    versions = []
    for summary in packages['ModelPackageSummaryList']:
        detail = sm.describe_model_package(ModelPackageName=summary['ModelPackageArn'])
        versions.append({'version': detail['ModelPackageVersion'], 'status': detail['ModelApprovalStatus'],
                         'created': detail['CreationTime'], 'description': detail.get('ModelPackageDescription', ''),
                         'approval_note': detail.get('ApprovalDescription', ''),
                         'metadata': detail.get('CustomerMetadataProperties', {})})
    return versions


def model_card(sm, name: str) -> Dict[str, Any]:
    card = sm.describe_model_card(ModelCardName=name)
    return {'name': name, 'status': card['ModelCardStatus'], 'version': card['ModelCardVersion'],
            'last_modified': card.get('LastModifiedTime')}


def endpoint(sm, name: str) -> Dict[str, Any]:
    """Endpoint status and Data Capture settings; a deleted endpoint reports status 'Deleted'."""
    try:
        detail = sm.describe_endpoint(EndpointName=name)
    except Exception as error:
        if 'Could not find endpoint' in str(error):
            return {'name': name, 'status': 'Deleted'}
        raise
    capture = detail.get('DataCaptureConfig', {})
    return {'name': name, 'status': detail['EndpointStatus'], 'config': detail['EndpointConfigName'],
            'created': detail['CreationTime'], 'data_capture': capture.get('EnableCapture', False),
            'capture_percent': capture.get('CurrentSamplingPercentage'),
            'instance': [v.get('CurrentInstanceCount') for v in detail.get('ProductionVariants', [])]}


def alarm(cw, name: str) -> Dict[str, Any]:
    alarms = cw.describe_alarms(AlarmNames=[name])['MetricAlarms']
    if not alarms:
        return {'name': name, 'state': 'missing'}
    item = alarms[0]
    return {'name': name, 'state': item['StateValue'], 'reason': item.get('StateReason', ''),
            'threshold': item['Threshold'], 'metric': item['MetricName']}


def latest_metrics(cw, namespace: str, endpoint_name: str, names: List[str],
                   start: datetime, end: datetime) -> Dict[str, Optional[float]]:
    """The most recent value of each custom metric in [start, end] (None if no data point)."""
    queries = [{'Id': f'm{i}', 'Label': name, 'ReturnData': True,
                'MetricStat': {'Metric': {'Namespace': namespace, 'MetricName': name,
                                          'Dimensions': [{'Name': 'EndpointName', 'Value': endpoint_name}]},
                               'Period': 300, 'Stat': 'Maximum'}}
               for i, name in enumerate(names)]
    response = cw.get_metric_data(MetricDataQueries=queries, StartTime=start, EndTime=end, ScanBy='TimestampDescending')
    return {item['Label']: (round(item['Values'][0], 4) if item['Values'] else None)
            for item in response['MetricDataResults']}
