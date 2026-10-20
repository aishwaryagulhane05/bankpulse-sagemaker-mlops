"""Safe deployment of an approved model version, with automatic and manual rollback (BP-1106).

    deploy_approved(sm, cw, package_arn, endpoint_name, role)   # blue/green update of the live endpoint
    rollback(sm, endpoint_name, previous_config)                 # one call back to the old version
    handler(event, context)                                      # Lambda entry for the EventBridge
                                                                 #   "Model Package State Change" event

The endpoint name never changes, so callers need no change and see no downtime. The previous
endpoint configuration is kept (never deleted) so a rollback is always possible.

Not wired up yet: connecting `handler` to approvals would switch the live endpoint, so it is
installed only when you choose to (docs/retraining.md, section 6).
"""

import os
import time
from typing import Dict, List

VARIANT = 'AllTraffic'
LATENCY_LIMIT_MICROSECONDS = 500_000          # ModelLatency is reported in microseconds


def _name(*parts: str) -> str:
    """A SageMaker resource name: letters, digits and hyphens, at most 63 characters."""
    return '-'.join(parts)[:63].rstrip('-')


def rollback_alarms(cw, endpoint_name: str) -> List[str]:
    """Alarms that make SageMaker roll the update back by itself: any 5XX error, or slow responses."""
    dimensions = [{'Name': 'EndpointName', 'Value': endpoint_name}, {'Name': 'VariantName', 'Value': VARIANT}]
    alarms = [
        {'AlarmName': f'{endpoint_name}-5xx-errors', 'MetricName': 'Invocation5XXErrors', 'Statistic': 'Sum',
         'Threshold': 0.0},
        {'AlarmName': f'{endpoint_name}-latency-p99', 'MetricName': 'ModelLatency', 'ExtendedStatistic': 'p99',
         'Threshold': float(LATENCY_LIMIT_MICROSECONDS)},
    ]
    for alarm in alarms:
        cw.put_metric_alarm(Namespace='AWS/SageMaker', Dimensions=dimensions, Period=60, EvaluationPeriods=1,
                            ComparisonOperator='GreaterThanThreshold', TreatMissingData='notBreaching', **alarm)
    return [alarm['AlarmName'] for alarm in alarms]


def deploy_approved(sm, cw, package_arn: str, endpoint_name: str, role: str,
                    instance_type: str = 'ml.m5.large') -> Dict:
    """Serve an approved model package from the existing endpoint, via a blue/green canary."""
    # Step 1: only an approved version may serve customers.
    package = sm.describe_model_package(ModelPackageName=package_arn)
    if package['ModelApprovalStatus'] != 'Approved':
        raise ValueError(f'{package_arn} is {package["ModelApprovalStatus"]}, not Approved')
    version, stamp = str(package['ModelPackageVersion']), time.strftime('%Y%m%d-%H%M%S')

    # Step 2: a SageMaker Model for that version.
    model_name = _name('bankpulse', f'v{version}', stamp)
    sm.create_model(ModelName=model_name, ExecutionRoleArn=role, PrimaryContainer={'ModelPackageName': package_arn})

    # Step 3: a new endpoint config, keeping the current Data Capture settings so monitoring continues.
    current = sm.describe_endpoint(EndpointName=endpoint_name)
    previous_config = current['EndpointConfigName']
    capture = sm.describe_endpoint_config(EndpointConfigName=previous_config).get('DataCaptureConfig')
    new_config = _name('bankpulse', f'v{version}', stamp)
    sm.create_endpoint_config(
        EndpointConfigName=new_config,
        ProductionVariants=[{'VariantName': VARIANT, 'ModelName': model_name, 'InitialInstanceCount': 1,
                             'InstanceType': instance_type}],
        **({'DataCaptureConfig': capture} if capture else {}))

    # Step 4: blue/green update: 10% of capacity first for 10 minutes, then everything; SageMaker
    # rolls back by itself if the error or latency alarm fires during the update.
    alarms = rollback_alarms(cw, endpoint_name)
    sm.update_endpoint(
        EndpointName=endpoint_name, EndpointConfigName=new_config,
        DeploymentConfig={
            'BlueGreenUpdatePolicy': {
                'TrafficRoutingConfiguration': {'Type': 'CANARY', 'WaitIntervalInSeconds': 600,
                                                'CanarySize': {'Type': 'CAPACITY_PERCENT', 'Value': 10}},
                'TerminationWaitInSeconds': 300,
                'MaximumExecutionTimeoutInSeconds': 3600,
            },
            'AutoRollbackConfiguration': {'Alarms': [{'AlarmName': name} for name in alarms]},
        })

    # Step 5: record which encoding the endpoint now needs (the version's own feature schema).
    schema_uri = package.get('CustomerMetadataProperties', {}).get('feature_schema_uri')
    if schema_uri:
        sm.add_tags(ResourceArn=current['EndpointArn'], Tags=[{'Key': 'feature_schema_uri', 'Value': schema_uri}])
    return {'endpoint': endpoint_name, 'version': version, 'model': model_name, 'endpoint_config': new_config,
            'previous_config': previous_config, 'rollback_alarms': alarms, 'feature_schema_uri': schema_uri}


def rollback(sm, endpoint_name: str, previous_config: str) -> None:
    """Manual rollback: serve the previous endpoint configuration again (it was never deleted)."""
    sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=previous_config)


def handler(event, context=None, sm=None, cw=None):
    """Lambda entry point for EventBridge 'SageMaker Model Package State Change' events."""
    detail = event.get('detail', {})
    if detail.get('ModelApprovalStatus') != 'Approved' or detail.get('ModelPackageGroupName') != os.environ['GROUP']:
        return {'skipped': True, 'reason': 'not an approval in this model group'}
    if sm is None or cw is None:
        import boto3

        sm, cw = boto3.client('sagemaker'), boto3.client('cloudwatch')
    return deploy_approved(sm, cw, detail['ModelPackageArn'], os.environ['ENDPOINT_NAME'], os.environ['ROLE_ARN'])
