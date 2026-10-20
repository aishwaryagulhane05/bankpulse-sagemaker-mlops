"""Check that this machine can reach the AWS lab, without printing any secret.

Run in PowerShell from the repository folder, after pasting the lab credentials into
~/.aws/credentials under [bankpulse-lab] (see issue BP-400):

    $env:AWS_PROFILE = "bankpulse-lab"
    $env:BANKPULSE_ROLE_ARN = "arn:aws:iam::<account-id>:role/LabRole"
    .venv\\Scripts\\python scripts\\check_aws.py

Checks, in order: credentials -> region -> LabRole ARN -> SageMaker access -> default bucket.
Note: the last check creates the SageMaker default bucket (sagemaker-<region>-<account>)
if it doesn't exist yet. That bucket is where the later steps store data and models.
"""

import os
import re
import sys

import boto3
from botocore.exceptions import ClientError, NoCredentialsError, ProfileNotFound

ROLE_PATTERN = re.compile(r'^arn:aws:iam::\d{12}:role/\S+$')
EXPIRED = ('ExpiredToken', 'ExpiredTokenException', 'InvalidClientTokenId', 'UnrecognizedClientException')


def fail(message: str) -> None:
    print(f'FAILED: {message}')
    sys.exit(1)


def main() -> None:
    # Step 1: credentials. boto3 reads the profile named in AWS_PROFILE; keys are never printed.
    profile = os.environ.get('AWS_PROFILE')
    try:
        session = boto3.Session()
        identity = session.client('sts').get_caller_identity()
    except ProfileNotFound:
        fail(f'profile {profile!r} not found in ~/.aws/credentials or ~/.aws/config')
    except NoCredentialsError:
        fail('no credentials found: set $env:AWS_PROFILE = "bankpulse-lab" in this terminal')
    except ClientError as exc:
        if exc.response['Error']['Code'] in EXPIRED:
            fail('credentials expired or invalid: start the lab and paste fresh values from AWS Details')
        raise
    print(f'OK  credentials: account {identity["Account"]}, signed in as {identity["Arn"].split("/")[-2]}')

    # Step 2: region, from ~/.aws/config.
    if not session.region_name:
        fail('no region: add "region = us-east-1" under [profile bankpulse-lab] in ~/.aws/config')
    print(f'OK  region: {session.region_name}')

    # Step 3: LabRole ARN. Outside Studio, get_execution_role() doesn't work, so the role
    # SageMaker jobs run as comes from this environment variable.
    role = os.environ.get('BANKPULSE_ROLE_ARN', '')
    if not ROLE_PATTERN.match(role):
        fail('set $env:BANKPULSE_ROLE_ARN to the LabRole ARN from IAM -> Roles -> LabRole')
    if role.split(':')[4] != identity['Account']:
        fail('BANKPULSE_ROLE_ARN belongs to a different account than the credentials')
    print(f'OK  role: {role.split("/")[-1]}')

    # Step 4: SageMaker is reachable with these credentials (read-only call).
    try:
        session.client('sagemaker').list_training_jobs(MaxResults=1)
    except ClientError as exc:
        fail(f'SageMaker access denied: {exc.response["Error"]["Code"]}')
    print('OK  SageMaker: reachable')

    # Step 5: the default bucket (created on first use).
    import sagemaker

    bucket = sagemaker.Session(boto_session=session).default_bucket()
    print(f'OK  default bucket: {bucket}')
    print('All checks passed.')


if __name__ == '__main__':
    main()
