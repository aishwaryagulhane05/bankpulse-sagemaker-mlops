"""Check that this machine can reach the AWS lab, without printing any secret.

Run from the repository folder, after pasting the lab credentials into
~/.aws/credentials under [bankpulse-lab] (see issue BP-400):

    .venv\\Scripts\\python scripts\\check_aws.py

The bankpulse-lab profile is picked automatically, and the role is LabRole in the signed-in
account (set BANKPULSE_ROLE_ARN only to use a different role).

Checks, in order: credentials -> region -> role -> SageMaker access -> default bucket.
Note: the last check creates the SageMaker default bucket (sagemaker-<region>-<account>)
if it doesn't exist yet. That bucket is where the later steps store data and models.
"""

import re
import sys

from botocore.exceptions import ClientError, NoCredentialsError, ProfileNotFound

from bankpulse_core.aws import lab_boto_session, lab_role

ROLE_PATTERN = re.compile(r'^arn:aws:iam::\d{12}:role/\S+$')
EXPIRED = ('ExpiredToken', 'ExpiredTokenException', 'InvalidClientTokenId', 'UnrecognizedClientException')


def fail(message: str) -> None:
    print(f'FAILED: {message}')
    sys.exit(1)


def main() -> None:
    # Step 1: credentials, from the bankpulse-lab profile; keys are never printed.
    try:
        session = lab_boto_session()
        identity = session.client('sts').get_caller_identity()
    except ProfileNotFound as exc:
        fail(f'{exc} (check ~/.aws/credentials and ~/.aws/config, without a .txt extension)')
    except NoCredentialsError:
        fail('no credentials found: paste the lab values into ~/.aws/credentials under [bankpulse-lab]')
    except ClientError as exc:
        if exc.response['Error']['Code'] in EXPIRED:
            fail('credentials expired or invalid: start the lab and paste fresh values from AWS Details')
        raise
    print(f'OK  credentials: account {identity["Account"]}, signed in as {identity["Arn"].split("/")[-2]}')

    # Step 2: region, from ~/.aws/config.
    if not session.region_name:
        fail('no region: add "region = us-east-1" under [profile bankpulse-lab] in ~/.aws/config')
    print(f'OK  region: {session.region_name}')

    # Step 3: the role SageMaker jobs and endpoints run as.
    role = lab_role(session)
    if not ROLE_PATTERN.match(role):
        fail(f'BANKPULSE_ROLE_ARN is not a role ARN: {role!r}')
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
