"""AWS lab connection: one session, the SageMaker role, S3 URIs and state shared between notebooks.

    boto_session = lab_boto_session()          # laptop: the bankpulse-lab profile; Studio: its role
    role = lab_role(boto_session)              # arn:aws:iam::<account>:role/LabRole
    uri = s3_uri(bucket, 'bank-campaign-mlops', 'train', 'train.csv')
    update_state(state_path, model_data=...)   # hand results from notebook 02 to notebook 03

No credentials are handled here: boto3 reads them from ~/.aws (laptop) or the Studio role.
"""

import os
from pathlib import Path
from typing import Any, Dict, Union

import boto3

from bankpulse_core import io as bankpulse_io

DEFAULT_PROFILE = 'bankpulse-lab'
# AWS Academy labs give SageMaker this role; jobs and endpoints run as it.
LAB_ROLE_NAME = 'LabRole'
# Present only inside SageMaker Studio / notebook instances.
_STUDIO_METADATA = Path('/opt/ml/metadata/resource-metadata.json')


def in_studio() -> bool:
    return _STUDIO_METADATA.exists()


def lab_boto_session(profile: str = DEFAULT_PROFILE) -> boto3.Session:
    """A boto3 session for the lab: the named profile on a laptop, the default chain elsewhere."""
    if profile in boto3.Session().available_profiles:
        # Every boto3 client created later in this process (including io.py's S3 client)
        # should use the same profile, so set it for the whole process.
        os.environ['AWS_PROFILE'] = profile
        bankpulse_io._s3.cache_clear()
        return boto3.Session(profile_name=profile)
    return boto3.Session()


def lab_role(boto_session: boto3.Session) -> str:
    """ARN of the role SageMaker jobs and endpoints run as.

    Order: BANKPULSE_ROLE_ARN if set; the execution role inside Studio; otherwise LabRole in
    the signed-in account. (Outside Studio, sagemaker.get_execution_role() would return the
    lab's sign-in role, voclabs, which SageMaker cannot use.)
    """
    if os.environ.get('BANKPULSE_ROLE_ARN'):
        return os.environ['BANKPULSE_ROLE_ARN']
    if in_studio():
        import sagemaker

        return sagemaker.get_execution_role(sagemaker_session=sagemaker.Session(boto_session=boto_session))
    account = boto_session.client('sts').get_caller_identity()['Account']
    return f'arn:aws:iam::{account}:role/{LAB_ROLE_NAME}'


def s3_uri(bucket: str, *parts: str) -> str:
    """'s3://bucket/a/b/c' from parts, ignoring stray slashes."""
    keys = [part.strip('/') for part in parts if part and part.strip('/')]
    return 's3://' + '/'.join([bucket, *keys])


def read_state(path: Union[str, Path]) -> Dict[str, Any]:
    """Values saved by earlier notebooks (bucket, job names, model package ARN, ...)."""
    path = Path(path)
    return bankpulse_io.read_json(path) if path.exists() else {}


def update_state(path: Union[str, Path], **values: Any) -> Dict[str, Any]:
    """Merge values into the state file and return the full state.

    The file lives in data/processed/ (git-ignored), because it holds account-specific
    ARNs and bucket names that don't belong in a public repository.
    """
    state = {**read_state(path), **values}
    bankpulse_io.write_json(state, path)
    return state
