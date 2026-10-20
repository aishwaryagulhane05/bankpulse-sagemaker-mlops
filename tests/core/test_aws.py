"""Tests for bankpulse_core.aws (no AWS access: sessions are faked)."""

import pytest

from bankpulse_core import aws
from bankpulse_core.aws import lab_boto_session, lab_role, read_state, s3_uri, update_state


class FakeSession:
    def __init__(self, profile_name=None, profiles=()):
        self.profile_name, self.available_profiles = profile_name, list(profiles)

    def client(self, name):
        assert name == 'sts'
        return type('Sts', (), {'get_caller_identity': lambda self: {'Account': '123456789012'}})()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv('BANKPULSE_ROLE_ARN', raising=False)
    monkeypatch.delenv('AWS_PROFILE', raising=False)
    monkeypatch.setattr(aws, 'in_studio', lambda: False)


def test_profile_is_used_when_it_exists(monkeypatch):
    monkeypatch.setattr(aws.boto3, 'Session', lambda profile_name=None: FakeSession(profile_name, ['bankpulse-lab']))
    session = lab_boto_session()
    assert session.profile_name == 'bankpulse-lab'
    assert aws.os.environ['AWS_PROFILE'] == 'bankpulse-lab'      # shared with io.py's S3 client


def test_default_chain_when_profile_is_missing(monkeypatch):
    monkeypatch.setattr(aws.boto3, 'Session', lambda profile_name=None: FakeSession(profile_name, []))
    assert lab_boto_session().profile_name is None
    assert 'AWS_PROFILE' not in aws.os.environ


def test_role_is_lab_role_in_signed_in_account():
    assert lab_role(FakeSession()) == 'arn:aws:iam::123456789012:role/LabRole'


def test_role_from_environment_wins(monkeypatch):
    monkeypatch.setenv('BANKPULSE_ROLE_ARN', 'arn:aws:iam::111111111111:role/Custom')
    assert lab_role(FakeSession()) == 'arn:aws:iam::111111111111:role/Custom'


def test_s3_uri_joins_parts():
    assert s3_uri('bucket', 'bank-campaign-mlops/', '/train', 'train.csv') == 's3://bucket/bank-campaign-mlops/train/train.csv'
    assert s3_uri('bucket', 'prefix', '') == 's3://bucket/prefix'


def test_state_is_merged_between_notebooks(tmp_path):
    path = tmp_path / 'sagemaker_state.json'
    assert read_state(path) == {}
    update_state(path, bucket='b', prefix='p')
    assert update_state(path, training_job='job-1') == {'bucket': 'b', 'prefix': 'p', 'training_job': 'job-1'}
