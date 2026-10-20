"""Smoke test: every section of the Streamlit app renders offline (no AWS credentials)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[2] / 'ui' / 'app.py'
SECTIONS = ['Overview', 'SageMaker tracking', 'Model results', 'Monitoring', 'Score customers']


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """Point boto3 at empty credential files, so the app can't reach AWS."""
    monkeypatch.delenv('AWS_PROFILE', raising=False)
    for name in ('AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('AWS_SHARED_CREDENTIALS_FILE', str(tmp_path / 'credentials'))
    monkeypatch.setenv('AWS_CONFIG_FILE', str(tmp_path / 'config'))
    monkeypatch.setenv('AWS_EC2_METADATA_DISABLED', 'true')


@pytest.mark.parametrize('section', SECTIONS)
def test_section_renders_offline(offline, section):
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.sidebar.radio(key='section').set_value(section).run()
    assert not app.exception, [e.value for e in app.exception]
    assert app.sidebar.info, 'the sidebar should explain the app is offline'


def test_overview_shows_project_story(offline):
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    subheaders = [s.value for s in app.subheader]
    for heading in ('The problem', 'Why it matters', 'How it helps the end user', 'Who it is for'):
        assert heading in subheaders
    assert any(m.label == 'Extra subscribers found vs baseline' and m.value == '+35%' for m in app.metric)
