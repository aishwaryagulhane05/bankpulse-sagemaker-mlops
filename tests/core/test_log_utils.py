"""Tests for bankpulse_core.log_utils."""

import io
import logging

import pytest

from bankpulse_core.log_utils import PACKAGES, log_stage, setup_logging


@pytest.fixture
def stream():
    """Capture bankpulse log output; restore the package loggers afterwards."""
    saved = {name: (logging.getLogger(name).handlers[:], logging.getLogger(name).level,
                    logging.getLogger(name).propagate) for name in PACKAGES}
    buffer = io.StringIO()
    setup_logging('INFO', stream=buffer)
    yield buffer
    for name, (handlers, level, propagate) in saved.items():
        logger = logging.getLogger(name)
        logger.handlers, logger.level, logger.propagate = handlers, level, propagate


def test_stage_logs_start_and_duration(stream):
    with log_stage('split'):
        pass
    lines = stream.getvalue().splitlines()
    assert lines[0].endswith('bankpulse_core.stages: start: split')
    assert 'done: split in ' in lines[1] and lines[1].endswith('s')


def test_failed_stage_is_logged_and_error_re_raised(stream):
    with pytest.raises(ValueError, match='bad input'):
        with log_stage('validate'):
            raise ValueError('bad input')
    assert 'ERROR bankpulse_core.stages: failed: validate after ' in stream.getvalue()


def test_setup_twice_does_not_duplicate_lines(stream):
    setup_logging('INFO', stream=stream)
    logging.getLogger('features.campaign').info('hello')
    assert stream.getvalue().count('hello') == 1


def test_level_filters_messages(stream):
    setup_logging('WARNING', stream=stream)
    logging.getLogger('bankpulse_core.io').info('hidden')
    logging.getLogger('bankpulse_core.io').warning('shown')
    assert 'hidden' not in stream.getvalue() and 'shown' in stream.getvalue()
