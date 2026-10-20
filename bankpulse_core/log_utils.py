"""Logging: shared logging setup for all entry points (notebooks, tests, processing jobs).

    setup_logging()        one log format for the bankpulse_core and features packages
    log_stage("split")     context manager that logs when a step starts, ends and how long it took

Modules log through logging.getLogger(__name__), so their messages land under the
"bankpulse_core" or "features" loggers configured here.
"""

import logging
import sys
import time
from contextlib import contextmanager
from typing import IO, Iterator, Optional

# The packages whose loggers this module configures.
PACKAGES = ('bankpulse_core', 'features')

LOG_FORMAT = '%(asctime)s %(levelname)s %(name)s: %(message)s'


def setup_logging(level: str = 'INFO', stream: Optional[IO[str]] = None) -> None:
    """Send bankpulse log messages to stream (default: stdout) in one format.

    Safe to call more than once, e.g. when a notebook cell is re-run: the old handler is
    replaced, so lines are never printed twice.
    """
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    for package in PACKAGES:
        logger = logging.getLogger(package)
        # Step 1: drop handlers from an earlier call, which would duplicate every line.
        for old in list(logger.handlers):
            logger.removeHandler(old)
        # Step 2: attach the new handler and level.
        logger.addHandler(handler)
        logger.setLevel(level)
        # Step 3: don't pass messages to the root logger too; Jupyter and SageMaker often
        # configure root, which would print each line a second time in another format.
        logger.propagate = False


@contextmanager
def log_stage(name: str, logger: Optional[logging.Logger] = None) -> Iterator[None]:
    """Log the start, end and duration of a pipeline step; a failure is logged and re-raised."""
    logger = logger or logging.getLogger('bankpulse_core.stages')
    logger.info('start: %s', name)
    started = time.perf_counter()
    try:
        yield
    except Exception:
        # Log which step failed and after how long, then let the error stop the run.
        logger.error('failed: %s after %.2fs', name, time.perf_counter() - started)
        raise
    logger.info('done: %s in %.2fs', name, time.perf_counter() - started)
