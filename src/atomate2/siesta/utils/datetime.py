"""Helper functions for datetime objects."""  # noqa: A005

from __future__ import annotations

import logging
from datetime import datetime, timezone


logger = logging.getLogger(__name__)


def datetime_str() -> str:
    """
    Get a string representation of the current time.

    Returns
    -------
    str
        The current time.
    """
    logger.info("datetime_str()")
    return str(datetime.now(tz=timezone.utc))
