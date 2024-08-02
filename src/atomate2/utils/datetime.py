"""Helper functions for datetime objects."""

from __future__ import annotations

from datetime import datetime, timezone


def datetime_str() -> str:
    """
    Get a string representation of the current time.

    Returns
    -------
    str
        The current time.
    """
    return str(datetime.now(tz=timezone.utc))
