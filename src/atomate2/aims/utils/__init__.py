"""A collection of helper utils found in atomate2 package."""

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
