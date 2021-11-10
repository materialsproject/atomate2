"""Helper functions for datetime objects."""

from datetime import datetime

__all__ = ["datetime_str"]


def datetime_str() -> str:
    """
    Get a string representation of the current time.

    Returns
    -------
    str
        The current time.
    """
    return str(datetime.utcnow())
