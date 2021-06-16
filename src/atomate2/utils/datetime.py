"""Helper functions for datetime objects."""

from datetime import datetime

__all__ = ["datetime_str"]


def datetime_str():
    """Get a string representation of the current time."""
    return str(datetime.utcnow())
