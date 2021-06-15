"""Helper functions for datetime objects."""

from datetime import datetime


def datetime_str():
    """Get a string representation of the current time."""
    return str(datetime.utcnow())
