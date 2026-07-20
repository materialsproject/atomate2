"""
Module defining base SIESTA Atomate2 VerbosityLevel set and generator.

"""

# import logging
# logger = logging.getLogger(__name__)
# from atomate2.siesta.utils.common import logger
from enum import Enum


class VerbosityLevel(Enum):
    SILENT = 0  # None Color
    INFO = 1  # Green
    ERROR = 2  # Red
    WARNING = 3  # Yellow
    DEBUG = 4  # Blue
    VERBOSE = 5

    # SILENT = 0    # None Color
    # ERROR = 1     # Red
    # WARNING = 2   # Yellow
    # INFO = 3      # Green
    # DEBUG = 4     # Blue


def get_verbosity_value(verbosity) -> int:
    """
    Safely get the verbosity value, handling both VerbosityLevel enum and int/dict types.

    This is needed because jobflow serialization can convert the enum to int or dict.

    Args:
        verbosity: The verbosity value (VerbosityLevel enum, int, or dict)

    Returns
    -------
        int: The verbosity level value
    """
    # If it's already a VerbosityLevel enum, get its value
    if hasattr(verbosity, "value"):
        return verbosity.value

    # If it's a dict (from serialization), try to get the value
    if isinstance(verbosity, dict):
        return verbosity.get("value", VerbosityLevel.ERROR.value)

    # If it's already an int, return it
    if isinstance(verbosity, int):
        return verbosity

    # Fallback to ERROR level
    return VerbosityLevel.ERROR.value
