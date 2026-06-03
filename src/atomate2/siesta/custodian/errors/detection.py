"""Error detection functions for SIESTA calculations."""

from __future__ import annotations

import logging
from pathlib import Path

from atomate2.siesta.custodian.errors.base import ErrorPattern, ErrorType
from atomate2.siesta.custodian.errors.patterns import SIESTA_ERROR_PATTERNS

logger = logging.getLogger(__name__)


def detect_error(directory: Path | str) -> list[ErrorPattern]:
    """Detect errors in SIESTA calculation output.

    Parameters
    ----------
    directory : Path or str
        Directory containing SIESTA output files

    Returns
    -------
    list of ErrorPattern
        List of detected error patterns (empty if no errors)

    Example
    -------
    >>> errors = detect_error("job_001")
    >>> if errors:
    ...     print(f"Found {len(errors)} errors")
    ...     for error in errors:
    ...         print(f"  {error.error_type.value}: {error.description}")
    """
    directory = Path(directory)
    detected_errors = []

    for error_pattern in SIESTA_ERROR_PATTERNS:
        file_path = directory / error_pattern.file_to_check

        # Skip if file doesn't exist
        if not file_path.exists():
            continue

        try:
            # Read file content
            with open(file_path, "r") as f:
                content = f.read()

            # Check for pattern matches
            if error_pattern.matches(content):
                detected_errors.append(error_pattern)

        except Exception as e:
            # Log but don't fail on file read errors
            logger.warning(f"Could not read {file_path}: {e}")

    return detected_errors


def get_error_type(directory: Path | str) -> ErrorType | None:
    """Get the first detected error type.

    Parameters
    ----------
    directory : Path or str
        Directory containing SIESTA output files

    Returns
    -------
    ErrorType or None
        The first error type detected, or None if no errors

    Example
    -------
    >>> error_type = get_error_type("job_001")
    >>> if error_type:
    ...     print(f"Error: {error_type.value}")
    """
    errors = detect_error(directory)
    return errors[0].error_type if errors else None


def check_for_errors(directory: Path | str) -> bool:
    """Check if any errors occurred in calculation.

    Parameters
    ----------
    directory : Path or str
        Directory containing SIESTA output files

    Returns
    -------
    bool
        True if errors were detected, False otherwise

    Example
    -------
    >>> if check_for_errors("job_001"):
    ...     print("Errors found!")
    """
    return len(detect_error(directory)) > 0
