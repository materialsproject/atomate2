"""Base error types and classes for SIESTA calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Pattern


class ErrorType(Enum):
    """Types of errors that can occur in SIESTA calculations."""

    SCF_CONVERGENCE = "scf_convergence"
    MEMORY = "memory"
    TIME_LIMIT = "time_limit"
    NUMERICAL = "numerical"
    BASIS = "basis"
    GRID = "grid"
    FILE_IO = "file_io"
    SEGFAULT = "segmentation_fault"
    GEOMETRY = "geometry"
    PSEUDOPOTENTIAL = "pseudopotential"
    PARALLEL = "parallelization"
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """Severity levels for errors."""

    RECOVERABLE = "recoverable"  # Can be fixed automatically
    WARNING = "warning"  # Non-critical, may not need fixing
    CRITICAL = "critical"  # Requires intervention or major changes
    FATAL = "fatal"  # Cannot recover


@dataclass
class ErrorPattern:
    """Define an error pattern for detection.

    Parameters
    ----------
    error_type : ErrorType
        Type of error this pattern detects
    patterns : list of Pattern
        Compiled regex patterns to search for
    file_to_check : str
        Which file to check for this pattern (e.g., 'siesta.out', 'slurm.out')
    severity : ErrorSeverity
        Severity level of this error
    description : str, optional
        Human-readable description of the error
    """

    error_type: ErrorType
    patterns: list[Pattern] = field(default_factory=list)
    file_to_check: str = "siesta.out"
    severity: ErrorSeverity = ErrorSeverity.RECOVERABLE
    description: str = ""

    def matches(self, content: str) -> bool:
        """Check if content matches any of the patterns.

        Parameters
        ----------
        content : str
            File content to search

        Returns
        -------
        bool
            True if any pattern matches
        """
        for pattern in self.patterns:
            if pattern.search(content):
                return True
        return False


class SiestaError(Exception):
    """Base exception for SIESTA calculation errors."""

    def __init__(self, error_type: ErrorType, message: str = ""):
        """Initialize SiestaError.

        Parameters
        ----------
        error_type : ErrorType
            Type of error
        message : str, optional
            Error message
        """
        self.error_type = error_type
        self.message = message or f"SIESTA error: {error_type.value}"
        super().__init__(self.message)


class RecoverableError(SiestaError):
    """Error that can potentially be recovered from."""

    pass


class UnrecoverableError(SiestaError):
    """Error that cannot be recovered from automatically."""

    pass


class ValidationError(SiestaError):
    """Error during output validation."""

    pass


class MaxErrorsError(SiestaError):
    """Maximum number of error correction attempts reached."""

    pass
