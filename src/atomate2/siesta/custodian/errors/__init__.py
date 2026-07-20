"""Error patterns and definitions for SIESTA calculations.

This package provides error type definitions, detection patterns,
and error detection functions for the custodian system.
"""

from atomate2.siesta.custodian.errors.base import (
    ErrorPattern,
    ErrorSeverity,
    ErrorType,
    MaxErrorsError,
    RecoverableError,
    SiestaError,
    UnrecoverableError,
    ValidationError,
)
from atomate2.siesta.custodian.errors.detection import (
    check_for_errors,
    detect_error,
    get_error_type,
)
from atomate2.siesta.custodian.errors.patterns import (
    BASIS_PATTERNS,
    FILE_IO_PATTERNS,
    GEOMETRY_PATTERNS,
    GRID_PATTERNS,
    MEMORY_PATTERNS,
    NUMERICAL_PATTERNS,
    PSEUDOPOTENTIAL_PATTERNS,
    SCF_CONVERGENCE_PATTERNS,
    SEGFAULT_PATTERNS,
    SIESTA_ERROR_PATTERNS,
    TIME_LIMIT_PATTERNS,
)

__all__ = [
    "BASIS_PATTERNS",
    "FILE_IO_PATTERNS",
    "GEOMETRY_PATTERNS",
    "GRID_PATTERNS",
    "MEMORY_PATTERNS",
    "NUMERICAL_PATTERNS",
    "PSEUDOPOTENTIAL_PATTERNS",
    "SCF_CONVERGENCE_PATTERNS",
    "SEGFAULT_PATTERNS",
    # Pattern collections
    "SIESTA_ERROR_PATTERNS",
    "TIME_LIMIT_PATTERNS",
    "ErrorPattern",
    "ErrorSeverity",
    # Base types
    "ErrorType",
    "MaxErrorsError",
    "RecoverableError",
    # Exceptions
    "SiestaError",
    "UnrecoverableError",
    "ValidationError",
    "check_for_errors",
    # Detection functions
    "detect_error",
    "get_error_type",
]
