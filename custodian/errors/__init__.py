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
    # Base types
    "ErrorType",
    "ErrorSeverity",
    "ErrorPattern",
    # Exceptions
    "SiestaError",
    "RecoverableError",
    "UnrecoverableError",
    "ValidationError",
    "MaxErrorsError",
    # Detection functions
    "detect_error",
    "get_error_type",
    "check_for_errors",
    # Pattern collections
    "SIESTA_ERROR_PATTERNS",
    "SCF_CONVERGENCE_PATTERNS",
    "MEMORY_PATTERNS",
    "TIME_LIMIT_PATTERNS",
    "NUMERICAL_PATTERNS",
    "BASIS_PATTERNS",
    "GRID_PATTERNS",
    "FILE_IO_PATTERNS",
    "SEGFAULT_PATTERNS",
    "GEOMETRY_PATTERNS",
    "PSEUDOPOTENTIAL_PATTERNS",
]
