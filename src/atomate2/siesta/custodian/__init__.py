"""Custodian error handling and recovery system for SIESTA calculations.

This package provides automatic error detection, correction, and recovery
capabilities for SIESTA calculations, similar to Custodian for VASP.

Key Components
--------------
- errors: Error pattern definitions and detection
- handlers: Error handlers with correction strategies
- validators: Output validation
- jobs: Job wrappers with error handling
- fdf_utils: FDF file update utilities

Example
-------
>>> from atomate2.siesta.custodian import run_custodian_job
>>> result = run_custodian_job(
...     "siesta < siesta.fdf > siesta.out",
...     directory="job_001",
...     max_errors=5,
... )
"""

from atomate2.siesta.custodian.errors import (
    SIESTA_ERROR_PATTERNS,
    ErrorPattern,
    ErrorSeverity,
    ErrorType,
    MaxErrorsError,
    RecoverableError,
    SiestaError,
    UnrecoverableError,
    ValidationError,
    check_for_errors,
    detect_error,
    get_error_type,
)
from atomate2.siesta.custodian.fdf_utils import (
    apply_corrections,
    get_fdf_parameter,
    read_fdf_file,
    update_fdf_file,
)
from atomate2.siesta.custodian.handlers import (
    DEFAULT_HANDLERS,
    BasisSetHandler,
    ErrorHandler,
    MemoryHandler,
    NumericalHandler,
    SCFConvergenceHandler,
    TimeHandler,
)
from atomate2.siesta.custodian.jobs import CustodianJob, run_custodian_job
from atomate2.siesta.custodian.validators import (
    BandStructureValidator,
    OutputValidator,
    RelaxationValidator,
    SiestaOutputValidator,
    get_validator,
)

__all__ = [
    "DEFAULT_HANDLERS",
    "SIESTA_ERROR_PATTERNS",
    "BandStructureValidator",
    "BasisSetHandler",
    # Jobs
    "CustodianJob",
    # Handlers
    "ErrorHandler",
    # Errors
    "ErrorPattern",
    "ErrorSeverity",
    "ErrorType",
    "MaxErrorsError",
    "MemoryHandler",
    "NumericalHandler",
    # Validators
    "OutputValidator",
    "RecoverableError",
    "RelaxationValidator",
    "SCFConvergenceHandler",
    "SiestaError",
    "SiestaOutputValidator",
    "TimeHandler",
    "UnrecoverableError",
    "ValidationError",
    "apply_corrections",
    "check_for_errors",
    "detect_error",
    "get_error_type",
    "get_fdf_parameter",
    "get_validator",
    # FDF Utils
    "read_fdf_file",
    "run_custodian_job",
    "update_fdf_file",
]
