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
    ErrorPattern,
    ErrorSeverity,
    ErrorType,
    MaxErrorsError,
    RecoverableError,
    SIESTA_ERROR_PATTERNS,
    SiestaError,
    UnrecoverableError,
    ValidationError,
    check_for_errors,
    detect_error,
    get_error_type,
)
from atomate2.siesta.custodian.handlers import (
    ErrorHandler,
    SCFConvergenceHandler,
    BasisSetHandler,
    MemoryHandler,
    TimeHandler,
    NumericalHandler,
    DEFAULT_HANDLERS,
)
from atomate2.siesta.custodian.validators import (
    OutputValidator,
    SiestaOutputValidator,
    RelaxationValidator,
    BandStructureValidator,
    get_validator,
)
from atomate2.siesta.custodian.fdf_utils import (
    read_fdf_file,
    update_fdf_file,
    apply_corrections,
    get_fdf_parameter,
)
from atomate2.siesta.custodian.jobs import (
    CustodianJob,
    run_custodian_job,
)

__all__ = [
    # Errors
    "ErrorPattern",
    "ErrorSeverity",
    "ErrorType",
    "MaxErrorsError",
    "RecoverableError",
    "SIESTA_ERROR_PATTERNS",
    "SiestaError",
    "UnrecoverableError",
    "ValidationError",
    "check_for_errors",
    "detect_error",
    "get_error_type",
    # Handlers
    "ErrorHandler",
    "SCFConvergenceHandler",
    "BasisSetHandler",
    "MemoryHandler",
    "TimeHandler",
    "NumericalHandler",
    "DEFAULT_HANDLERS",
    # Validators
    "OutputValidator",
    "SiestaOutputValidator",
    "RelaxationValidator",
    "BandStructureValidator",
    "get_validator",
    # FDF Utils
    "read_fdf_file",
    "update_fdf_file",
    "apply_corrections",
    "get_fdf_parameter",
    # Jobs
    "CustodianJob",
    "run_custodian_job",
]
