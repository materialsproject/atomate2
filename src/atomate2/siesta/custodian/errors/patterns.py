"""Error pattern definitions for SIESTA calculations."""

from __future__ import annotations

import re

from atomate2.siesta.custodian.errors.base import (
    ErrorPattern,
    ErrorSeverity,
    ErrorType,
)


# SCF Convergence Errors
SCF_CONVERGENCE_PATTERNS = ErrorPattern(
    error_type=ErrorType.SCF_CONVERGENCE,
    patterns=[
        re.compile(r"SCF_NOT_CONV:\s*SCF did not converge", re.IGNORECASE),
        re.compile(r"SCF cycle not converged", re.IGNORECASE),
        re.compile(r"SCF did not converge", re.IGNORECASE),
        re.compile(r"Maximum number of SCF iterations reached", re.IGNORECASE),
        re.compile(r"scf_not_conv", re.IGNORECASE),
        re.compile(r"SCF Convergence by \w+ criterion:\s*F", re.IGNORECASE),
    ],
    file_to_check="siesta.out",
    severity=ErrorSeverity.RECOVERABLE,
    description="SCF cycle failed to converge within maximum iterations",
)

# Memory Errors
MEMORY_PATTERNS = ErrorPattern(
    error_type=ErrorType.MEMORY,
    patterns=[
        re.compile(r"Out of memory", re.IGNORECASE),
        re.compile(r"Cannot allocate memory", re.IGNORECASE),
        re.compile(r"memory allocation failed", re.IGNORECASE),
        re.compile(r"Allocation would exceed memory limits", re.IGNORECASE),
        re.compile(r"re_alloc: allocation failed", re.IGNORECASE),
        re.compile(r"Out of memory in \w+", re.IGNORECASE),
        re.compile(r"ALLOCATION ERROR", re.IGNORECASE),
    ],
    file_to_check="siesta.out",
    severity=ErrorSeverity.CRITICAL,
    description="Out of memory error during calculation",
)

# Time Limit Errors (from job scheduler)
TIME_LIMIT_PATTERNS = ErrorPattern(
    error_type=ErrorType.TIME_LIMIT,
    patterns=[
        re.compile(r"CANCELLED AT .* DUE TO TIME LIMIT", re.IGNORECASE),
        re.compile(r"walltime .* exceeded", re.IGNORECASE),
        re.compile(r"DUE TO TIME LIMIT", re.IGNORECASE),
        re.compile(r"Job exceeded time limit", re.IGNORECASE),
        re.compile(r"TIME LIMIT", re.IGNORECASE),
    ],
    file_to_check="slurm.out",
    severity=ErrorSeverity.RECOVERABLE,
    description="Job exceeded wall time limit",
)

# Numerical Instability Errors
NUMERICAL_PATTERNS = ErrorPattern(
    error_type=ErrorType.NUMERICAL,
    patterns=[
        re.compile(r"Numerical problem in Cholesky decomposition", re.IGNORECASE),
        re.compile(r"Singular overlap matrix", re.IGNORECASE),
        re.compile(r"Cholesky: singular matrix", re.IGNORECASE),
        re.compile(r"LAPACK error", re.IGNORECASE),
        re.compile(r"NaN detected", re.IGNORECASE),
        re.compile(r"Inf detected", re.IGNORECASE),
        re.compile(r"divide by zero", re.IGNORECASE),
        re.compile(r"Numerical instability", re.IGNORECASE),
    ],
    file_to_check="siesta.out",
    severity=ErrorSeverity.CRITICAL,
    description="Numerical instability or NaN/Inf values detected",
)

# Basis Set Errors
BASIS_PATTERNS = ErrorPattern(
    error_type=ErrorType.BASIS,
    patterns=[
        re.compile(r"Basis orbital extends beyond Vna range", re.IGNORECASE),
        re.compile(r"Orbital is not confined", re.IGNORECASE),
        re.compile(r"Basis set error", re.IGNORECASE),
        re.compile(r"PAO\.EnergyShift too small", re.IGNORECASE),
        re.compile(r"Split_norm too large", re.IGNORECASE),
        re.compile(r"Split-norm parameter is too small.*degenerate", re.IGNORECASE),
        re.compile(r"Basis generation failed", re.IGNORECASE),
    ],
    file_to_check="siesta.out",
    severity=ErrorSeverity.RECOVERABLE,
    description="Basis set generation or confinement error",
)

# Grid/Mesh Errors
GRID_PATTERNS = ErrorPattern(
    error_type=ErrorType.GRID,
    patterns=[
        re.compile(r"Grid dimensions too small", re.IGNORECASE),
        re.compile(r"Mesh cutoff too small", re.IGNORECASE),
        re.compile(r"Grid error", re.IGNORECASE),
        re.compile(r"Real space grid problem", re.IGNORECASE),
    ],
    file_to_check="siesta.out",
    severity=ErrorSeverity.RECOVERABLE,
    description="Real-space grid or mesh cutoff error",
)

# File I/O Errors
FILE_IO_PATTERNS = ErrorPattern(
    error_type=ErrorType.FILE_IO,
    patterns=[
        re.compile(r"Error opening file", re.IGNORECASE),
        re.compile(r"Cannot open file", re.IGNORECASE),
        re.compile(r"File not found", re.IGNORECASE),
        re.compile(r"I/O error", re.IGNORECASE),
        re.compile(r"Read error", re.IGNORECASE),
        re.compile(r"Write error", re.IGNORECASE),
        re.compile(r"Permission denied", re.IGNORECASE),
    ],
    file_to_check="siesta.out",
    severity=ErrorSeverity.CRITICAL,
    description="File input/output error",
)

# Segmentation Fault
SEGFAULT_PATTERNS = ErrorPattern(
    error_type=ErrorType.SEGFAULT,
    patterns=[
        re.compile(r"segmentation fault", re.IGNORECASE),
        re.compile(r"segfault", re.IGNORECASE),
        re.compile(r"SIGSEGV", re.IGNORECASE),
        re.compile(r"signal 11", re.IGNORECASE),
    ],
    file_to_check="siesta.out",
    severity=ErrorSeverity.FATAL,
    description="Segmentation fault (program crash)",
)

# Parallelization Errors
PARALLEL_PATTERNS = ErrorPattern(
    error_type=ErrorType.PARALLEL,
    patterns=[
        re.compile(r"You have too many processors for the system size", re.IGNORECASE),
        re.compile(r"Some processors are idle.*Check PARALLEL_DIST", re.IGNORECASE),
        re.compile(r"Orbital distribution balance.*0$", re.IGNORECASE),
    ],
    file_to_check="siesta.out",
    severity=ErrorSeverity.RECOVERABLE,
    description="Too many processors for system size",
)

# Geometry Errors
GEOMETRY_PATTERNS = ErrorPattern(
    error_type=ErrorType.GEOMETRY,
    patterns=[
        re.compile(r"Geometry optimization failed", re.IGNORECASE),
        re.compile(r"Maximum force exceeded", re.IGNORECASE),
        re.compile(r"Coordinate update failed", re.IGNORECASE),
        re.compile(r"FIRE: Too many steps without moving", re.IGNORECASE),
        re.compile(r"Geometry step failed", re.IGNORECASE),
    ],
    file_to_check="siesta.out",
    severity=ErrorSeverity.RECOVERABLE,
    description="Geometry optimization failure",
)

# Pseudopotential Errors
PSEUDOPOTENTIAL_PATTERNS = ErrorPattern(
    error_type=ErrorType.PSEUDOPOTENTIAL,
    patterns=[
        re.compile(r"Pseudopotential not found", re.IGNORECASE),
        re.compile(r"Cannot read pseudopotential", re.IGNORECASE),
        re.compile(r"PS file error", re.IGNORECASE),
        re.compile(r"Invalid pseudopotential", re.IGNORECASE),
    ],
    file_to_check="siesta.out",
    severity=ErrorSeverity.CRITICAL,
    description="Pseudopotential file error",
)

# Collect all patterns
SIESTA_ERROR_PATTERNS: list[ErrorPattern] = [
    SCF_CONVERGENCE_PATTERNS,
    MEMORY_PATTERNS,
    TIME_LIMIT_PATTERNS,
    NUMERICAL_PATTERNS,
    BASIS_PATTERNS,
    GRID_PATTERNS,
    FILE_IO_PATTERNS,
    SEGFAULT_PATTERNS,
    PARALLEL_PATTERNS,
    GEOMETRY_PATTERNS,
    PSEUDOPOTENTIAL_PATTERNS,
]
