"""Error handlers for SIESTA calculations.

This package provides handler classes for detecting and correcting
common SIESTA calculation errors.
"""

from atomate2.siesta.custodian.handlers.base import ErrorHandler
from atomate2.siesta.custodian.handlers.basis import BasisSetHandler
from atomate2.siesta.custodian.handlers.geometry_convergence import (
    GeometryConvergenceHandler,
)
from atomate2.siesta.custodian.handlers.memory import MemoryHandler
from atomate2.siesta.custodian.handlers.numerical import NumericalHandler
from atomate2.siesta.custodian.handlers.parallel import ParallelDistributionHandler
from atomate2.siesta.custodian.handlers.scf import SCFConvergenceHandler
from atomate2.siesta.custodian.handlers.scf_relaxation import SCFRelaxationHandler
from atomate2.siesta.custodian.handlers.time import TimeHandler

# Default handler list for static/SCF calculations
DEFAULT_HANDLERS = [
    SCFConvergenceHandler(),
    BasisSetHandler(),
    MemoryHandler(),
    ParallelDistributionHandler(),
    TimeHandler(),
    NumericalHandler(),
]

# Default handler list for relaxation calculations
# Uses SCFRelaxationHandler which removes DM and increases SCF.MaxIter
# Uses GeometryConvergenceHandler which increases MD.NumCGsteps if not converged
DEFAULT_RELAXATION_HANDLERS = [
    SCFRelaxationHandler(),  # Handle SCF failures (remove DM, increase SCF.MaxIter)
    GeometryConvergenceHandler(),  # Handle non-converged geometries (increase MD steps)
    BasisSetHandler(),
    MemoryHandler(),
    ParallelDistributionHandler(),  # Handle too many processors for system size
    TimeHandler(),
    NumericalHandler(),
]

__all__ = [
    "ErrorHandler",
    "SCFConvergenceHandler",
    "SCFRelaxationHandler",
    "GeometryConvergenceHandler",
    "BasisSetHandler",
    "MemoryHandler",
    "ParallelDistributionHandler",
    "TimeHandler",
    "NumericalHandler",
    "DEFAULT_HANDLERS",
    "DEFAULT_RELAXATION_HANDLERS",
]
