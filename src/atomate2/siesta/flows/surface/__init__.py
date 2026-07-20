"""Surface-related flow modules for SIESTA calculations."""

# Re-export everything for backward compatibility
# This allows: from atomate2.siesta.flows.surface import ...
# Instead of: from atomate2.siesta.flows.surface.core import ...

from atomate2.siesta.flows.surface.adsorption import (
    AdsorptionOptimizationFlowMaker,
    AdsorptionScanFlowMaker,
)
from atomate2.siesta.flows.surface.convergence import SurfaceEnergyConvergenceFlowMaker
from atomate2.siesta.flows.surface.core import SurfaceEnergyFlowMaker
from atomate2.siesta.flows.surface.multi_surface import (
    MultiSurfaceEnergyFlowMaker,
    calculate_multi_surface_energies,
)

__all__ = [
    "AdsorptionOptimizationFlowMaker",
    "AdsorptionScanFlowMaker",
    "MultiSurfaceEnergyFlowMaker",
    "SurfaceEnergyConvergenceFlowMaker",
    "SurfaceEnergyFlowMaker",
    "calculate_multi_surface_energies",
]
