"""Basis-related convergence workflows for SIESTA."""

from atomate2.siesta.flows.basis.complete import CompleteBasisConvergenceFlowMaker
from atomate2.siesta.flows.basis.core import DifferentBasisSCFAdvanceFlowMaker
from atomate2.siesta.flows.basis.eos import EOSBasisConvergenceFlowMaker
from atomate2.siesta.flows.basis.parameters import BasisParametersConvergenceFlowMaker
from atomate2.siesta.flows.basis.size import BasisSizeConvergenceFlowMaker

# Backwards compatibility alias
BasisConvergenceFlowMaker = CompleteBasisConvergenceFlowMaker

__all__ = [
    "BasisConvergenceFlowMaker",  # Alias for CompleteBasisConvergenceFlowMaker
    "BasisParametersConvergenceFlowMaker",
    "BasisSizeConvergenceFlowMaker",
    "CompleteBasisConvergenceFlowMaker",
    "DifferentBasisSCFAdvanceFlowMaker",
    "EOSBasisConvergenceFlowMaker",
]
